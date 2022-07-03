from __future__ import division
from __future__ import print_function
from data_utils import load_dataset
from score_utils import calc_f1, calc_overlap_nmi, calc_jaccard, calc_omega
from score_utils import normalized_overlap
from sklearn.cluster import KMeans 
from subprocess import check_output
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
import argparse
import collections
import community
import math
import networkx as nx
import numpy as np
import re
import scipy.sparse as sp
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from IPython import embed

## 
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='vGraph', help="models used")
parser.add_argument('--lamda', type=float, default=0, help="")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5001, help='Number of epochs to train.')
parser.add_argument('--embedding-dim', type=int, default=128, help='')
parser.add_argument('--lr', type=float, default=0.05, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='FB15k-237', help='type of dataset.')
parser.add_argument('--log-file', type=str, default='overlapping.log', help='log path')
# parser.add_argument('--task', type=str, default='community', help='type of dataset.')

def logging(args, epochs, cur_loss, f1, nmi, jaccard, modularity):
    with open(args.log_file, 'a+') as f:
        # f.write('{},{},{},{},{},{},{},{},{},{}\n'.format('vGraph',
        #     args.dataset_str,
        #     args.lr,
        #     cur_loss, args.lamda, epochs,  nmi, modularity, f1, jaccard))
        f.write('{},{},{},{},{},{},{}\n'.format('vGraph',
            args.dataset_str,
            args.lr, args.lamda, 
            epochs, f1, jaccard))

def write_to_file(fpath, clist):
    with open(fpath, 'w') as f:
        for c in clist:
            f.write(' '.join(map(str, c)) + '\n')

def preprocess(fpath): 
    clist = []
    with open(fpath, 'rb') as f:
        for line in f:
            tmp = re.split(b' |\t', line.strip())[1:]
            clist.append([x.decode('utf-8') for x in tmp])
    
    write_to_file(fpath, clist)
            
def get_assignment(G, model, num_classes=5, tpe=0):
    model.eval()
    edges = [(u,v) for u,v in G.edges()]
    batch = torch.LongTensor(edges)
    _, q, _ = model(batch[:, 0], batch[:, 1], 1.)

    num_classes = q.shape[1]
    q_argmax = q.argmax(dim=-1)

    assignment = {}

    n_nodes = G.number_of_nodes()

    res = np.zeros((n_nodes, num_classes))
    for idx, e in enumerate(edges):
        if tpe == 0:
            res[e[0], :] += q[idx, :].cpu().data.numpy()
            res[e[1], :] += q[idx, :].cpu().data.numpy()
        else:
            res[e[0], q_argmax[idx]] += 1
            res[e[1], q_argmax[idx]] += 1

    res = res.argmax(axis=-1)
    assignment = {i : res[i] for i in range(res.shape[0])}
    return assignment

def classical_modularity_calculator(graph, embedding, model='vGraph', cluster_number=5):
    """
    Function to calculate the DeepWalk cluster centers and assignments.
    """    
    if model == 'vGraph':
        assignments = embedding
    else:
        kmeans = KMeans(n_clusters=cluster_number, random_state=0, n_init = 1).fit(embedding)
        assignments = {i: int(kmeans.labels_[i]) for i in range(0, embedding.shape[0])}

    modularity = community.modularity(assignments, graph)
    return modularity

def loss_function(recon_c, q_y, prior, c, norm=None, pos_weight=None):
    BCE = F.cross_entropy(recon_c, c, reduction='sum') / c.shape[0]
    # BCE = F.binary_cross_entropy_with_logits(recon_c, c, pos_weight=pos_weight)
    # return BCE

    log_qy = torch.log(q_y  + 1e-20)
    KLD = torch.sum(q_y*(log_qy - torch.log(prior)),dim=-1).mean()

    ent = (- torch.log(q_y) * q_y).sum(dim=-1).mean()
    return BCE + KLD

class GCNModelGumbel(nn.Module):
    def __init__(self, size, embedding_dim, categorical_dim, dropout, device):
        super(GCNModelGumbel, self).__init__()
        self.embedding_dim = embedding_dim
        self.categorical_dim = categorical_dim
        self.device = device
        self.size = size

        self.community_embeddings = nn.Linear(embedding_dim, categorical_dim, bias=False).to(device)
        self.node_embeddings = nn.Embedding(size, embedding_dim)
        self.relation_embeddings = nn.Embedding(size, embedding_dim)
        

        self.decoder = nn.Sequential(
          nn.Linear(embedding_dim, size),
        ).to(device)

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, w, c, r, temp):
    
        w = self.node_embeddings(w).to(self.device)
        c = self.node_embeddings(c).to(self.device)
        r = self.relation_embeddings(r).to(self.device)

        q = self.community_embeddings(w*c*r) # w * r * c
        # q.shape: [batch_size, categorical_dim]
        # z = self._sample_discrete(q, temp)
        if self.training:
            z = F.gumbel_softmax(logits=q, tau=temp, hard=True)
        else:
            tmp = q.argmax(dim=-1).reshape(q.shape[0], 1)
            z = torch.zeros(q.shape).to(self.device).scatter_(1, tmp, 1.)

        prior = self.community_embeddings(w)
        prior = F.softmax(prior, dim=-1)

        # z.shape: [batch_size, categorical_dim] -> (r,v) where r.shape: [R, K], v.shape: [V, K]
        # V: number of nodes
        # R: number of relations
        # K: dim of embedding
        new_z = torch.mm(z, self.community_embeddings.weight)
        # TODO: configure the decoder
        recon = self.decoder(new_z) 
        return recon, F.softmax(q, dim=-1), prior

def get_overlapping_community(G, model, tpe=1):
    model.eval()
    edges = [(u,v) for u,v in G.edges()]
    batch = torch.LongTensor(edges)
    _, q, _ = model(batch[:, 0], batch[:, 1], 1.)

    num_classes = q.shape[1]
    q_argmax = q.argmax(dim=-1)

    assignment = {}

    n_nodes = G.number_of_nodes()

    res = np.zeros((n_nodes, num_classes))
    for idx, e in enumerate(edges):
        if tpe == 0:
            res[e[0], :] += q[idx, :].cpu().data.numpy()
            res[e[1], :] += q[idx, :].cpu().data.numpy()
        else:
            # if tpe = 1, then we have:
            # res[i,j] indicating node u_i belong to community_j or not
            # if res[i,j] > 0, then yes, otherwise it does not belong to
            res[e[0], q_argmax[idx]] += 1
            res[e[1], q_argmax[idx]] += 1

    communities = [[] for _ in range(num_classes)]
    for i in range(n_nodes):
        for j in range(num_classes):
            if res[i, j] > 0:
                communities[j].append(i)

    # communities contain num_classes of list, each contains the node
    # that belong to that class (community in this case)
    return communities

class GraphDataSet(Dataset):
    def __init__(self,edges, relations):
        self.edges = edges
        self.relations = relations
    
    def __len__(self):
        return len(self.edges)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        u = self.edges[idx][0]
        v = self.edges[idx][1]
        r = self.relations[idx]
        return u,v,r # determine the tensor order in DataLoader iterable


if __name__ == '__main__': # execute the following if current file is exectued through cmd line: python3 current_file_name.py [-args]
    args = parser.parse_args() 
    embedding_dim = args.embedding_dim # automattically change the option name from "embedding-dim" into a attribute embedding_dim
    lr = args.lr
    epochs = args.epochs
    temp = 1.
    temp_min = 0.1
    ANNEAL_RATE = 0.00003
    batch_size = 5000
    categorical_dim = 12  # decide the cat dim
    torch.manual_seed(2022)
   
    G, adj, gt_communities = load_dataset(args.dataset_str)

    # adj_orig is the sparse matrix of edge_node adjacency matrix
    adj_orig = adj
    
    # adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    n_nodes = G.number_of_nodes()
    print(n_nodes, categorical_dim)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = GCNModelGumbel(adj.shape[0], embedding_dim, categorical_dim, args.dropout, device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    hidden_emb = None
    history_valap = []
    history_mod = []

    
    train_edges = [(u,v) for u,v in G.edges()]
    train_relations = [G.edges[u,v,c]["relation"] for u,v,c in G.edges]
    train_dataset = GraphDataSet(edges=train_edges, relations=train_relations)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size)

    
    n_nodes = G.number_of_nodes()
    print('len(train_edges)', len(train_edges))
    # print('calculating normalized_overlap')
    # overlap: the alpha, regularization weight in the paper
    # overlap = for every edge (u,v) in graph G = the number of intersection of neighbours of u and v / the number of union of neighbours of u and v
    # FIXME: how to calculate overlap can be hard to tell, because there is more than 1 edge between u and v
    # how to make it same shape as len(train_edges) (num of nodes)
    # overlap = torch.Tensor([normalized_overlap(G,u,v) for u,v in all_edges]).to(device)
    overlap = 0

    # overlap = torch.Tensor([(G.degree(u)-G.degree(v))**2 for u,v in train_edges]).to(device)
    # overlap = torch.Tensor([1. for u,v in train_edges]).to(device)
    # overlap = torch.Tensor([float(max(G.degree(u), G.degree(v))**2) for u,v in train_edges]).to(device)
    cur_lr = args.lr

    for epoch in range(epochs):
        #np.random.shuffle(train_edges)

        t = time.time()
        
        # FIXME: what exactly is our batch? -> every edge triple (w,c,r)
        batch = torch.LongTensor(train_edges)
        batch_r = torch.LongTensor(train_relations)
        assert batch.shape == (len(train_edges), 2)
        assert batch_r.shape[0] == len(train_relations)


        # every iterable in train_dataloader is a list of 3 tensors, every such list has len <= batch_size
        # list[0] -> the tensor stores mapped node w
        # list[1] -> the tensor stores mapped node v
        # list[2] -> the tensor stores mapped relation r
        # such order is determined in __getitem__ of our own dataset
        for edge in train_dataloader:
            w = edge[0]
            c = edge[1]
            r = edge[2]
            # turn on the training mode
            model.train()
            optimizer.zero_grad()
            # TODO (Done): training with mini batch
            recon, q, prior = model(w, c, r, temp)
            loss = loss_function(recon, q, prior, c.to(device), None, None)

        
        # pass the edge (w,c,r) to the model to train, w,c are nodes, r is the relation
        # w = batch[:,0]
        # c = batch[:,1]
        # r = batch_r
        

        if args.lamda > 0:
            tmp_w, tmp_c = batch[:, 0], batch[:, 1]
            res = torch.zeros([n_nodes, categorical_dim], dtype=torch.float32, requires_grad=True).to(device)
            for idx, e in enumerate(train_edges):
                res[e[0], :] += q[idx, :]
                res[e[1], :] += q[idx, :]
                #res[e[0], :] += q[idx, :]/G.degree(e[0])
                #res[e[1], :] += q[idx, :]/G.degree(e[1])
            # res /= res.sum(dim=-1).unsqueeze(-1).detach()
            # tmp = F.mse_loss(res[tmp_w], res[tmp_c])

            # tmp: the distance between two distributions (squared difference in our experiments)
            # tmp = avg(sum([p(z|c) - p(z|w)]^2)), such term exits for every pair of edge (u,v)
            tmp = ((res[tmp_w] - res[tmp_c])**2).mean(dim=-1)
            assert overlap.shape == tmp.shape
            smoothing_loss = (overlap*tmp).mean()
            # regularization term = lamda (or called overlap) * tmp
            loss += args.lamda * smoothing_loss
        loss.backward()
        cur_loss = loss.item()

        optimizer.step()

        if np.isnan(loss.item()):
           break
        
        if epoch % 10 == 0:
            temp = np.maximum(temp*np.exp(-ANNEAL_RATE*epoch),temp_min)

        if epoch % 100 == 0:
            
            model.eval()
            
            # TODO: implement our own performance measure metric
            # 1. Topic Quality -> need to know the true labels? NO!
            #                  -> ETM model
            # 2. Link Prediction -> sigmoid(embedding(u) dot_product embedding(v)) = the prob distribution between u and v
            #                    -> CompGCN

        

            # cur_lr = cur_lr * .95
            cur_lr = cur_lr * .99
            for param_group in optimizer.param_groups:
                param_group['lr'] = cur_lr
            
    print("Optimization Finished!")
