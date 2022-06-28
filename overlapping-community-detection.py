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
import numpy as np
import numpy as np
import numpy as np
import re
import scipy.sparse as sp
import sys
import time
import torch
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

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

        #self.contextnode_embeddings = nn.Embedding(size, embedding_dim)

        self.decoder = nn.Sequential(
          nn.Linear( embedding_dim, size),
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

        # we learn from every edge, thus w,c,r has shape [num_of_edges, embedding_dim]

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

        # w: N x ED, c, N x ED, -> N x N (dot product)
        wT = torch.transpose(w,0,1)
        prob = torch.sigmoid(torch.mm(c,wT))

        # z.shape [batch_size, categorical_dim]
        new_z = torch.mm(z, self.community_embeddings.weight)

        # decoder needs to output 2 embeddings: 1. R x K, 2. V x K, where 
        # R: num of relations
        # K: embedding dimension
        # V: number of nodes
        recon = self.decoder(new_z) 
            
        return recon, F.softmax(q, dim=-1), prior, prob

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


if __name__ == '__main__': # execute the following if current file is exectued through cmd line: python3 current_file_name.py [-args]
    args = parser.parse_args() 
    embedding_dim = args.embedding_dim # automattically change the option name from "embedding-dim" into a attribute embedding_dim
    lr = args.lr
    epochs = args.epochs
    temp = 1.
    temp_min = 0.1
    ANNEAL_RATE = 0.00003
    torch.manual_seed(2022)
   
    # randomly assign some communities number
    G, adj, gt_communities = load_dataset(args.dataset_str)

    # adj_orig is the sparse matrix of edge_node adjacency matrix
    adj_orig = adj
    
    
    # adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    categorical_dim = len(gt_communities)
    n_nodes = G.number_of_nodes()
    print(n_nodes, categorical_dim)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = GCNModelGumbel(adj.shape[0], embedding_dim, categorical_dim, args.dropout, device)
    optimizer = optim.Adam(model.parameters(), lr=lr) # change optmizer

    hidden_emb = None
    history_valap = []
    history_mod = []

    # FIXME: In the original code, len(train_edges) = num_nodes because there is only ONE edge between each pair of nodes
    # however, in our case, there can be multiple edges between nodes.
    # need to change from [(u,v0), (u,v1), (u,v2) ...] -> (u,[v0,v1,v2...])

    #train_edges = np.concatenate([train_edges, val_edges, test_edges])
    # dict -> u: [v0, v1, v2....], how many edges every u has, len(train_edges) = num_of_nodes
    train_edges = {}
    for u,v in G.edges():
        if u in train_edges.keys():
            train_edges[u].append(v)
        else:
            train_edges[u] = [v]
        
    all_edges = [(u,v) for u,v in G.edges()]

    # dict -> u: [r0, r1, r2....], how many diff types of relations every u has, len(train_relations) = num_of_nodes
    train_relations = {}
    for u,v,c in G.edges:
        if u in train_edges.keys():
            relation = G.edges[u,v,c]["relation"]
            if u in train_relations.keys():
                if relation not in train_relations[u]:
                    train_relations[u].append(relation)
            else:
                train_relations[u] = [relation]
    
    # train_relations = [G.edges[u,v,c]["relation"] for u,v,c in G.edges]

    n_nodes = G.number_of_nodes()
    print('len(train_edges)', len(train_edges))
    print('calculating normalized_overlap')
    # overlap: the alpha, regularization weight in the paper
    # overlap = for every edge (u,v) in graph G, 
    # = the number of intersection of neighbours of u and v / the number of union of neighbours of u and v
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
        
        # FIXME: what exactly is our batch?
        # the original paper propose: batch = [(u0,v0), (u1,v1)....], and it implicitly included the edge information
        # but we now have: E = {u0:[v0,v1,v2...], u1: [v3,v5,....]}
        # and a relation: R = {u0:[r1,r3,...], u1: [r0...]}, |E| = |R|
        # how to decide the batch for nodes and relations?

        # batch = torch.LongTensor(train_edges)
        # batch_r = torch.LongTensor(train_relations)
        # assert batch.shape == (len(train_edges), 2)
        # assert batch_r.shape[0] == len(train_relations)
        batch = torch.LongTensor(list(train_edges.keys()))
        batch_r = batch
        model.train()
        optimizer.zero_grad()
        
        # everytime, train from 5000 edges randomly sampled from all edges, along with the corresponding relations
        # rand_idx = torch.randperm(batch.size(0))[:5000]
        # batch = batch[rand_idx]
        # w = batch[:,0]
        # c = batch[:,1]
        # r = batch_r[rand_idx]
        w = batch
        c = batch
        r = batch_r
        
        recon, q, prior, link_prob = model(w, c, r, temp)
        loss = loss_function(recon, q, prior, c.to(device), None, None)

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
            # 1. Topic Quality -> need to know the true labels?
            # 2. Link Prediction -> sigmoid(embedding(u) dot_product embedding(v)) = the prob distribution between u and v

            
            # assignment = get_assignment(G, model, categorical_dim)
            # modularity = classical_modularity_calculator(G, assignment)
            
            # # communities: group the nodes with same community in the same list,
            # # it holds all these lists
            # communities = get_overlapping_community(G, model)
            # # nmi = calc_overlap_nmi(n_nodes, communities, gt_communities)
            # f1 = calc_f1(n_nodes, communities, gt_communities)
            # jaccard = calc_jaccard(n_nodes, communities, gt_communities)
            # # omega = calc_omega(n_nodes, communities, gt_communities)

            # nmi = 0
            # omega = 0

            # if args.lamda > 0:
            #     print("Epoch:", '%04d' % (epoch + 1),
            #                   "lr:", '{:.5f}'.format(cur_lr),
            #                   "temp:", '{:.5f}'.format(temp),
            #                   "train_loss=", "{:.5f}".format(cur_loss),
            #                   "smoothing_loss=", "{:.5f}".format(args.lamda * smoothing_loss.item()),
            #                   "modularity=", "{:.5f}".format(modularity),
            #                   "nmi", nmi, "f1", f1, 'jaccard', jaccard, "omega", omega)
            # else:
            #     print("Epoch:", '%04d' % (epoch + 1),
            #                   "lr:", '{:.5f}'.format(cur_lr),
            #                   "temp:", '{:.5f}'.format(temp),
            #                   "train_loss=", "{:.5f}".format(cur_loss),
            #                   "modularity=", "{:.5f}".format(modularity),
            #                   "nmi", nmi, "f1", f1, 'jaccard', jaccard, "omega", omega)
            # logging(args, epoch, cur_loss, f1, nmi, jaccard, modularity)

            # cur_lr = cur_lr * .95
            cur_lr = cur_lr * .99
            for param_group in optimizer.param_groups:
                param_group['lr'] = cur_lr
            
    print("Optimization Finished!")
