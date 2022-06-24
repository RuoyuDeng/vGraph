from __future__ import division
from __future__ import print_function

from data_utils import load_cora_citeseer, load_webkb
from sklearn.cluster import KMeans
from subprocess import check_output
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
from score_utils import calc_nonoverlap_nmi
import argparse
import collections
import community
import math
import networkx as nx
import numpy as np
import re
import scipy.sparse as sp
import time
import torch
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from IPython import embed


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='vGraph', help="models used")
parser.add_argument('--lamda', type=float, default=0, help="")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5001, help='Number of epochs to train.')
parser.add_argument('--embedding-dim', type=int, default=128, help='')
parser.add_argument('--lr', type=float, default=0.05, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')
parser.add_argument('--log-file', type=str, default='nonoverlapping.log', help='log path')
# these files are only used to do community detection, no need to put a option for switching tasks
# parser.add_argument('--task', type=str, default='community', help='type of dataset.') 

def logging(args, epochs, nmi, modularity):
    with open(args.log_file, 'a+') as f:
        f.write('{},{},{},{},{},{},{},{}\n'.format('vGraph',
            args.dataset_str,
            args.lr,
            args.embedding_dim, args.lamda, epochs, nmi, modularity))

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
    model.eval() # change the model to evaluation mode
    edges = [(u,v) for u,v in G.edges()]
    batch = torch.LongTensor(edges)

    # model(listof_u_nodes, listof_v_nodes, 1) -> evaluate which community does (u,v) belong to
    _, q, _ = model(batch[:, 0], batch[:, 1], 1.)

    num_classes = q.shape[1]
    q_argmax = q.argmax(dim=-1)

    assignment = {}

    n_nodes = G.number_of_nodes()

    res = np.zeros((n_nodes, num_classes))
    for idx, e in enumerate(edges):
        if tpe == 0:
            # when idx = 0, we update the prob of which community e_0 = (u_0, v_0) belongs to by adding all prob
            # of diff communities to current values
            # e is different in every iteration!!!
            res[e[0], :] += q[idx, :].cpu().data.numpy()
            res[e[1], :] += q[idx, :].cpu().data.numpy()
        else:
            res[e[0], q_argmax[idx]] += 1
            res[e[1], q_argmax[idx]] += 1

    res = res.argmax(axis=-1) # res is then our prediction after argmax (np.array)
    assignment = {i : res[i] for i in range(res.shape[0])} # dict version of our prdiction, key:node, value: predict_class
    
    return res, assignment

def classical_modularity_calculator(graph, embedding, model='gcn_vae', cluster_number=5):
    """
    Function to calculate the DeepWalk cluster centers and assignments.
    """    
    if model == 'gcn_vae':
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
        # size: number of nodes
        super(GCNModelGumbel, self).__init__()
        self.embedding_dim = embedding_dim
        self.categorical_dim = categorical_dim
        self.device = device
        self.size = size

        # initialize the layers for the model

        # nn.Linear has shape (128,7), then its weight matrix has shape (7,128)
        self.community_embeddings = nn.Linear(embedding_dim, categorical_dim, bias=False).to(device)
        self.node_embeddings = nn.Embedding(size, embedding_dim)
        self.contextnode_embeddings = nn.Embedding(size, embedding_dim)

        self.decoder = nn.Sequential(
          nn.Linear( embedding_dim, size),
        ).to(device)

        self.init_emb() # initialize embeddings

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, w, c, temp):

        # TODO: figure out how forward works


        # w and c are mapped edge (u,v) representations stored in tensor
        # change to CUDA tensors based on self.device

        # input_w and input_c have same shape: [2 * numof_nodes]
        # w -> [2 * numof_nodes, embed_dim]
        # c -> [2 * numof_nodes, embed_dim]
        w = self.node_embeddings(w).to(self.device)
        c = self.node_embeddings(c).to(self.device)

        # community_embedding works as a linear map L: input(A,B) -> output(B,C)
        # maps [2 * numof_nodes, embed_dim] -> [2 * numof_nodes, catg_classes]
        # q.shape: [batch_size, categorical_dim], after community_embedding map
        q = self.community_embeddings(w*c)
        
        # z = self._sample_discrete(q, temp)

        # once called model.train(), self.training is set to be True
        if self.training:
            z = F.gumbel_softmax(logits=q, tau=temp, hard=True)
        else:
            tmp = q.argmax(dim=-1).reshape(q.shape[0], 1)
            z = torch.zeros(q.shape).to(self.device).scatter_(1, tmp, 1.)

        # prior probability generated based on input embeddings of all nodes
        # shape: [batch_size, catogorical_dim]
        prior = self.community_embeddings(w)
        prior = F.softmax(prior, dim=-1)
        # prior.shape [batch_num_nodes, 

        # z.shape [batch_size, categorical_dim]
        # self.community_embeddings.weight.shape [cat_dim, embed_dim]
        new_z = torch.mm(z, self.community_embeddings.weight)

        # decoder: decode the embedding back to node shape
        # [batch_size, embed_dim] -decode-> [batch_size, node_num]
        # decoder has a linear map of [embed_dim,node_num], map each row back to 
        # their node_num dimmensions
        recon = self.decoder(new_z)
        
        # return
        # recon: the reconstructed tensor after decoding
        # F.softmax(q,dim=-1): the prob dist of nodes inputed (w and c)
        return recon, F.softmax(q, dim=-1), prior


if __name__ == '__main__':
    args = parser.parse_args()
    embedding_dim = args.embedding_dim
    lr = args.lr
    epochs = args.epochs
    temp = 1.
    temp_min = 0.3
    ANNEAL_RATE = 0.00003
    

    # In[13]:
    if args.dataset_str in ['cora', 'citeseer']:
        G, adj, gt_membership = load_cora_citeseer(args.dataset_str)
    else:
        G, adj, gt_membership = load_webkb(args.dataset_str)

    
    adj_orig = adj
    # offset = [0], means that we only fill the diagonal, offset[i] is the index of the diagonal，对角线为index 0,
    # 对角线往下一行的对角线为index: -1,above is 1, 
    # the adjacent matrix minus the diagonal to remove all self loop of each node
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    
    adj_orig.eliminate_zeros()
    categorical_dim = len(set(gt_membership))
    n_nodes = G.number_of_nodes()
    print(n_nodes, categorical_dim)
    # 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GCNModelGumbel(adj.shape[0], embedding_dim, categorical_dim, args.dropout, device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    hidden_emb = None
    history_valap = []
    history_mod = []

    train_edges = [(u,v) for u,v in G.edges()]
    n_nodes = G.number_of_nodes()
    print('len(train_edges)', len(train_edges))
    for epoch in range(epochs):

        t = time.time()
        # transform list into tensor (fancy term for matrix)
        batch = torch.LongTensor(train_edges)
        assert batch.shape == (len(train_edges), 2)

        # set model to training mode and clear previous gradient
        model.train()
        optimizer.zero_grad()

        # FIXME: Why making w and c: the nodes arranged in 2 diff orders? (edges have 2 directions?)
        # w.shape: [10556] 
        # c.shape: [10556]
        # batch.shape: [5278,2] -> [numof_train_edges, 2: the number of nodes for 1 edge]
        w = torch.cat((batch[:, 0], batch[:, 1]))
        c = torch.cat((batch[:, 1], batch[:, 0]))
        # embed()
        # we do not call model.forwad() directly to pass the arguments!
        # temp = 1 initalliy
        recon, q, prior = model(w, c, temp)

        res = torch.zeros([n_nodes, categorical_dim], dtype=torch.float32).to(device)
        for idx, e in enumerate(train_edges):
            res[e[0], :] += q[idx, :]
            res[e[1], :] += q[idx, :]
        smoothing_loss = args.lamda * ((res[w] - res[c])**2).mean()

        loss = loss_function(recon, q, prior, c.to(device), None, None)
        loss += smoothing_loss

        loss.backward() # Computes the gradient of current tensor w.r.t. graph leaves.
        cur_loss = loss.item() # get the number stored in the tensor
        optimizer.step() # optimizer takes a next step
        
       
        if epoch % 100 == 0:
            
            temp = np.maximum(temp*np.exp(-ANNEAL_RATE*epoch),temp_min)
            model.eval()
            
            # membership and assignment are same prediction result stored in diff data type
            # which are used to compute modularity and nmi
            membership, assignment = get_assignment(G, model, categorical_dim)
            #print([(membership == i).sum() for i in range(categorical_dim)])
            #print([(np.array(gt_membership) == i).sum() for i in range(categorical_dim)])
            modularity = classical_modularity_calculator(G, assignment)
            nmi = calc_nonoverlap_nmi(membership.tolist(), gt_membership)
            
            print('Epoch', epoch, "lr", lr, 'nmi', nmi, 'modularity', modularity)
            logging(args, epoch, nmi, modularity)

            ########## added by Rouyu
            lr = lr * 0.99
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            ##########  
            
            

            
            
    print("Optimization Finished!")
