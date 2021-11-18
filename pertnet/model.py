import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

from torch_geometric.nn import GINConv, GCNConv, GATConv, GraphConv, SGConv
from torch.nn import Sequential, Linear, ReLU, LayerNorm, PReLU
import pandas as pd

import sys

class No_Perturb(torch.nn.Module):
    """
    No Perturbation
    """

    def __init__(self):
        super(No_Perturb, self).__init__()        

    def forward(self, data):
        
        x = data.x
        x = x[:, 0].reshape(*data.y.shape)
        
        return x

class MLP(torch.nn.Module):

    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 1 else None,
                torch.nn.ReLU()
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        self.network = torch.nn.Sequential(*layers)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        if self.activation == "ReLU":
            return self.relu(x)
        else:
	        return self.network(x)


class PertNet(torch.nn.Module):
    """
    PertNet
    """

    def __init__(self, args):
        super(PertNet, self).__init__()

        self.num_genes = args['num_genes']
        hidden_size = args['node_hidden_size']
        self.uncertainty = args['uncertainty']
        self.num_layers = args['num_of_gnn_layers']
        self.args = args

        # perturbation similarity network
        self.G_sim = args['G_sim'].to(args['device'])
        self.G_sim_weight = args['G_sim_weight'].to(args['device'])

        # lambda for aggregation between global perturbation emb + gene embedding
        self.pert_emb_lambda = args['pert_emb_lambda']
        
        # perturbation positional embedding added only to the perturbed genes
        self.pert_w = nn.Linear(1, hidden_size)

        # gene basal embedding - encoding gene expression
        self.gene_basal_w = nn.Linear(1, hidden_size)
        
        # gene/globel perturbation embedding dictionary lookup            
        self.gene_emb = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.pert_emb = nn.Embedding(self.num_genes, hidden_size, max_norm=True)

        # transformation layer
        self.emb_trans = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        self.pert_base_trans_w = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        self.gene_base_trans_w = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        self.transform = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        self.recovery_w = MLP([hidden_size, hidden_size*2, hidden_size, 1], last_layer_act='linear')  
        
        ### perturbation embedding similarity
        self.sim_layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            self.sim_layers.append(SGConv(hidden_size, hidden_size, 1))

        # batchnorms
        self.bn_pert_emb = nn.BatchNorm1d(hidden_size)
        self.bn_emb = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base_trans = nn.BatchNorm1d(hidden_size)
        self.bn_final = nn.BatchNorm1d(hidden_size)
        self.bn_gene_base_trans = nn.BatchNorm1d(hidden_size)
        self.bn_gene_base = nn.BatchNorm1d(hidden_size)
        self.bn_post_gnn = nn.BatchNorm1d(hidden_size)
        self.bn_base_emb = nn.BatchNorm1d(hidden_size)

        # uncertainty mode
        if self.uncertainty:
            self.uncertainty_w = MLP([hidden_size, hidden_size*2, hidden_size, 1], last_layer_act='linear')
        

    def forward(self, data):
        x, batch = data.x, data.batch
        num_graphs = len(data.batch.unique())
        
        ## get base gene embeddings
        emb = self.gene_emb(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))        
        emb = self.bn_emb(emb)
        base_emb = self.emb_trans(emb)
        #base_emb = self.bn_base_emb(base_emb)

        ## get perturbation index and embeddings
        pert = x[:, 1].reshape(-1,1)
        pert_index = torch.where(pert.reshape(num_graphs, int(x.shape[0]/num_graphs)) == 1)
        pert_global_emb = self.pert_emb(torch.LongTensor(list(range(self.num_genes))).to(self.args['device']))
        #pert_global_emb = self.bn_pert_emb(pert_global_emb)

        ## augment global perturbation embedding with GNN
        for idx, layer in enumerate(self.sim_layers):
            pert_global_emb = layer(pert_global_emb, self.G_sim, self.G_sim_weight)
            if idx < self.num_layers - 1:
                pert_global_emb = pert_global_emb.relu()
        #pert_global_emb = self.bn_post_gnn(pert_global_emb)

        ## add global perturbation embedding to each gene in each cell in the batch
        base_emb = base_emb.reshape(num_graphs, self.num_genes, -1)
        for i, j in enumerate(pert_index[0]):
            lambda_i = self.pert_emb_lambda 
            base_emb[j] += lambda_i * pert_global_emb[pert_index[1][i]]
        base_emb = base_emb.reshape(num_graphs * self.num_genes, -1)
        
        '''
        ## add the gene expression positional embedding
        gene_base = x[:, 0].reshape(-1,1)
        gene_emb = self.gene_basal_w(gene_base)
        combined = gene_emb+base_emb
        combined = self.bn_gene_base_trans(combined)
        base_emb = self.gene_base_trans_w(combined)
        base_emb = self.bn_gene_base(base_emb)
        '''
        ## add the perturbation positional embedding
        pert_emb = self.pert_w(pert)
        combined = pert_emb+base_emb
        combined = self.bn_pert_base_trans(combined)
        base_emb = self.pert_base_trans_w(combined)
        base_emb = self.bn_pert_base(base_emb)
        
        ## apply the first MLP
        base_emb = self.transform(base_emb)
        #base_emb = self.bn_final(base_emb)

        ## apply the final MLP to predict delta only and then add back the x. 
        out = self.recovery_w(base_emb) + x[:, 0].reshape(-1,1)
        out = torch.split(torch.flatten(out), self.num_genes)

        ## uncertainty head
        if self.uncertainty:
            out_logvar = self.uncertainty_w(base_emb)
            out_logvar = torch.split(torch.flatten(out_logvar), self.num_genes)
            return torch.stack(out), torch.stack(out_logvar)
        
        return torch.stack(out)        
        