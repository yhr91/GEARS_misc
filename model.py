import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch.optim as optim
import matplotlib.pyplot as plt
import networkx as nx
import torch_geometric.nn as pyg_nn
import scanpy as sc

from copy import deepcopy
from torch_geometric.nn import GINConv
from torch.utils.data import DataLoader
from torch_geometric.datasets import TUDataset
from sklearn.metrics import *
from torch.nn import Sequential, Linear, ReLU
from deepsnap.dataset import GraphDataset, Graph
from deepsnap.batch import Batch
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data
import pandas as pd

import sys
sys.path.append('/dfs/user/yhr/cell_reprogram/model/')
from flow import get_graph, get_expression_data,\
            add_weight, I_TF, get_TFs, solve,\
            solve_parallel, get_expression_lambda


# Create adjacency matrix for computation

class linear_model():
    def __init__(self, args):
        self.TFs = get_TFs(args['species'])

        # Set up graph structure
        G_df = get_graph(name = args['regulon_name'],
                           TF_only=False)
        print('Edges: '+str(len(G_df)))
        self.G = nx.from_pandas_edgelist(G_df, source=0,
                            target=1, create_using=nx.DiGraph())

        # Add edge weights
        self.read_weights = pd.read_csv(args['adjacency'] , index_col=0)
        try:
            self.read_weights = self.read_weights.set_index('TF')
        except:
            pass

        # Get adjacency matrix
        # self.adj_mat = self.create_adj_mat()
    
    
    def create_adj_mat(self):
        # Create a df version of the graph for merging
        G_df = pd.DataFrame(self.G.edges(), columns=['TF', 'target'])

        # Merge it with the weights DF
        weighted_G_df = self.read_weights.merge(G_df, on=['TF', 'target'])
        for w in weighted_G_df.iterrows():
            add_weight(self.G, w[1]['TF'], w[1]['target'], w[1]['importance'])

        # Get an adjacency matrix based on the gene ordering from the DE list
        return nx.linalg.graphmatrix.adjacency_matrix(
            self.G, nodelist=args['gene_list']).todense()
    
    
class MLP(torch.nn.Module):
    """
    A multilayer perceptron with ReLU activations and optional BatchNorm.
    """

    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 2 else None,
                torch.nn.ReLU()
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        if self.activation == "linear":
            pass
        elif self.activation == "ReLU":
            self.relu = torch.nn.ReLU()
        else:
            raise ValueError("last_layer_act must be one of 'linear' or 'ReLU'")

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        if self.activation == "ReLU":
           x = self.network(x)
           dim = x.size(1) // 2
           return torch.cat((self.relu(x[:, :dim]), x[:, dim:]), dim=1)
        return self.network(x)
    
    
class GIN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, args):
        super(GIN, self).__init__()
        self.num_layers = args["num_layers"]

        self.pre_mp = nn.Linear(input_size, hidden_size)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for l in range(self.num_layers):
            layer = Sequential(
                Linear(hidden_size, hidden_size), 
                nn.LeakyReLU(), 
                Linear(hidden_size, hidden_size)
            )
            self.convs.append(GINConv(layer))
            self.bns.append(nn.BatchNorm1d(hidden_size))
        self.post_mp = Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(), 
            nn.Linear(hidden_size, output_size),
        )
        self.decoder = MLP(
            [self.hparams["dim"]] +
            [self.hparams["autoencoder_width"]] *
            self.hparams["autoencoder_depth"] +
            [num_genes * 2], last_layer_act=decoder_activation)
        
        self.encoder = MLP(
            [num_genes] +
            [self.hparams["autoencoder_width"]] *
            self.hparams["autoencoder_depth"] +
            [self.hparams["dim"]])

    def forward(self, data):
        x, edge_index, batch = data.node_feature, data.edge_index, data.batch
        x = self.pre_mp(x)
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
        x = self.convs[-1](x, edge_index)
        x = pyg_nn.global_add_pool(x, batch) # Maybe remove this
        
        x = self.decoder()(x)
        #x = self.post_mp(x)          ## Change
        #x = F.log_softmax(x, dim=1)  ## Change        
        
        return x

    def loss(self, pred, y):
        return F.mse_loss(pred, y)
    
    
