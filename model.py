import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch.optim as optim
import networkx as nx
import torch_geometric.nn as pyg_nn

from torch_geometric.nn import GINConv
from torch.nn import Sequential, Linear, ReLU
from deepsnap.batch import Batch
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

        self.gene_list = args['gene_list']
        # Get adjacency matrix
        # self.adj_mat = self.create_adj_mat()
    
    
    def create_adj_mat(self, gene_list):
        # Create a df version of the graph for merging
        G_df = pd.DataFrame(self.G.edges(), columns=['TF', 'target'])

        # Merge it with the weights DF
        weighted_G_df = self.read_weights.merge(G_df, on=['TF', 'target'])
        for w in weighted_G_df.iterrows():
            add_weight(self.G, w[1]['TF'], w[1]['target'], w[1]['importance'])

        # Get an adjacency matrix based on the gene ordering from the DE list
        return nx.linalg.graphmatrix.adjacency_matrix(
            self.G, nodelist=self.gene_list).todense()
    
    
class MLP(torch.nn.Module):
    """
    A multilayer perceptron with ReLU activations and optional BatchNorm.
    """

    def __init__(self, sizes, batch_norm=False, last_layer_act="linear"):
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
    def __init__(self, num_feats, num_layers, hidden_size, output_size=8):
        super(GIN, self).__init__()
        self.mp_layers = num_layers
        self.pre_mp = nn.Linear(num_feats, hidden_size)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for l in range(self.mp_layers):
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

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = np.zeros(data.x.shape[1])
        x = self.pre_mp(x.T)
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
        x = self.convs[-1](x, edge_index)

        #x = pyg_nn.global_add_pool(x, batch) # Replace this with concat

        x = self.post_mp(x)          ## Change
        #x = F.log_softmax(x, dim=1)  ## Change        
        
        return torch.unsqueeze(torch.flatten(x),0)

class GNN_AE(torch.nn.Module):
    def __init__(self, num_node_features, num_genes,
                 gnn_num_layers, node_embed_size,
                 ae_num_layers, ae_hidden_size, ae_input_size):
        super(GNN_AE, self).__init__()
        self.GNN = GIN(num_node_features, gnn_num_layers, node_embed_size)
        self.encoder = MLP(
            [ae_input_size] + [ae_hidden_size] * ae_num_layers + [ae_hidden_size])
        self.decoder = MLP(
            [ae_hidden_size] + [ae_hidden_size] * ae_num_layers + [num_genes],
            last_layer_act='linear')

    def forward(self, x):
        post_gnn = self.GNN(x)
        encoded = self.encoder(post_gnn)
        x = self.decoder(encoded)

        return x

    def loss(self, pred, y):
        return F.mse_loss(pred, y)
    
    
