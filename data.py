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




# Set up feature matrix and output
def create_cell_graph(X, node_map, G):
    """
    Uses the gene expression information for a cell and an underlying 
    graph (e.g coexpression) to create a cell specific graph
    """
    
    feature_mat = torch.Tensor(np.concatenate([X, np.zeros([1,len(X[0])])]))
    y = torch.Tensor(X)

    # Set up edges
    edge_index = [(node_map[e[0]], node_map[e[1]]) for e in G.edges]
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Create graph dataset
    return Data(x=feature_mat, edge_index=edge_index, y=X)

def create_cell_graph_dataset(adata, G):
    """
    Create a dataset of graphs using AnnData object
    TODO: This can be optimized for larger datasets
    """
    
    node_map = {x:it for it,x in enumerate(adata.var.gene_symbols)}
    
    cell_graphs = []
    for X in adata.X[:]:
        cell_graphs.append(create_cell_graph(X.toarray(), node_map, G))
    
    return cell_graphs
    