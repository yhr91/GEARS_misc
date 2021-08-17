import torch
import numpy as np
from torch_geometric.data import Data

import sys
sys.path.append('/dfs/user/yhr/cell_reprogram/model/')

# Set up feature matrix and output
def create_cell_graph(X, node_map, G):
    """
    Uses the gene expression information for a cell and an underlying 
    graph (e.g coexpression) to create a cell specific graph
    """
    
    feature_mat = torch.Tensor(np.concatenate([X, np.zeros([1,len(X[0])])])).T
    y = torch.Tensor(X)

    # Set up edges
    edge_index = [(node_map[e[0]], node_map[e[1]]) for e in G.edges]
    edge_index = torch.tensor(edge_index, dtype=torch.long).T

    # Create graph dataset
    return Data(x=feature_mat, edge_index=edge_index, y=y)

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
    