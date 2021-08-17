import torch
import numpy as np
from torch_geometric.data import Data

import sys
sys.path.append('/dfs/user/yhr/cell_reprogram/model/')


def get_pert_idx(pert_category, adata, ctrl_adata, binary_pert=True):
    """
    Get indices (and signs) of perturbations

    """

    gene_names = adata.var.gene_symbols
    pert_idx = [np.where(p == gene_names)[0][0]
                for p in pert_category.split('+')
                if p != 'ctrl']

    # In case of binary perturbations, attach a sign to index value
    for i, p in enumerate(pert_idx):
        if binary_pert:
            pert_idx[i] = np.sign(adata.X[0,p] -
                                  ctrl_adata.X[0,p]) * pert_idx[i]

    return pert_idx

# Set up feature matrix and output
def create_cell_graph(X, y, node_map, G, de_idx, pert_idx=None):
    """
    Uses the gene expression information for a cell and an underlying 
    graph (e.g coexpression) to create a cell specific graph for control
    """

    pert_feats = np.zeros(len(X[0]))
    if pert_idx is not None:
        for p in pert_idx:
            pert_feats[int(np.abs(p))] = np.sign(p)
    pert_feats = np.expand_dims(pert_feats,0)
    feature_mat = torch.Tensor(np.concatenate([X, pert_feats])).T

    # Set up edges
    edge_index = [(node_map[e[0]], node_map[e[1]]) for e in G.edges]
    edge_index = torch.tensor(edge_index, dtype=torch.long).T

    # Create graph dataset
    return Data(x=feature_mat, edge_index=edge_index, y=torch.Tensor(y),
                de_idx=de_idx)


def create_cell_graph_dataset(adata, G, pert_category, num_samples=1,
                              binary_pert=True):
    """
    Create a dataset of graphs using AnnData object
    """

    adata_ = adata[adata.obs['condition'] == pert_category]
    node_map = {x:it for it,x in enumerate(adata_.var.gene_symbols)}
    de_genes = adata_.uns['rank_genes_groups_cov']
    Xs = []
    ys = []

    # When considering a non-control perturbation
    if pert_category != 'ctrl':
        ctrl_adata = adata[adata.obs['condition'] == 'ctrl']

        # Get the indices (and signs) of applied perturbation
        pert_idx = get_pert_idx(pert_category, adata_, ctrl_adata, binary_pert)

        # Store list of genes that are most differentially expressed for testing
        pert_de_category = adata_.obs['cov_drug_dose_name'][0]
        de_idx = np.where(adata_.var_names.isin(
                 np.array(de_genes[pert_de_category])))[0]

        for cell_z in adata_.X:
            # Use samples from control for input to the GNN_AE model
            ctrl_samples = ctrl_adata[np.random.randint(0,len(ctrl_adata),
                                                        num_samples),:]
            for c in ctrl_samples.X:
                Xs.append(c)
                ys.append(cell_z)

    # When considering a control perturbation
    else:
        pert_idx = None
        de_idx = None
        for cell_z in adata_.X:
            Xs.append(cell_z)
            ys.append(cell_z)

    # Create cell graphs
    cell_graphs = []
    for X,y in zip(Xs, ys):
        cell_graphs.append(create_cell_graph(X.toarray(), y.toarray(),
                                             node_map, G, de_idx, pert_idx))

    return cell_graphs