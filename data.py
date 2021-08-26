from torch_geometric.data import Data
import torch
import numpy as np
import pickle
from torch_geometric.data import DataLoader
import os
from random import shuffle
from multiprocessing import Pool
import tqdm
import scanpy as sc

import sys
sys.path.append('/dfs/user/yhr/cell_reprogram/model/')


def Map(F, x, workers):
    """
    wrapper for map()
    Spawn workers for parallel processing

    """
    with Pool(workers) as pool:
        # ret = pool.map(F, x)
        ret = list(tqdm.tqdm(pool.imap(F, x), total=len(x)))
    return ret


def get_ood_pert_classes(adata, split, batch_size, single_only=True):
    # Get names of a subset of out of distribution perturbations

    # Single perts only
    if single_only:
        ood_perts = [n for n in list(
            adata[adata.obs[split] == 'ood'].obs['condition'].unique()) if
                     'ctrl' in n]

    # All perts
    else:
        ood_perts = [n for n in list(
            adata[adata.obs[split] == 'ood'].obs['condition'].unique())]

    return ood_perts


def get_train_test_split(args):

    adata = sc.read(args['fname'])
    train_adata = adata[adata.obs[args['split_key']] == 'train']
    val_adata = adata[adata.obs[args['split_key']] == 'test']
    ood_adata = adata[adata.obs[args['split_key']] == 'ood']

    train_split = list(train_adata.obs['condition'].unique())
    val_split = list(val_adata.obs['condition'].unique())
    ood_split = list(ood_adata.obs['condition'].unique())

    return train_split, val_split, ood_split


def create_dataloaders(adata, G, args):
    """
    Set up dataloaders and splits

    """
    print("Creating dataloaders")

    # Create control dataset
    train_split, val_split, ood_split = get_train_test_split(args)

    # Check if graphs have already been created and saved
    saved_graphs = './saved_graphs/'+ args['modelname'] +'.pkl'
    if os.path.isfile(saved_graphs):
        cell_graphs = pickle.load(open(saved_graphs, "rb"))
    else:
        def mapping_func(p_):
            print(p_)
            return create_cell_graph_dataset(adata, G, p_,
                                      num_samples=args['num_ctrl_samples'],
                                      binary_pert=args['binary_pert'])
        all_perts = train_split + val_split + ood_split

        if args['workers']>1:
            print('Running process pool')
            # TODO this isn't working because the function is not global
            cell_graphs = Map(mapping_func, all_perts, workers=50)
        else:
            cell_graphs = [mapping_func(p) for p in all_perts]
        cell_graphs = {pert: graph for pert, graph in
                       zip(all_perts, cell_graphs)}
        # Save graphs
        pickle.dump(cell_graphs, open(saved_graphs, "wb"))



    # Create a perturbation train/test set
    # Train/Test splits
    train = [cell_graphs[p] for p in train_split]
    train = [item for sublist in train for item in sublist]
    shuffle(train)

    val = [cell_graphs[p] for p in val_split]
    val = [item for sublist in val for item in sublist]
    shuffle(val)

    ood = [cell_graphs[p] for p in ood_split]
    ood = [item for sublist in ood for item in sublist]
    shuffle(ood)

    # Set up dataloaders
    train_loader = DataLoader(train, batch_size=args['batch_size'],
                              shuffle=True)
    val_loader = DataLoader(val, batch_size=args['batch_size'],
                              shuffle=True)
    ood_loader = DataLoader(ood, batch_size=args['batch_size'],
                              shuffle=True)

    # Get ood pert class scores separately
    ood_perts = get_ood_pert_classes(adata, args['split_key'],
                batch_size=args['batch_size'], single_only=False)
    pert_loaders = []
    for p in ood_perts:
        pert_loaders.append(DataLoader(cell_graphs[p],
                            batch_size=args['batch_size'], shuffle=True))

    print("Dataloaders created")
    return {'train_loader':train_loader,
            'val_loader':val_loader,
            'ood_loader':ood_loader,
            'pert_loaders': pert_loaders}


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
            sign = np.sign(adata.X[0,p] - ctrl_adata.X[0,p])
            pert_idx[i] = sign * pert_idx[i]

    return pert_idx

# Set up feature matrix and output
def create_cell_graph(X, y, node_map, G, de_idx, pert, pert_idx=None):
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
                de_idx=de_idx, pert=pert)


def create_cell_graph_dataset(adata, G, pert_category, num_samples=1,
                              binary_pert=True):
    """
    Create a dataset of graphs using AnnData object
    """

    num_de_genes = 20
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
        de_idx = [-1] * num_de_genes
        for cell_z in adata_.X:
            Xs.append(cell_z)
            ys.append(cell_z)

    # Create cell graphs
    cell_graphs = []
    for X,y in zip(Xs, ys):
        cell_graphs.append(create_cell_graph(X.toarray(), y.toarray(),
                        node_map, G, de_idx, pert_category, pert_idx))

    return cell_graphs