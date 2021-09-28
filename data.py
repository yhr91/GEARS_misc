from torch_geometric.data import Data
import torch
import numpy as np
import pickle
from torch_geometric.data import DataLoader
import os
from random import shuffle
import pandas as pd
import scanpy as sc

import sys
sys.path.append('/dfs/user/yhr/cell_reprogram/model/')


class PertDataloader():
    """
    DataLoader class for creating perturbation graph objects
    Each graph uses prior biological information for determining graph structure
    Each node represents a gene and its expression value is assigned as a
    node feature.
    An additional feature is assigned to each node that represents the
    perturbation being applied to that node
    """

    def __init__(self, adata, G, weights, args, binary_pert=True):
        self.args = args
        self.adata = adata
        self.ctrl_adata = adata[adata.obs[args['perturbation_key']] == 'ctrl']
        self.G = G
        self.weights = weights
        self.node_map = {x: it for it, x in enumerate(adata.var.gene_symbols)}
        self.binary_pert=binary_pert
        self.gene_names = self.adata.var.gene_symbols
        self.loaders = self.create_dataloaders()

    def create_dataloaders(self):
        """
        Main routine for setting up dataloaders

        """
        print("Creating dataloaders")
        # Check if graphs have already been created and saved
        saved_graphs_fname = './saved_graphs/' + self.args['modelname'] + '.pkl'
        if os.path.isfile(saved_graphs_fname):
            cell_graphs = pickle.load(open(saved_graphs_fname, "rb"))
        else:
            cell_graphs = {}
            cell_graphs['train'] = self.create_split_dataloader('train')
            cell_graphs['val'] = self.create_split_dataloader('test')
            cell_graphs['ood'] = self.create_split_dataloader('ood')
            # Save graphs
            pickle.dump(cell_graphs, open(saved_graphs_fname, "wb"))

        # Set up dataloaders
        train_loader = DataLoader(cell_graphs['train'],
                            batch_size=self.args['batch_size'], shuffle=True)
        val_loader = DataLoader(cell_graphs['val'],
                            batch_size=self.args['batch_size'], shuffle=True)
        ood_loader = DataLoader(cell_graphs['ood'],
                            batch_size=self.args['batch_size'], shuffle=True)

        print("Dataloaders created")
        return {'train_loader': train_loader,
                'val_loader': val_loader,
                'ood_loader': ood_loader}

    def create_split_dataloader(self, split='train'):
        """
        Creates a dataloader for train, validation or test split
        """
        split_adata = self.adata[self.adata.obs[self.args['split_key']] == split]
        split_dl = []
        for p in split_adata.obs[self.args['perturbation_key']].unique():
            split_dl.extend(self.create_cell_graph_dataset(split_adata,
                            p, num_samples=self.args['num_ctrl_samples']))
        shuffle(split_dl)
        return split_dl

    def get_pert_idx(self, pert_category, adata_):
        """
        Get indices (and signs) of perturbations

        """

        pert_idx = [np.where(p == self.gene_names)[0][0]
                    for p in pert_category.split('+')
                    if p != 'ctrl']

        # In case of binary perturbations, attach a sign to index value
        for i, p in enumerate(pert_idx):
            if self.binary_pert:
                sign = np.sign(adata_.X[0, p] - self.ctrl_adata.X[0, p])
                if sign == 0:
                    sign = 1
                pert_idx[i] = sign * pert_idx[i]

        return pert_idx

    # Set up feature matrix and output
    def create_cell_graph(self, X, y, de_idx, pert, pert_idx=None):
        """
        Uses the gene expression information for a cell and an underlying
        graph (e.g coexpression) to create a graph for each cell
        """

        if self.args['pert_feats']:
            # If perturbations will be represented as node features
            pert_feats = np.zeros(len(X[0]))
            if pert_idx is not None:
                for p in pert_idx:
                    pert_feats[int(np.abs(p))] = np.sign(p)
            pert_feats = np.expand_dims(pert_feats, 0)
            feature_mat = torch.Tensor(np.concatenate([X, pert_feats])).T

        else:
            # If perturbations will NOT be represented as node features
            if pert_idx is not None:
                for p in pert_idx:
                    X[0][int(p)] += 1.0
            feature_mat = torch.Tensor(X).T

        if pert_idx is not None:
            # ONLY for perturbed cells
            if self.args['pert_delta']:
                # If making predictions on delta instead of absolute value
                temp = torch.zeros(feature_mat.shape)
                for p in pert_idx:
                    p_ = int(np.abs(p))
                    temp[p_, 0] = y[0, p_] - feature_mat[p_, 0]
                y = torch.Tensor(y.T) - feature_mat
                y = y.T
                feature_mat = temp

        # Set up edges
        if self.args['edge_filter']:
            if pert_idx is not None:
                edge_index_ = [(self.node_map[e[0]], self.node_map[e[1]]) for e in
                              self.G.edges if self.node_map[e[0]] in pert_idx]
            else:
                edge_index_ = []
        else:
            edge_index_ = [(self.node_map[e[0]], self.node_map[e[1]]) for e in
                          self.G.edges]

        # Set edge features
        if self.args['edge_attr']:
            edge_attr = self.weights.copy()
            edge_attr.index = edge_attr.index.map(self.node_map)
            edge_attr['target'] = edge_attr['target'].map(self.node_map)
            edges_df = pd.DataFrame(edge_index_, columns=['TF', 'target'])

            # Keep only those edges that have weights assigned
            edge_attr = edge_attr.reset_index().merge(edges_df,
                                    on=['TF', 'target'], how='inner')
            edge_index_ = [(s,t) for s,t in zip(edge_attr['TF'].values,
                                  edge_attr['target'].values)]
            edge_attr = edge_attr['importance'].values
            assert len(edge_attr) == len(edge_index_)

            # This prevents PyG from throwing errors if it finds no edges
            if len(edge_attr) == 0:
                edge_index_ = [(0,0)]
                edge_attr = [0]

            edge_attr = torch.Tensor(edge_attr).unsqueeze(1)
        else:
            edge_attr = None

        edge_index = torch.tensor(edge_index_, dtype=torch.long).T

        # Create graph dataset
        return Data(x=feature_mat, edge_index=edge_index, edge_attr=edge_attr,
                    y=torch.Tensor(y), de_idx=de_idx, pert=pert)

    def create_cell_graph_dataset(self, split_adata, pert_category,
                                  num_samples=1):
        """
        Combine cell graphs to create a dataset of cell graphs
        """

        print(pert_category)
        num_de_genes = 20
        adata_ = split_adata[split_adata.obs['condition'] == pert_category]
        de_genes = adata_.uns['rank_genes_groups_cov']
        Xs = []
        ys = []

        # When considering a non-control perturbation
        if pert_category != 'ctrl':
            # Get the indices (and signs) of applied perturbation
            pert_idx = self.get_pert_idx(pert_category, adata_)

            # Store list of genes that are most differentially expressed for testing
            pert_de_category = adata_.obs['cov_drug_dose_name'][0]
            de_idx = np.where(adata_.var_names.isin(
                np.array(de_genes[pert_de_category])))[0]

            for cell_z in adata_.X:
                # Use samples from control for input to the GNN_AE model
                ctrl_samples = self.ctrl_adata[np.random.randint(0,
                                        len(self.ctrl_adata), num_samples), :]
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
        for X, y in zip(Xs, ys):
            cell_graphs.append(self.create_cell_graph(X.toarray(), y.toarray(),
                                            de_idx, pert_category, pert_idx))

        return cell_graphs


    def get_train_test_split(self):
        """
        Get train, validation ('test' following naming convention from
        Lotfollahi) and test set ('ood') split from input data
        """

        adata = sc.read(self.args['fname'])
        train_adata = adata[adata.obs[self.args['split_key']] == 'train']
        val_adata = adata[adata.obs[self.args['split_key']] == 'test']
        ood_adata = adata[adata.obs[self.args['split_key']] == 'ood']

        train_split = list(train_adata.obs['condition'].unique())
        val_split = list(val_adata.obs['condition'].unique())
        ood_split = list(ood_adata.obs['condition'].unique())

        return train_split, val_split, ood_split