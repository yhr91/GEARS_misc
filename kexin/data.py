from torch_geometric.data import Data
import torch
import numpy as np
import pickle
from torch_geometric.data import DataLoader
import os
from random import shuffle
import pandas as pd
import scanpy as sc
import networkx as nx
from tqdm import tqdm

import sys
sys.path.append('/dfs/user/yhr/cell_reprogram/model/')

from utils import parse_single_pert, parse_combo_pert, parse_any_pert

class Network():
    """
    Reads in background network with associated weights
    """
    def __init__(self, fname, gene_list, percentile=None, feature='importance'):
        self.gene_list = gene_list
        self.percentile = percentile
        self.feature = feature

        self.edge_list = pd.read_csv(fname)
        self.correct_node_list()
        if self.percentile is not None:
            self.get_top_edges()

        self.G = self.create_graph()
        self.add_missing_nodes()
        self.weights = self.edge_list[self.feature].values

    def create_graph(self):
        """
        Create networkx graph object
        """
        G = nx.from_pandas_edgelist(self.edge_list, source='source',
                        target='target', edge_attr=[self.feature],
                        create_using=nx.DiGraph())
        return G

    def correct_node_list(self):
        """
        Remove nodes from graph that are not in the gene list
        """
        gene_list_df = pd.DataFrame(self.gene_list, columns=['gene'])
        self.edge_list = self.edge_list.merge(gene_list_df, left_on='source',
                                              right_on='gene', how='inner')
        self.edge_list = self.edge_list.merge(gene_list_df, left_on='target',
                                              right_on='gene', how='inner')

    def add_missing_nodes(self):
        """
        Add disconnected nodes to the graph that are in gene list but not in G
        """

        for n in self.gene_list:
            if n not in self.G.nodes():
                self.G.add_node(n)

    def get_top_edges(self):
        """
        Get top edges from network
        """
        self.edge_list = self.edge_list.sort_values(self.feature,
                                                    ascending=False)
        num_top_edges = int(len(self.edge_list) * self.percentile/100)
        self.edge_list = self.edge_list[:num_top_edges]
        print('There are ' + str(len(self.edge_list)) + ' edges in the PPI.')
        self.edge_list.sort_values('source')


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
        self.edge_index, self.edge_attr = self.compute_edge_index()
        self.binary_pert=binary_pert
        self.gene_names = self.adata.var.gene_symbols
        self.loaders = self.create_dataloaders()

    def create_dataloaders(self):
        """
        Main routine for setting up dataloaders
        """
        print("Creating pyg object for each cell in the data...")
        
        # create dataset processed pyg objects
        dataset_fname = './data_pyg/' + \
                             self.args['dataset'] + '.pkl'
                
        if os.path.isfile(dataset_fname):
            print("Local copy of pyg dataset is detected. Loading...")
            dataset_processed = pickle.load(open(dataset_fname, "rb"))
        else:
            print("Processing dataset...")
            
            dataset_processed = self.create_dataset_file()
            print("Saving new dataset pyg object at " + dataset_fname) 
            pickle.dump(dataset_processed, open(dataset_fname, "wb"))
            
        print("Loading splits...")
        
        split_path = './splits/' + self.args['dataset'] + '_' + self.args['split'] + '_' + str(self.args['seed']) + '_' + str(self.args['test_set_fraction']) + '.pkl'
        
        if os.path.exists(split_path):
            print("Local copy of split is detected. Loading...")
            set2conditions = pickle.load(open(split_path, "rb"))
            if self.args['split'] == 'simulation':
                subgroup_path = split_path[:-4] + '_subgroup.pkl'
                subgroup = pickle.load(open(subgroup_path, "rb"))
                self.subgroup = subgroup
        else:
            print("Creating new splits....")
            if self.args['split'] == 'simulation':
                DS = DataSplitter(self.adata, split_type='simulation')
                
                adata, subgroup = DS.split_data(train_gene_set_size = self.args['train_gene_set_size'], 
                                                combo_seen2_train_frac = self.args['combo_seen2_train_frac'],
                                                seed=self.args['seed'])
                subgroup_path = split_path[:-4] + '_subgroup.pkl'
                pickle.dump(subgroup, open(subgroup_path, "wb"))
                self.subgroup = subgroup
                
                
            elif self.args['split'][:5] == 'combo':
                split_type = 'combo'
                seen = int(self.args['split'][-1])
                DS = DataSplitter(self.adata, split_type=split_type,
                                  seen=int(seen))
                adata = DS.split_data(test_size=self.args['test_set_fraction'], split_name='split',
                                       seed=self.args['seed'])
            else:
                DS = DataSplitter(self.adata, split_type=self.args['split'])
            
                adata = DS.split_data(test_size=self.args['test_set_fraction'], split_name='split',
                                       seed=self.args['seed'])
            
            set2conditions = dict(adata.obs.groupby('split').agg({'condition': lambda x: x}).condition)
            set2conditions = {i: j.unique().tolist() for i,j in set2conditions.items()} 
            pickle.dump(set2conditions, open(split_path, "wb"))
            print("Saving new splits at " + split_path) 
        
        if self.args['split'] == 'simulation':
            print('Simulation split test composition:')
            for i,j in subgroup['test_subgroup'].items():
                print(i + ':' + str(len(j)))
        
        # Create cell graphs
        cell_graphs = {}
        for i in ['train', 'val', 'test']:
            if (i == 'train' and self.args['ctrl_remove_train']) or (i == 'val'):
                # remove control set from training given the args
                # remove control set from validation in default
                if 'ctrl' in set2conditions[i]:
                    set2conditions[i].remove('ctrl')           
            cell_graphs[i] = []
            for p in set2conditions[i]:
                cell_graphs[i].extend(dataset_processed[p])
        
        print("Creating dataloaders....")
        # Set up dataloaders
        train_loader = DataLoader(cell_graphs['train'],
                            batch_size=self.args['batch_size'], shuffle=True)
        val_loader = DataLoader(cell_graphs['val'],
                            batch_size=self.args['batch_size'], shuffle=True)
        test_loader = DataLoader(cell_graphs['test'],
                            batch_size=self.args['batch_size'], shuffle=True)

        print("Dataloaders created...")
        return {'train_loader': train_loader,
                'val_loader': val_loader,
                'test_loader': test_loader,
                'edge_index': self.edge_index,
                'edge_attr': self.edge_attr}
    
    
    def create_dataset_file(self):
        """
        Creates a dataloader for adata dataset
        """
        dl = {}

        for p in tqdm(self.adata.obs[self.args['perturbation_key']].unique()):
            cell_graph_dataset = self.create_cell_graph_dataset(
                self.adata, p, num_samples=self.args['num_ctrl_samples'])
            dl[p] = cell_graph_dataset
        return dl
    
    '''
    def create_split_dataloader(self, split='train'):
        """
        Creates a dataloader for train, validation or test split
        """
        split_adata = self.adata[self.adata.obs[self.args['split_key']] == split]
        split_dl = []

        for p in split_adata.obs[self.args['perturbation_key']].unique():
            cell_graph_dataset = self.create_cell_graph_dataset(
                split_adata, p, num_samples=self.args['num_ctrl_samples'])
            split_dl.extend(cell_graph_dataset)
        shuffle(split_dl)
        return split_dl
    '''
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

        '''
        # In case network is different between different perturbations
        if self.args['edge_filter']:
            if pert_idx is not None:
                edge_index_ = [(self.node_map[e[0]], self.node_map[e[1]]) for e in
                              self.G.edges if self.node_map[e[0]] in pert_idx]
            else:
                edge_index_ = []
            edge_index = torch.tensor(edge_index_, dtype=torch.long).T
        else:
            edge_index = self.edge_index
        '''
        # We'll save the graph structure separately because it leads to a lot
        # of redundant memory usage between samples
        save_graph = None
        save_attr = None

        return Data(x=feature_mat, edge_index=save_graph, edge_attr=save_attr,
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
            cell_graphs.append(self.create_cell_graph(X.toarray(),
                                y.toarray(), de_idx, pert_category, pert_idx))

        return cell_graphs


    def get_train_test_split(self):
        """
        Get train, validation and test set split from input data
        """

        adata = sc.read(self.args['fname'])
        train_adata = adata[adata.obs[self.args['split_key']] == 'train']
        val_adata = adata[adata.obs[self.args['split_key']] == 'val']
        test_adata = adata[adata.obs[self.args['split_key']] == 'test']

        train_split = list(train_adata.obs['condition'].unique())
        val_split = list(val_adata.obs['condition'].unique())
        test_split = list(test_adata.obs['condition'].unique())

        return train_split, val_split, test_split

    def compute_edge_index(self):
        """
        Note: Assumes common graph shared by all cells
        In most cases, the same graph is shared by all cells and this can be
        represented in the PyG edge index format
        """
        edge_index_ = [(self.node_map[e[0]], self.node_map[e[1]]) for e in
                      self.G.edges]
        edge_index = torch.tensor(edge_index_, dtype=torch.long).T

        # Set edge features
        if self.args['edge_attr']:
            assert len(self.weights) == len(self.G.edges)
            edge_attr = torch.Tensor(self.weights).unsqueeze(1)
        else:
            edge_attr = None

        return edge_index, edge_attr
    
    
class DataSplitter():
    """
    Class for handling data splitting. This class is able to generate new
    data splits and assign them as a new attribute to the data file.
    """
    def __init__(self, adata, split_type='single', seen=0):
        self.adata = adata
        self.split_type = split_type
        self.seen = seen

    def split_data(self, test_size=0.1, test_pert_genes=None,
                   test_perts=None, split_name='split', seed=None, val_size = 0.1,
                   train_gene_set_size = 0.75, combo_seen2_train_frac = 0.75):
        """
        Split dataset and adds split as a column to the dataframe
        Note: split categories are train, val, test
        """
        np.random.seed(seed=seed)
        unique_perts = [p for p in self.adata.obs['condition'].unique() if
                        p != 'ctrl']
        
        if self.split_type == 'simulation':
            train, test, test_subgroup = self.get_simulation_split(unique_perts,
                                                                  train_gene_set_size,
                                                                  combo_seen2_train_frac, 
                                                                  seed)
            train, val, val_subgroup = self.get_simulation_split(train,
                                                                  0.9,
                                                                  0.9,
                                                                  seed)
            ## adding back ctrl to train...
            train.append('ctrl')
            
        else:
            train, test = self.get_split_list(unique_perts,
                                          test_pert_genes=test_pert_genes,
                                          test_perts=test_perts,
                                          test_size=test_size)
            
            train, val = self.get_split_list(train, test_size=val_size)

        map_dict = {x: 'train' for x in train}
        map_dict.update({x: 'val' for x in val})
        map_dict.update({x: 'test' for x in test})
        map_dict.update({'ctrl': 'train'})

        self.adata.obs[split_name] = self.adata.obs['condition'].map(map_dict)

        # Add some control to the validation set
        #ctrl_idx = self.adata.obs_names[self.adata.obs['condition'] == 'ctrl']
        #val_ctrl = np.random.choice(ctrl_idx, int(len(ctrl_idx) * test_size))
        #self.adata.obs.at[val_ctrl, 'split'] = 'val'
        if self.split_type == 'simulation':
            return self.adata, {'test_subgroup': test_subgroup, 
                                'val_subgroup': val_subgroup
                               }
        else:
            return self.adata
        
    def get_simulation_split(self, pert_list, train_gene_set_size = 0.85, combo_seen2_train_frac = 0.85, seed = 1):
        
        unique_pert_genes = self.get_genes_from_perts(pert_list)
        
        pert_train = []
        pert_test = []
        np.random.seed(seed=seed)
        
        ## pre-specified list of genes
        train_gene_candidates = np.random.choice(unique_pert_genes,
                                                int(len(unique_pert_genes) * train_gene_set_size), replace = False)
        ## ood genes
        ood_genes = np.setdiff1d(unique_pert_genes, train_gene_candidates)
        
        pert_single_train = self.get_perts_from_genes(train_gene_candidates, pert_list,'single')
        pert_combo = self.get_perts_from_genes(train_gene_candidates, pert_list,'combo')
        pert_train.extend(pert_single_train)
        
        ## the combo set with one of them in OOD
        combo_seen1 = [x for x in pert_combo if len([t for t in x.split('+') if
                                     t in train_gene_candidates]) == 1]
        pert_test.extend(combo_seen1)
        
        pert_combo = np.setdiff1d(pert_combo, combo_seen1)
        ## randomly sample the combo seen 2 as a test set, the rest in training set
        np.random.seed(seed=seed)
        pert_combo_train = np.random.choice(pert_combo, int(len(pert_combo) * combo_seen2_train_frac), replace = False)
       
        combo_seen2 = np.setdiff1d(pert_combo, pert_combo_train).tolist()
        pert_test.extend(combo_seen2)
        pert_train.extend(pert_combo_train)
        
        ## unseen single
        unseen_single = self.get_perts_from_genes(ood_genes, pert_list, 'single')
        combo_ood = self.get_perts_from_genes(ood_genes, pert_list, 'combo')
        pert_test.extend(unseen_single)
        
        ## here only keeps the seen 0, since seen 1 is tackled above
        combo_seen0 = [x for x in combo_ood if len([t for t in x.split('+') if
                                     t in train_gene_candidates]) == 0]
        pert_test.extend(combo_seen0)
        assert len(combo_seen1) + len(combo_seen0) + len(unseen_single) + len(pert_train) + len(combo_seen2) == len(pert_list)

        return pert_train, pert_test, {'combo_seen0': combo_seen0,
                                       'combo_seen1': combo_seen1,
                                       'combo_seen2': combo_seen2,
                                       'unseen_single': unseen_single}
        
    def get_split_list(self, pert_list, test_size=0.1,
                       test_pert_genes=None, test_perts=None,
                       hold_outs=True):
        """
        Splits a given perturbation list into train and test with no shared
        perturbations
        """

        single_perts = [p for p in pert_list if 'ctrl' in p and p != 'ctrl']
        combo_perts = [p for p in pert_list if 'ctrl' not in p]
        unique_pert_genes = self.get_genes_from_perts(pert_list)
        hold_out = []

        if test_pert_genes is None:
            test_pert_genes = np.random.choice(unique_pert_genes,
                                        int(len(single_perts) * test_size))

        # Only single unseen genes (in test set)
        # Train contains both single and combos
        if self.split_type == 'single' or self.split_type == 'single_only':
            test_perts = self.get_perts_from_genes(test_pert_genes, pert_list,
                                                   'single')
            if self.split_type == 'single_only':
                # Discard all combos
                hold_out = combo_perts
            else:
                # Discard only those combos which contain test genes
                hold_out = self.get_perts_from_genes(test_pert_genes, pert_list,
                                                     'combo')

        elif self.split_type == 'combo':
            if self.seen == 0:
                # NOTE: This can reduce the dataset size!
                # To prevent this set 'holdouts' to False, this will cause
                # the test set to have some perturbations with 1 gene seen
                single_perts = self.get_perts_from_genes(test_pert_genes,
                                                         pert_list, 'single')
                combo_perts = self.get_perts_from_genes(test_pert_genes,
                                                        pert_list, 'combo')

                if hold_outs:
                    # This just checks that none of the combos have 2 seen genes
                    hold_out = [t for t in combo_perts if
                                len([t for t in t.split('+') if
                                     t not in test_pert_genes]) > 0]
                combo_perts = [c for c in combo_perts if c not in hold_out]
                test_perts = single_perts + combo_perts

            elif self.seen == 1:
                # NOTE: This can reduce the dataset size!
                # To prevent this set 'holdouts' to False, this will cause
                # the test set to have some perturbations with 2 genes seen
                single_perts = self.get_perts_from_genes(test_pert_genes,
                                                         pert_list, 'single')
                combo_perts = self.get_perts_from_genes(test_pert_genes,
                                                        pert_list, 'combo')

                if hold_outs:
                    # This just checks that none of the combos have 2 seen genes
                    hold_out = [t for t in combo_perts if
                                len([t for t in t.split('+') if
                                     t not in test_pert_genes]) > 1]
                combo_perts = [c for c in combo_perts if c not in hold_out]
                test_perts = single_perts + combo_perts

            elif self.seen == 2:
                if test_perts is None:
                    test_perts = np.random.choice(combo_perts,
                                     int(len(combo_perts) * test_size))       
        
        train_perts = [p for p in pert_list if (p not in test_perts)
                                        and (p not in hold_out)]
        return train_perts, test_perts

    def get_perts_from_genes(self, genes, pert_list, type_='both'):
        """
        Returns all single/combo/both perturbations that include a gene
        """

        single_perts = [p for p in pert_list if ('ctrl' in p) and (p != 'ctrl')]
        combo_perts = [p for p in pert_list if 'ctrl' not in p]
        
        perts = []
        
        if type_ == 'single':
            pert_candidate_list = single_perts
        elif type_ == 'combo':
            pert_candidate_list = combo_perts
        elif type_ == 'both':
            pert_candidate_list = pert_list
            
        for p in pert_candidate_list:
            for g in genes:
                if g in parse_any_pert(p):
                    perts.append(p)
                    break
        '''
        perts = []
        for gene in genes:
            if type_ == 'single':
                perts.extend([p for p in single_perts if gene in parse_any_pert(p)])

            if type_ == 'combo':
                perts.extend([p for p in combo_perts if gene in parse_any_pert(p)])

            if type_ == 'both':
                perts.extend([p for p in pert_list if gene in parse_any_pert(p)])
        '''
        return perts

    def get_genes_from_perts(self, perts):
        """
        Returns list of genes involved in a given perturbation list
        """

        if type(perts) is str:
            perts = [perts]
        gene_list = [p.split('+') for p in np.unique(perts)]
        gene_list = [item for sublist in gene_list for item in sublist]
        gene_list = [g for g in gene_list if g != 'ctrl']
        return np.unique(gene_list)