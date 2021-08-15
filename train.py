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


def train(train_loader, val_loader, test_loader, args, num_node_features, num_classes, device="cpu"):
    model = GIN(num_node_features, args["hidden_size"], num_classes, args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=5e-4)
    best_model = None
    max_val = -1
    for epoch in range(args["epochs"]):
        total_loss = 0
        model.train()
        num_graphs = 0
        for genes, drugs, cell_types in datasets["loader_tr"]:  ## Change
            batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            label = batch.graph_label
            loss = model.loss(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            num_graphs += batch.num_graphs
        total_loss /= num_graphs
        train_acc = test(train_loader, model, device)
        val_acc = test(val_loader, model, device)
        if val_acc > max_val:
            max_val = val_acc
            best_model = deepcopy(model)
        test_acc = test(test_loader, model, device)
        log = "Epoch {}: Train: {:.4f}, Validation: {:.4f}. Test: {:.4f}, Loss: {:.4f}"
        print(log.format(epoch + 1, train_acc, val_acc, test_acc, total_loss))
    return best_model

def test(loader, model, device='cuda'):
    model.eval()
    correct = 0
    num_graphs = 0
    for batch in loader:
        batch.to(device)
        with torch.no_grad():
            pred = model(batch).max(dim=1)[1]
            label = batch.graph_label
        correct += pred.eq(label).sum().item()
        num_graphs += batch.num_graphs
    return correct / num_graphs


args = {}

args['fname'] = '/dfs/user/yhr/CPA_orig/datasets/Norman2019_prep_new_TFcombos.h5ad'
args['perturbation_key']='condition'
args['dose_key']='dose_val'
args['cell_type_key']='cell_type'
args['split_key']='split_yhr_TFcombos'
args['batch_size'] = 128
args['regulon_name'] = 'Norman2019_ctrl_only'
args['species'] = 'human'
args['adjacency'] = '/dfs/user/yhr/cell_reprogram/Data/learnt_weights/Norman2019_ctrl_only_learntweights.csv'

adata = sc.read_h5ad(args['fname'])
gene_list = [f for f in adata.var.gene_symbols.values]
args['gene_list'] = gene_list

hparams = {}
hparams['autoencoder_width'] = 512
hparams['autoencoder_depth'] =4
hparams['autoencoder_lr']=3e-4


model = linear_model(args)


## Ues control cells to create a graph dataset
control_adata = adata[adata.obs['condition'] == 'ctrl']
cell_graphs = create_cell_graph_dataset(control_adata, model.G)