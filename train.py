import torch
import torch.optim as optim
import scanpy as sc
import numpy as np
from torch_geometric.data import DataLoader
import torch.nn.functional as F

import argparse
from model import GNN_AE, linear_model
from data import create_cell_graph_dataset
from sklearn.model_selection import train_test_split
from copy import deepcopy

import sys
sys.path.append('/dfs/user/yhr/cell_reprogram/model/')


def train(train_loader, val_loader, test_loader, args,
                    num_node_features, device="cpu"):
    num_genes=5000 ## TODO this should be computed
    model = GNN_AE(num_node_features, num_genes,
                   args['gnn_num_layers'], args['node_hidden_size'],
                   args['node_embed_size'], args['ae_num_layers'],
                   args['ae_hidden_size']
                   ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=5e-4)
    best_model = None
    min_val = np.inf
    for epoch in range(args["max_epochs"]):
        total_loss = 0
        model.train()
        num_graphs = 0
        for itr, batch in enumerate(train_loader):  ## Change
            print(itr)
            batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = model.loss(pred, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            total_loss += loss.item()
            num_graphs += batch.num_graphs
        total_loss /= num_graphs
        train_mse = test(train_loader, model, device)
        val_mse = test(val_loader, model, device)
        if val_mse < min_val:
            min_val = val_mse
            best_model = deepcopy(model)
        test_mse = test(test_loader, model, device)
        log = "Epoch {}: Train: {:.4f}, Validation: {:.4f}. Test: {:.4f}, Loss: {:.4f}"
        print(log.format(epoch + 1, train_mse, val_mse, test_mse, total_loss))
    return best_model

def test(loader, model, device='cuda'):
    model.eval()
    for batch in loader:
        batch.to(device)
        with torch.no_grad():
            pred = model(batch)
            truth = batch.y
            mse = F.mse_loss(pred, truth)
    return mse


def create_dataloaders(adata, G, args):
    """
    Set up dataloaders and splits

    """
    print("Creating dataloaders")

    ## Ues control cells to create a graph dataset
    control_adata = adata[adata.obs['condition'] == 'ctrl']

    # Pick 50 random cells for testing code
    # control_dataset = control_adata[np.random.randint(0, len(control_adata),
    # 100),:]
    control_dataset = control_adata
    cell_graphs_dataset = create_cell_graph_dataset(control_dataset, G)

    # Train/Test splits
    train, test = train_test_split(cell_graphs_dataset, train_size=0.75,
                                   shuffle=True)
    val, test = train_test_split(test, train_size=0.5, shuffle=True)

    # Set up dataloaders
    train_loader = DataLoader(train, batch_size=args['batch_size'],
                              shuffle=True)
    val_loader = DataLoader(val, batch_size=args['batch_size'],
                              shuffle=True)
    test_loader = DataLoader(test, batch_size=args['batch_size'],
                              shuffle=True)

    print("Dataloaders created")
    return {'train_loader':train_loader,
            'val_loader':val_loader,
            'test_loader':test_loader}



def trainer(args):
    adata = sc.read_h5ad(args['fname'])
    gene_list = [f for f in adata.var.gene_symbols.values]
    args['gene_list'] = gene_list

    l_model = linear_model(args)

    loaders = create_dataloaders(adata, l_model.G, args)
    best_model = train(loaders['train_loader'], loaders['val_loader'],
                       loaders['test_loader'], args,
                       num_node_features=2,  device=args["device"])


def parse_arguments():
    """
    Argument parser
    """

    # dataset arguments
    parser = argparse.ArgumentParser(description='Perturbation response')
    parser.add_argument('--fname', type=str,
                        default='/dfs/user/yhr/CPA_orig/datasets'
                                '/Norman2019_prep_new_TFcombos.h5ad')
    parser.add_argument('--perturbation_key', type=str, default="condition")
    parser.add_argument('--species', type=str, default="human")
    parser.add_argument('--cell_type_key', type=str, default="cell_type")
    parser.add_argument('--split_key', type=str, default="split_yhr_TFcombos")
    parser.add_argument('--loss_ae', type=str, default='gauss')
    parser.add_argument('--doser_type', type=str, default='sigm')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--decoder_activation', type=str, default='linear')
    parser.add_argument('--seed', type=int, default=0)

    # network arguments
    parser.add_argument('--regulon_name', type=str,
                        default='Norman2019_ctrl_only')
    parser.add_argument('--adjacency', type=str,
                        default='/dfs/user/yhr/cell_reprogram/Data'
                    '/learnt_weights/Norman2019_ctrl_only_learntweights.csv')

    # training arguments
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--max_minutes', type=int, default=400)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--checkpoint_freq', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--node_hidden_size', type=int, default=16)
    parser.add_argument('--node_embed_size', type=int, default=8)
    parser.add_argument('--ae_hidden_size', type=int, default=512)
    parser.add_argument('--gnn_num_layers', type=int, default=4)
    parser.add_argument('--ae_num_layers', type=int, default=4)

    # output arguments
    parser.add_argument('--save_dir', type=str, default='./test/')
    return dict(vars(parser.parse_args()))


if __name__ == "__main__":
    trainer(parse_arguments())