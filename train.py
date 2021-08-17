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
from sklearn.metrics import r2_score
from copy import deepcopy
from random import shuffle

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
            #print(itr)
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
        train_res = evaluate(train_loader, model, device)
        val_res = evaluate(val_loader, model, device)
        if val_mse < min_val:
            min_val = val_mse
            best_model = deepcopy(model)

        # This test evaluation step should ideally be removed from here
        test_res = evaluate(test_loader, model, device)
        log = "Epoch {}: Train: {:.4f}, R2 {:.4f} " \
              "Validation: {:.4f}. R2 {:.4f} " \
              "Test: {:.4f}, R2 {:.4f} " \
              "Loss: {:.4f}"
        print(log.format(epoch + 1, train_res['mse'], train_res['r2'],
                         val_res['mse'], val_res['r2'],
                         test_res['mse'], test_res['r2'],
                         total_loss))
    return best_model


def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def evaluate(loader, model, device='cuda'):
    model.eval()
    for batch in loader:
        batch.to(device)
        results = {}

        de_idx = batch.de_idx
        with torch.no_grad():
            pred = model(batch)
            truth = batch.y

            # all genes
            results['mse'] = F.mse_loss(pred, truth)
            results['r2'] = r2_loss(pred, truth)

            # differentially expressed genes
            if de_idx is not None:
                results['mse_de'] = F.mse_loss(pred[:,de_idx], truth[:,de_idx])
                results['r2_de'] = r2_loss(pred[:,de_idx], truth[:,de_idx])
    return results


def create_dataloaders(adata, G, args):
    """
    Set up dataloaders and splits

    """
    print("Creating dataloaders")

    # Create control dataset
    cell_graphs = {}

    # Perturbation categories to use during training/validation
    trainval_category = ['ctrl', 'KLF1+ctrl']

    #'ctrl+KLF1', 'CEBPA+ctrl',
    #                     'ctrl+CEBPA', 'CEBPE+ctrl', 'ctrl+CEBPE']

    # Perturbation categories to use for OOD testing
    ood_category = ['KLF1+CEBPA']

    #, 'CEBPE+KLF1', 'ctrl+FOXA1', 'FOXA1+ctrl']

    for p in trainval_category + ood_category:
        cell_graphs[p] = create_cell_graph_dataset(adata, G, p,
                                    num_samples= args['num_ctrl_samples'],
                                    binary_pert=args['binary_pert'])

    # Create a perturbation train/test set

    # Train/Test splits
    trainval_graphs = [cell_graphs[p] for p in trainval_category]
    trainval_graphs = [item for sublist in trainval_graphs for item in sublist]
    train, val = train_test_split(trainval_graphs, train_size=0.75,shuffle=True)

    # Out of distribution split
    ood_graphs = [cell_graphs[p] for p in ood_category]
    ood_graphs = [item for sublist in ood_graphs for item in sublist]
    test = ood_graphs
    shuffle(test)

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

    # Compute best ood performance overall
    test_res = evaluate(loaders['test_loader'], best_model, args["device"])
    log = "Final best performing model: Test: {:.4f}, R2 {:.4f} "
    print(log.format(test_res['mse'], test_res['r2']))


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
    parser.add_argument('--num_ctrl_samples', type=int, default=2)
    parser.add_argument('--binary_pert', type=bool, default=True)

    # network arguments
    parser.add_argument('--regulon_name', type=str,
                        default='Norman2019_ctrl_only')
    parser.add_argument('--adjacency', type=str,
                        default='/dfs/user/yhr/cell_reprogram/Data'
                    '/learnt_weights/Norman2019_ctrl_only_learntweights.csv')

    # training arguments
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--max_epochs', type=int, default=20)
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