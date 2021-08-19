import torch
import torch.optim as optim
import scanpy as sc
import numpy as np
import pickle
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import os

import argparse
from model import GNN_AE, linear_model
from data import create_cell_graph_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from copy import deepcopy
from random import shuffle

import sys
sys.path.append('/dfs/user/yhr/cell_reprogram/model/')


def train(train_loader, val_loader, ood_loader, args,
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
        if val_res['mse'] < min_val:
            min_val = val_res['mse']
            best_model = deepcopy(model)

        # This test evaluation step should ideally be removed from here
        test_res = evaluate(ood_loader, model, device)
        log = "Epoch {}: Train: {:.4f}, R2 {:.4f} " \
              "Validation: {:.4f}. R2 {:.4f} " \
              "Test: {:.4f}, R2 {:.4f} " \
              "Loss: {:.4f}"
        print(log.format(epoch + 1, train_res['mse'], train_res['r2'],
                         val_res['mse'], val_res['r2'],
                         test_res['mse'], test_res['r2'],
                         total_loss))

        log = "DE_Train: {:.4f}, R2 {:.4f} " \
              "DE_Validation: {:.4f}. R2 {:.4f} " \
              "DE_Test: {:.4f}, R2 {:.4f} "
        print(log.format(train_res['mse_de'], train_res['r2_de'],
                         val_res['mse_de'], val_res['r2_de'],
                         test_res['mse_de'], test_res['r2_de']))
    return best_model


def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def evaluate(loader, model, device='cuda'):
    model.eval()
    pred = []
    truth = []
    pred_de = []
    truth_de = []

    for batch in loader:
        batch.to(device)
        results = {}

        if batch.de_idx is not None:
            non_ctrl_idx = np.where([np.sum(d != None) for d in batch.de_idx])[0]

        with torch.no_grad():
            p = model(batch)
            t = batch.y

            pred.extend(p)
            truth.extend(t)

            # differentially expressed genes
            if batch.de_idx is not None:
                p_de = [p[i,batch.de_idx[i]] for i in non_ctrl_idx]
                t_de = [t[i, batch.de_idx[i]] for i in non_ctrl_idx]

                pred_de.extend(p_de)
                truth_de.extend(t_de)

    # all genes
    pred = torch.stack(pred)
    truth = torch.stack(truth)

    results['mse'] = F.mse_loss(pred, truth)
    results['r2'] = r2_loss(pred, truth)

    results['mse_de'] = 0
    results['r2_de'] = 0

    if batch.de_idx is not None:
        pred_de = torch.stack(pred_de)
        truth_de = torch.stack(truth_de)

        results['mse_de'] = F.mse_loss(pred_de, truth_de)
        results['r2_de'] = r2_loss(pred_de, truth_de)

    return results


def get_train_test_split(args):

    adata = sc.read(args['fname'])
    train_adata = adata[adata.obs[args['split_key']] == 'train']
    val_adata = adata[adata.obs[args['split_key']] == 'test']
    ood_adata = adata[adata.obs[args['split_key']] == 'test']

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
    cell_graphs = {}
    train_split, val_split, ood_split = get_train_test_split(args)

    # Check if graphs have already been created and saved
    saved_graphs = './saved_graphs/'+ args['modelname'] +'.pkl'
    if os.path.isfile(saved_graphs):
        cell_graphs = pickle.load(open(saved_graphs, "rb"))
    else:
        for p in train_split + val_split + ood_split:
            cell_graphs[p] = create_cell_graph_dataset(adata, G, p,
                                        num_samples= args['num_ctrl_samples'],
                                        binary_pert=args['binary_pert'])
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

    print("Dataloaders created")
    return {'train_loader':train_loader,
            'val_loader':val_loader,
            'ood_loader':ood_loader}


def trainer(args):
    adata = sc.read_h5ad(args['fname'])
    gene_list = [f for f in adata.var.gene_symbols.values]
    args['gene_list'] = gene_list
    args['modelname'] = args['fname'].split('/')[-1].split('.h5ad')[0]

    l_model = linear_model(args)

    loaders = create_dataloaders(adata, l_model.G, args)
    best_model = train(loaders['train_loader'], loaders['val_loader'],
                       loaders['ood_loader'], args,
                       num_node_features=2,  device=args["device"])

    # Compute best ood performance overall
    test_res = evaluate(loaders['ood_loader'], best_model, args["device"])
    log = "Final best performing model: Test: {:.4f}, R2 {:.4f} "
    print(log.format(test_res['mse'], test_res['r2']))

    torch.save(best_model.state_dict(), './saved_models/'+args['modelname'])


def parse_arguments():
    """
    Argument parser
    """

    # dataset arguments
    parser = argparse.ArgumentParser(description='Perturbation response')
    parser.add_argument('--fname', type=str,
                        default='./datasets/small.h5ad')
                        #default='./datasets/Norman2019_prep_new_TFcombos.h5ad')
    parser.add_argument('--perturbation_key', type=str, default="condition")
    parser.add_argument('--species', type=str, default="human")
    parser.add_argument('--cell_type_key', type=str, default="cell_type")
    parser.add_argument('--split_key', type=str, default="split_small")
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
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--max_minutes', type=int, default=50)
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