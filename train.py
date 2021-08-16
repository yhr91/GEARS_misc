import torch
import torch.optim as optim
import scanpy as sc
import numpy as np

import argparse
from model import GNN_AE, linear_model
from data import create_cell_graph_dataset
from copy import deepcopy

import sys
sys.path.append('/dfs/user/yhr/cell_reprogram/model/')


def train(train_loader, val_loader, test_loader, args, num_node_features, device="cpu"):
    num_genes=5000 ## TODO this should be computed
    feature_scale_up = 8

    model = GNN_AE(num_node_features, num_genes,
                   args['gnn_num_layers'], args['node_embed_size'],
                   args['ae_num_layers'], args['ae_embed_size'],
                   num_genes*feature_scale_up).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=5e-4)
    best_model = None
    max_val = -1
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
            #total_loss += loss.item() * batch.size()[0] # TODO Change to cells
            total_loss += loss.item()
            ## num_graphs += batch.num_graphs TODO
        #total_loss /= num_graphs # TODO
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


def trainer(args):
    adata = sc.read_h5ad(args['fname'])
    gene_list = [f for f in adata.var.gene_symbols.values]
    args['gene_list'] = gene_list

    model = linear_model(args)

    ## Ues control cells to create a graph dataset
    control_adata = adata[adata.obs['condition'] == 'ctrl']

    ## TODO replace with dataloaders
    # Pick 50 random cells for testing code
    control_dataset = control_adata[np.random.randint(0, len(control_adata),100),:]

    cell_graphs_dataset = create_cell_graph_dataset(control_dataset, model.G)
    train_loader = cell_graphs_dataset[:60]
    val_loader = cell_graphs_dataset[60:80]
    test_loader = cell_graphs_dataset[80:]

    best_model = train(train_loader, val_loader, test_loader, args,
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
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--decoder_activation', type=str, default='linear')
    parser.add_argument('--seed', type=int, default=0)

    # network arguments
    parser.add_argument('--regulon_name', type=str,
                        default='Norman2019_ctrl_only')
    parser.add_argument('--adjacency', type=str,
                        default='/dfs/user/yhr/cell_reprogram/Data'
                    '/learnt_weights/Norman2019_ctrl_only_learntweights.csv')

    # training arguments
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--max_minutes', type=int, default=400)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--checkpoint_freq', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--node_embed_size', type=int, default=16)
    parser.add_argument('--ae_embed_size', type=int, default=512)
    parser.add_argument('--gnn_num_layers', type=int, default=4)
    parser.add_argument('--ae_num_layers', type=int, default=4)

    # output arguments
    parser.add_argument('--save_dir', type=str, default='./test/')
    return dict(vars(parser.parse_args()))


if __name__ == "__main__":
    trainer(parse_arguments())