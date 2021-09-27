import torch
import torch.optim as optim
import scanpy as sc
import numpy as np
import glob

import argparse
from model import linear_model, simple_AE
from data import PertDataloader
from copy import deepcopy
from inference import evaluate, compute_metrics, batch_predict

import sys
sys.path.append('/dfs/user/yhr/cell_reprogram/model/')


def trainer(args):
    """
    Stage 2 training for shared layers after GNN

    """
    # Set up data loaders
    adata = sc.read_h5ad(args['fname'])
    gene_list = [f for f in adata.var.gene_symbols.values]
    args['gene_list'] = gene_list
    args['num_genes'] = len(gene_list)
    args['modelname'] = args['fname'].split('/')[-1].split('.h5ad')[0] + \
                        args['data_suffix']

    l_model = linear_model(args['species'], args['regulon_name'],
                           args['gene_list'], args['adjacency'])
    pertdl = PertDataloader(adata, l_model.G, l_model.read_weights, args)

    # Compute number of features for each node
    item = [item for item in pertdl.loaders['train_loader']][0]
    args['num_node_features'] = item.x.shape[1]

    # Create a single dictionary indexed by gene id
    loaded_models = {}
    for m in glob.glob(args['modelnames']):
        loaded_models.update(torch.load(m, map_location=args['device']))

    train_preds = batch_predict(pertdl.loaders['train_loader'],
                                loaded_models, args)
    val_preds = batch_predict(pertdl.loaders['val_loader'],
                              loaded_models, args)

    best_model = train(args, train_preds, val_preds, pertdl.loaders)

    torch.save(best_model, args['out_name'])


def train(args, train_preds, val_preds, loaders):
    """
    Train a shared AE using outputs from stage 1 training

    """

    AE = simple_AE(args['num_node_features'],
                   args['num_genes'],
                   args['node_hidden_size'],
                   args['node_embed_size'],
                   args['ae_num_layers'],
                   args['ae_hidden_size'],
                   args['loss_type'])
    num_epochs = 500

    optimizer = optim.Adam(AE.parameters(), lr=args['lr'], weight_decay=5e-4)
    min_val = np.inf
    train_preds = torch.Tensor(train_preds)
    val_preds = torch.Tensor(val_preds)

    for epoch in range(num_epochs):
        num_graphs = 0
        total_loss = 0

        AE.to(args['device'])
        for itr, batch in enumerate(loaders['train_loader']):
            batch.to(args['device'])
            X = train_preds[num_graphs:num_graphs + batch.num_graphs]
            X = X.to(args['device'])

            y = batch.y
            optimizer.zero_grad()

            pred = AE(X)
            loss = AE.loss(pred, y, batch.pert, args['pert_loss_wt'])
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            num_graphs += batch.num_graphs

        total_loss /= num_graphs

        train_res = evaluate(loaders['train_loader'], AE,
                             train_preds, args, gene_idx=None)
        val_res = evaluate(loaders['val_loader'], AE,
                           val_preds, args, gene_idx=None)
        train_metrics, _ = compute_metrics(train_res, gene_idx=None)
        val_metrics, _ = compute_metrics(val_res, gene_idx=None)

        log = "Epoch {}: Train: {:.4f}, R2 {:.4f} " \
              "Validation: {:.4f}. R2 {:.4f} " \
              "Loss: {:.4f}"
        print(log.format(epoch + 1, train_metrics['mse'], train_metrics['r2'],
                         val_metrics['mse'], val_metrics['r2'],
                         total_loss))

        # Select best model
        if val_metrics['mse'] < min_val:
            min_val = val_metrics['mse']
            best_model = deepcopy(AE)

    return best_model


def parse_arguments():
    # TODO clean argument list
    # dataset arguments
    parser = argparse.ArgumentParser(description='Post training')
    parser.add_argument('--modelnames', type=str,
                    default='./saved_models/full_model_Norman2019_*GAT_relu_2lay*')
    parser.add_argument('--out_name', type=str,
                        default="AE_post_train_GAT_relu_2lay_deep")

    parser.add_argument('--fname', type=str,
                        default='./datasets/Norman2019_prep_new_TFcombosin5k_nocombo_somesingle_worstde_numsamples_1_new_method.h5ad')
    parser.add_argument('--perturbation_key', type=str, default="condition")
    parser.add_argument('--species', type=str, default="human")
    parser.add_argument('--split_key', type=str, default="split_yhr_TFcombos")
    parser.add_argument('--loss_ae', type=str, default='gauss')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--binary_pert', type=bool, default=True)
    parser.add_argument('--edge_attr', type=bool, default=True)
    parser.add_argument('--workers', type=int, default=1)

    # network arguments
    parser.add_argument('--regulon_name', type=str,
                        default='Norman2019_ctrl_only')
    parser.add_argument('--adjacency', type=str,
                        default='/dfs/user/yhr/cell_reprogram/Data'
                    '/learnt_weights/Norman2019_ctrl_only_learntweights.csv')

    # training arguments
    parser.add_argument('--device', type=str, default='cuda:9')
    parser.add_argument('--max_epochs', type=int, default=7)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--node_hidden_size', type=int, default=2)
    parser.add_argument('--node_embed_size', type=int, default=1)
    parser.add_argument('--ae_hidden_size', type=int, default=512)
    parser.add_argument('--gnn_num_layers', type=int, default=2)
    parser.add_argument('--ae_num_layers', type=int, default=2)
    parser.add_argument('--exp_name', type=str,
                        default='nodespecific_GAT_2_1_AE')
    parser.add_argument('--num_itr', type=int, default=1)
    parser.add_argument('--pert_loss_wt', type=int, default=1)
    parser.add_argument('--encode', type=bool, default=False)

    parser.add_argument('--GNN_node_specific', type=bool, default=True)
    parser.add_argument('--AE', type=bool, default=True)
    parser.add_argument('--train_genes', type=str, default='0,5')
    parser.add_argument('--GNN_simple', type=bool, default=False)
    parser.add_argument('--GNN', type=str, default='GraphConv')

    parser.add_argument('--diff_loss', type=bool, default=False)
    parser.add_argument('--edge_filter', type=bool, default=False)
    parser.add_argument('--edge_weights', type=bool, default=False)
    parser.add_argument('--zero_target_node', type=bool, default=True)
    parser.add_argument('--single_out', type=bool, default=True)
    parser.add_argument('--pert_feats', type=bool, default=False)
    parser.add_argument('--pert_delta', type=bool, default=True)

    parser.add_argument('--data_suffix', type=str, default='_pert_delta')
    parser.add_argument('--loss_type', type=str, default='micro')

    return dict(vars(parser.parse_args()))

if __name__ == "__main__":
    trainer(parse_arguments())