import torch
import torch.optim as optim
import scanpy as sc
import numpy as np

import argparse
from model import GNN_AE, linear_model
from data import create_dataloaders
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from copy import deepcopy

import sys
sys.path.append('/dfs/user/yhr/cell_reprogram/model/')


def train(train_loader, val_loader, args,
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

        # Evaluate model performance on train and val set
        train_res = evaluate(train_loader, model, device)
        val_res = evaluate(val_loader, model, device)
        train_metrics, _ = compute_metrics(train_res)
        val_metrics, _ = compute_metrics(val_res)

        if val_metrics['mse'] < min_val:
            min_val = val_metrics['mse']
            best_model = deepcopy(model)

        # Evaluate model performance on train and val set
        log = "Epoch {}: Train: {:.4f}, R2 {:.4f} " \
              "Validation: {:.4f}. R2 {:.4f} " \
              "Loss: {:.4f}"
        print(log.format(epoch + 1, train_metrics['mse'], train_metrics['r2'],
                         val_metrics['mse'], val_metrics['r2'],
                         total_loss))

        log = "DE_Train: {:.4f}, R2 {:.4f} " \
              "DE_Validation: {:.4f}. R2 {:.4f} "
        print(log.format(train_metrics['mse_de'], train_metrics['r2_de'],
                         val_metrics['mse_de'], val_metrics['r2_de']))
    return best_model


def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def evaluate(loader, model, device='cuda', num_de_idx=20):
    """
    Run model in inference mode using a given data loader
    """

    model.eval()
    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    truth_de = []
    results = {}

    for batch in loader:
        batch.to(device)
        results = {}
        pert_cat.extend(batch.pert)

        if batch.de_idx is not None:
            non_ctrl_idx = np.where([np.sum(d != None) for d in batch.de_idx])[0]
        else:
            print('here')

        with torch.no_grad():
            p = model(batch)
            t = batch.y

            pred.extend(p)
            truth.extend(t)

            # Differentially expressed genes
            for itr, de_idx in enumerate(batch.de_idx):
                if de_idx is not None:
                    pred_de.append(p[itr, de_idx])
                    truth_de.append(t[itr, de_idx])

                else:
                    pred_de.append([torch.zeros(num_de_idx).to(device)])
                    truth_de.append([torch.zeros(num_de_idx).to(device)])

    # all genes
    results['pert_cat'] = np.array(pert_cat)

    pred = torch.stack(pred)
    truth = torch.stack(truth)
    results['pred']= pred.detach().cpu().numpy()
    results['truth']= truth.detach().cpu().numpy()

    pred_de = torch.stack(pred_de)
    truth_de = torch.stack(truth_de)
    results['pred_de']= pred_de.detach().cpu().numpy()
    results['truth_de']= truth_de.detach().cpu().numpy()

    return results


def compute_metrics(results):
    """
    Given results from a model run and the ground truth, compute metrics

    """
    metrics = {}
    metrics_pert = {}
    metrics['mse'] = []
    metrics['r2'] = []
    metrics['mse_de'] = []
    metrics['r2_de'] = []

    for pert in np.unique(results['pert_cat']):

        metrics_pert[pert] = {}
        p_idx = np.where(results['pert_cat'] == pert)[0]
        metrics['r2'].append(r2_score(results['pred'][p_idx].mean(0),
                                      results['truth'][p_idx].mean(0)))
        metrics['mse'].append(mse(results['pred'][p_idx].mean(0),
                                  results['truth'][p_idx].mean(0)))
        metrics_pert[pert]['r2'] = metrics['r2'][-1]
        metrics_pert[pert]['mse'] = metrics['mse'][-1]

        if pert != 'ctrl':
            metrics['r2_de'].append(r2_score(results['pred_de'][p_idx].mean(0),
                                           results['truth_de'][p_idx].mean(0)))
            metrics['mse_de'].append(mse(results['pred_de'][p_idx].mean(0),
                                         results['truth_de'][p_idx].mean(0)))
            metrics_pert[pert]['r2_de'] = metrics['r2_de'][-1]
            metrics_pert[pert]['mse_de'] = metrics['mse_de'][-1]
        else:
            metrics_pert[pert]['r2_de'] = 0
            metrics_pert[pert]['mse_de'] = 0



    metrics['mse'] = np.mean(metrics['mse'])
    metrics['r2'] = np.mean(metrics['r2'])
    metrics['mse_de'] = np.mean(metrics['mse_de'])
    metrics['r2_de'] = np.mean(metrics['r2_de'])

    return metrics, metrics_pert


def trainer(args):
    adata = sc.read_h5ad(args['fname'])
    gene_list = [f for f in adata.var.gene_symbols.values]
    args['gene_list'] = gene_list
    args['modelname'] = args['fname'].split('/')[-1].split('.h5ad')[0]

    try:
        args['num_ctrl_samples'] = adata.uns['num_ctrl_samples']
    except:
        args['num_ctrl_samples'] = 1

    print('Training '+ args['modelname'] + '_' + args['exp_name'])

    # Set up data loaders
    l_model = linear_model(args)
    loaders = create_dataloaders(adata, l_model.G, args)

    # Train a model
    best_model = train(loaders['train_loader'], loaders['val_loader'],
                       args, num_node_features=2,  device=args["device"])

    test_res = evaluate(loaders['ood_loader'], best_model, args["device"])
    test_metrics, test_pert_res = compute_metrics(test_res)

    log = "Final best performing model: Test_DE: {:.4f}, R2 {:.4f} "
    print(log.format(test_metrics['mse_de'], test_metrics['r2_de']))

    # Save model outputs and best model
    np.save('./saved_metrics/'+args['modelname'], test_pert_res)
    torch.save(best_model.state_dict(), './saved_models/'+args['modelname']+
               '_'+ args['exp_name'])


def parse_arguments():
    """
    Argument parser
    """

    # dataset arguments
    parser = argparse.ArgumentParser(description='Perturbation response')
    parser.add_argument('--fname', type=str,
                        default='./datasets/Norman2019_prep_new_TFcombosin5k_nocombo_somesingle_worstde_numsamples_1.h5ad')
    parser.add_argument('--perturbation_key', type=str, default="condition")
    parser.add_argument('--species', type=str, default="human")
    parser.add_argument('--cell_type_key', type=str, default="cell_type")
    parser.add_argument('--split_key', type=str, default="split_yhr_TFcombos")
    parser.add_argument('--loss_ae', type=str, default='gauss')
    parser.add_argument('--doser_type', type=str, default='sigm')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--decoder_activation', type=str, default='linear')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--binary_pert', type=bool, default=True)

    # network arguments
    parser.add_argument('--regulon_name', type=str,
                        default='Norman2019_ctrl_only')
    parser.add_argument('--adjacency', type=str,
                        default='/dfs/user/yhr/cell_reprogram/Data'
                    '/learnt_weights/Norman2019_ctrl_only_learntweights.csv')

    # training arguments
    parser.add_argument('--device', type=str, default='cuda:4')
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--max_minutes', type=int, default=50)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--checkpoint_freq', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--node_hidden_size', type=int, default=16)
    parser.add_argument('--node_embed_size', type=int, default=8)
    parser.add_argument('--ae_hidden_size', type=int, default=512)
    parser.add_argument('--gnn_num_layers', type=int, default=4)
    parser.add_argument('--ae_num_layers', type=int, default=4)
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--num_itr', type=int, default=1)
    parser.add_argument('--workers', type=int, default=1)

    # output arguments
    parser.add_argument('--save_dir', type=str, default='./test/')
    return dict(vars(parser.parse_args()))


if __name__ == "__main__":
    trainer(parse_arguments())
