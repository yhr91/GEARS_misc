import torch
import torch.optim as optim
import scanpy as sc
import numpy as np

import argparse
from model import GNN_AE, linear_model, simple_GNN, simple_GNN_AE
from data import PertDataloader
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from copy import deepcopy

import sys
sys.path.append('/dfs/user/yhr/cell_reprogram/model/')


def zero_params(model):
    """
    Zeros out all parameters of the model
    Return: None
    """
    state_dict = model.state_dict()
    for k in state_dict.keys():
        state_dict[k] = torch.zeros(state_dict[k].shape)
    model.load_state_dict(state_dict)


def check_node_predictable(loader, gene_idx):
    """
    Checks if node is predictable
    """
    gene_vals = []
    for it, t in enumerate(loader):
        # Check if no incoming edges to target gene
        if it == 0:
            if len(np.where(t.edge_index[1] == gene_idx)[0]) == 0:
                return False
        gene_vals.append(np.mean((t.y[:, gene_idx] != 0).numpy()))

    # Check if target gene is always zero
    if np.mean(gene_vals) <= 0.01:
        return False
    return True


def train(model, train_loader, val_loader, args, device="cpu", gene_idx=None):
    optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=5e-4)
    min_val = np.inf

    # Don't waste time training unpredictable target node
    if gene_idx is not None:
        if not check_node_predictable(train_loader, gene_idx):
            zero_params(model)
            return model

    for epoch in range(args["max_epochs"]):
        total_loss = 0
        model.train()
        num_graphs = 0

        for batch in train_loader:

            # Make correction to feature set
            if gene_idx is not None and args['zero_target_node']:
                x_idx = [(gene_idx + args['num_genes'] * itr)
                         for itr in range(batch.num_graphs)]
                batch.x[x_idx, :] = 0

            batch.to(device)
            model.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            y = batch.y

            # Direct gradients through only specific gene
            if gene_idx is not None:
                if not args['single_out']:
                    pred = pred[:,gene_idx]
                y = y[:,gene_idx]

            # Compute loss
            loss = model.loss(pred, y, batch.pert, args['pert_loss_wt'])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            num_graphs += batch.num_graphs

        # Evaluate model performance on train and val set
        total_loss /= num_graphs
        train_res = evaluate(train_loader, model, args, gene_idx=gene_idx)
        val_res = evaluate(val_loader, model, args, gene_idx=gene_idx)
        train_metrics, _ = compute_metrics(train_res, gene_idx=gene_idx)
        val_metrics, _ = compute_metrics(val_res, gene_idx=gene_idx)

        # Print epoch performance
        log = "Epoch {}: Train: {:.4f}, R2 {:.4f} " \
              "Validation: {:.4f}. R2 {:.4f} " \
              "Loss: {:.4f}"
        print(log.format(epoch + 1, train_metrics['mse'], train_metrics['r2'],
                         val_metrics['mse'], val_metrics['r2'],
                         total_loss))

        # Print epoch performance for DE genes
        if gene_idx is None:
            log = "DE_Train: {:.4f}, R2 {:.4f} " \
                  "DE_Validation: {:.4f}. R2 {:.4f} "
            print(log.format(train_metrics['mse_de'], train_metrics['r2_de'],
                             val_metrics['mse_de'], val_metrics['r2_de']))

        # Select best model
        if val_metrics['mse'] < min_val:
            min_val = val_metrics['mse']
            best_model = deepcopy(model)

    return best_model


def evaluate(loader, model, args, num_de_idx=20, gene_idx=None):
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
        batch.to(args['device'])
        model.to(args['device'])
        results = {}
        pert_cat.extend(batch.pert)

        with torch.no_grad():

            p = model(batch)
            t = batch.y

            if gene_idx is not None:
                if not args['single_out']:
                    p = p[:,gene_idx]
                t = t[:,gene_idx]

            pred.extend(p)
            truth.extend(t)

            # Differentially expressed genes
            if gene_idx is None:
                for itr, de_idx in enumerate(batch.de_idx):
                    if de_idx is not None:
                        pred_de.append(p[itr, de_idx])
                        truth_de.append(t[itr, de_idx])

                    else:
                        pred_de.append([torch.zeros(num_de_idx).to(args['device'])])
                        truth_de.append([torch.zeros(num_de_idx).to(args['device'])])

    # all genes
    results['pert_cat'] = np.array(pert_cat)

    pred = torch.stack(pred)
    truth = torch.stack(truth)
    results['pred']= pred.detach().cpu().numpy()
    results['truth']= truth.detach().cpu().numpy()

    if gene_idx is None:
        pred_de = torch.stack(pred_de)
        truth_de = torch.stack(truth_de)
        results['pred_de']= pred_de.detach().cpu().numpy()
        results['truth_de']= truth_de.detach().cpu().numpy()
    else:
        results['pred_de'] = pred_de
        results['truth_de'] = truth_de

    return results


def compute_metrics(results, gene_idx=None):
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
        if gene_idx is None:
            metrics['r2'].append(r2_score(results['pred'][p_idx].mean(0),
                                          results['truth'][p_idx].mean(0)))
            metrics['mse'].append(mse(results['pred'][p_idx].mean(0),
                                      results['truth'][p_idx].mean(0)))
        else:
            metrics['r2'].append(0)
            metrics['mse'].append((results['pred'][p_idx].mean(0) -
                                   results['truth'][p_idx].mean(0))**2)
        metrics_pert[pert]['r2'] = metrics['r2'][-1]
        metrics_pert[pert]['mse'] = metrics['mse'][-1]

        if pert != 'ctrl' and gene_idx is None:
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
    args['num_genes'] = len(gene_list)
    args['modelname'] = args['fname'].split('/')[-1].split('.h5ad')[0] + \
                        args['data_suffix']

    try:
        args['num_ctrl_samples'] = adata.uns['num_ctrl_samples']
    except:
        args['num_ctrl_samples'] = 1

    print('Training '+ args['modelname'] + '_' + args['exp_name'])

    # Set up data loaders
    l_model = linear_model(args)
    pertdl = PertDataloader(adata, l_model.G, l_model.read_weights, args)

    # Compute number of features for each node
    item = [item for item in pertdl.loaders['train_loader']][0]
    num_node_features = item.x.shape[1]

    # Train a model
    all_test_pert_res = []
    # Run training 3 times to measure variability of result
    # TODO check if random seeds need to be changed here
    for itr in range(args['num_itr']):

        # Define model
        if args['GNN_node_specific']:
            best_models = {}
            start, stop = args['train_genes'].split(',')
            start = int(start)
            stop = int(stop)
            for idx in range(start,stop):
                print('Gene ' + str(idx))
                if args['AE']:
                    model = simple_GNN_AE(num_node_features, args['num_genes'],
                                   args['node_hidden_size'],
                                   args['node_embed_size'],
                                   args['edge_weights'],
                                   args['ae_num_layers'],
                                   args['ae_hidden_size'],
                                   args['loss_type'])
                else:
                    model = simple_GNN(num_node_features, args['num_genes'],
                               args['node_hidden_size'],
                               args['node_embed_size'],
                               args['edge_weights'],
                               args['loss_type'])

                best_models[idx] = train(model, pertdl.loaders['train_loader'],
                                   pertdl.loaders['val_loader'],
                                   args, device=args["device"], gene_idx=idx)

        elif args['GNN_simple']:
            model = simple_GNN(num_node_features, args['num_genes'],
                               args['node_hidden_size'],
                               args['node_embed_size'],
                               args['edge_weights'],
                               args['loss_type'])

        elif args['GNN_AE']:
            model = simple_GNN_AE(num_node_features, args['num_genes'],
                               args['node_hidden_size'],
                               args['node_embed_size'],
                               args['edge_weights'],
                               args['ae_num_layers'],
                               args['ae_hidden_size'],
                               args['loss_type'])

        if not args['GNN_node_specific']:
            best_model = train(model, pertdl.loaders['train_loader'],
                               pertdl.loaders['val_loader'],
                               args, device=args["device"])

            test_res = evaluate(pertdl.loaders['ood_loader'], best_model, args)
            test_metrics, test_pert_res = compute_metrics(test_res)

        else:
            # Node specific model needs special testing
            test_res = {}
            test_res['pred'] = []
            test_res['pred_de'] = []
            for idx in range(start, stop):
                test_res_gene = evaluate(pertdl.loaders['ood_loader'],
                                    best_models[idx], args, gene_idx=idx)
                test_res['pred'].append(test_res_gene['pred'])
                test_res['pred_de'].append(test_res_gene['pred_de'])
            test_res['pred'] = np.vstack(test_res['pred']).T
            test_res['pred_de'] = np.vstack(test_res['pred_de']).T
            test_res['truth'] = test_res_gene['truth']
            test_res['truth_de'] = test_res_gene['truth_de']
            test_res['pert_cat'] = test_res_gene['pert_cat']
            test_metrics, test_pert_res = compute_metrics(test_res, gene_idx=-1)

            # For saving
            best_model = best_models

        all_test_pert_res.append(test_pert_res)
        log = "Final best performing model" + str(itr) +\
              ": Test_DE: {:.4f}, R2 {:.4f} "
        print(log.format(test_metrics['mse_de'], test_metrics['r2_de']))

    # Save model outputs and best model
    np.save('./saved_metrics/'+args['modelname']
            + '_'+ args['exp_name']
            + args['train_genes'],
            all_test_pert_res)
    np.save('./saved_args/'
            + args['modelname']
            + '_'+ args['exp_name']
            + args['train_genes'], args)
    torch.save(best_model, './saved_models/full_model_'
               +args['modelname']
               +'_'+ args['exp_name']
               + args['train_genes'])


def parse_arguments():
    """
    Argument parser
    """

    # dataset arguments
    parser = argparse.ArgumentParser(description='Perturbation response')
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
