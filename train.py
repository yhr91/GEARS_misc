import torch
import torch.optim as optim
import scanpy as sc
import numpy as np

import argparse
from model import linear_model, simple_GNN, simple_GNN_AE
from data import PertDataloader, Network
from copy import deepcopy
from inference import evaluate, compute_metrics

import sys
sys.path.append('/dfs/user/yhr/cell_reprogram/model/')


def train(model, train_loader, val_loader, args, device="cpu", gene_idx=None):
    if args['wandb']:
        import wandb
        
    optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=5e-4)
    min_val = np.inf
    
    print('Start Training...')
    
    for epoch in range(args["max_epochs"]):
        total_loss = 0
        model.train()
        num_graphs = 0

        for batch in train_loader:

            batch.to(device)
            model.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            y = batch.y

            # Compute loss
            loss = model.loss(pred, y, batch.pert, args['pert_loss_wt'])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            num_graphs += batch.num_graphs
            
            if args['wandb']:
                wandb.log({'training_loss': loss.item()})
        
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
        
        if args['wandb']:
            wandb.log({'train_mse': train_metrics['mse'],
                     'train_r2': train_metrics['r2'],
                     'val_mse': val_metrics['mse'],
                     'val_r2': val_metrics['r2']})
        
        
        # Print epoch performance for DE genes
        log = "DE_Train: {:.4f}, R2 {:.4f} " \
              "DE_Validation: {:.4f}. R2 {:.4f} "
        print(log.format(train_metrics['mse_de'], train_metrics['r2_de'],
                         val_metrics['mse_de'], val_metrics['r2_de']))

        if args['wandb']:
            wandb.log({'train_de_mse': train_metrics['mse_de'],
                     'train_de_r2': train_metrics['r2_de'],
                     'val_de_mse': val_metrics['mse_de'],
                     'val_de_r2': val_metrics['r2_de']})
            
            
        # Select best model
        if val_metrics['mse'] < min_val:
            min_val = val_metrics['mse']
            best_model = deepcopy(model)

    return best_model


def trainer(args):
    
    ## wandb exp name setup
    
    if args['GNN_simple']:
        exp_name = 'GNN_Simple'
    elif args['GNN_AE']:
        exp_name = 'GNN_AE'
        
        
    if args['wandb']:
        import wandb
        wandb.init(project=args['project_name'], entity=args['entity_name'], name=exp_name)
        wandb.config.update(args)
    
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

    # Set up message passing network
    network = Network(args['network_name'], args['gene_list'])

    # Pertrubation dataloader
    pertdl = PertDataloader(adata, network, network.weights, args)

    # Compute number of features for each node
    item = [item for item in pertdl.loaders['train_loader']][0]
    args['num_node_features'] = item.x.shape[1]

    # Train a model
    all_test_pert_res = []
    # Run training 3 times to measure variability of result
    # TODO check if random seeds need to be changed here
    for itr in range(args['num_itr']):

        # Define model
        if args['GNN_simple']:
            model = simple_GNN(args['num_node_features'],
                               args['num_genes'],
                               args['node_hidden_size'],
                               args['node_embed_size'],
                               args['edge_weights'],
                               args['loss_type'])

        elif args['GNN_AE']:
            model = simple_GNN_AE(args['num_node_features'],
                               args['num_genes'],
                               args['node_hidden_size'],
                               args['node_embed_size'],
                               args['edge_weights'],
                               args['ae_num_layers'],
                               args['ae_hidden_size'],
                               args['loss_type'])

        best_model = train(model, pertdl.loaders['train_loader'],
                               pertdl.loaders['val_loader'],
                               args, device=args["device"])

        test_res = evaluate(pertdl.loaders['ood_loader'], best_model, args)
        test_metrics, test_pert_res = compute_metrics(test_res)
        all_test_pert_res.append(test_pert_res)
        log = "Final best performing model" + str(itr) +\
              ": Test_DE: {:.4f}, R2 {:.4f} "
        print(log.format(test_metrics['mse_de'], test_metrics['r2_de']))
        
        if args['wandb']:
            wandb.log({'Test_DE_MSE': test_metrics['mse_de'],
                      'Test_R2': test_metrics['r2_de']})
        
    # Save model outputs and best model
    np.save('./saved_metrics/'+args['modelname']
            + '_'+ args['exp_name'],
            all_test_pert_res)
    np.save('./saved_args/'
            + args['modelname']
            + '_'+ args['exp_name'], args)
    torch.save(best_model, './saved_models/full_model_'
               +args['modelname']
               +'_'+ args['exp_name'])


def parse_arguments():
    """
    Argument parser
    """

    # dataset arguments
    parser = argparse.ArgumentParser(description='Perturbation response')
    parser.add_argument('--fname', type=str,
                        default='/dfs/project/perturb-gnn/datasets/Norman2019_prep_new_TFcombosin5k_nocombo_somesingle_worstde_numsamples_1_new_method.h5ad')
    parser.add_argument('--perturbation_key', type=str, default="condition")
    parser.add_argument('--species', type=str, default="human")
    parser.add_argument('--split_key', type=str, default="split_yhr_TFcombos")
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--binary_pert', type=bool, default=True)
    parser.add_argument('--edge_attr', type=bool, default=True)

    # network arguments
    parser.add_argument('--network_name', type=str,
                        default='/dfs/project/perturb-gnn/graphs/STRING_full_9606.csv',
                        help='select network to use')

    # training arguments
    parser.add_argument('--device', type=str, default='cuda:8')
    parser.add_argument('--max_epochs', type=int, default=7)
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--node_hidden_size', type=int, default=2,
                        help='hidden dimension for GNN')
    parser.add_argument('--node_embed_size', type=int, default=1,
                        help='final node embedding size for GNN')
    parser.add_argument('--ae_hidden_size', type=int, default=512,
                        help='hidden dimension for AE')
    parser.add_argument('--gnn_num_layers', type=int, default=2,
                        help='number of layers in GNN')
    parser.add_argument('--ae_num_layers', type=int, default=2,
                        help='number of layers in autoencoder')
    parser.add_argument('--exp_name', type=str, default='full_GNN',
                        help='experiment name to be added to saved model file')
    parser.add_argument('--num_itr', type=int, default=1,
                        help='number of iterations of complete model trianing')
    parser.add_argument('--pert_loss_wt', type=int, default=1,
                        help='weights for perturbed cells compared to control cells')
    parser.add_argument('--edge_weights', type=bool, default=False,
                        help='whether to include linear edge weights during '
                             'GNN training')
    parser.add_argument('--GNN', type=str, default='GraphConv',
                        help='Which GNN to use form GCN/GraphConv/GATConv')
    parser.add_argument('--loss_type', type=str, default='micro',
                        help='micro averaged or not')
    parser.add_argument('--encode', type=bool, default=False,
                        help='whether to use AE after GNN, GNN_AE must be True')

    # Only one of these can be True
    parser.add_argument('--GNN_simple', type=bool, default=True,
                        help='Use simple GNN')
    parser.add_argument('--GNN_AE', type=bool, default=False,
                        help='Use GNN followed by AE')


    # Dataloader related
    parser.add_argument('--pert_feats', type=bool, default=True,
                        help='Separate feature to indicate perturbation')
    parser.add_argument('--pert_delta', type=bool, default=False,
                        help='Represent perturbed cells using delta gene '
                             'expression')
    parser.add_argument('--edge_filter', type=bool, default=False,
                        help='Filter edges based on applied perturbation')
    parser.add_argument('--data_suffix', type=str, default='_pert_feats',
                        help='Suffix to add to dataloader file and modelname')
    
    parser.add_argument('--wandb', type=bool, default=False,
                    help='Use wandb or not')
    parser.add_argument('--project_name', type=str, default='pert_gnn_v1',
                        help='project name')
    parser.add_argument('--entity_name', type=str, default='kexinhuang',
                        help='entity name')
    
    return dict(vars(parser.parse_args()))


if __name__ == "__main__":
    trainer(parse_arguments())
