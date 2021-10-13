import torch
import torch.optim as optim
import scanpy as sc
import numpy as np

import argparse
from model import simple_GNN, simple_GNN_AE
from data import PertDataloader, Network
from copy import deepcopy
from inference import evaluate, compute_metrics

import sys
sys.path.append('/dfs/user/yhr/cell_reprogram/model/')


def train(model, train_loader, val_loader, graph, weights, args,
          device="cpu", gene_idx=None):
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
            graph = graph.to(device)
            if weights is not None:
                weights = weights.to(device)
            model.to(device)
            optimizer.zero_grad()

            pred = model(batch, graph, weights)
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
        train_res = evaluate(train_loader, graph, weights, model,
                             args, gene_idx=gene_idx)
        val_res = evaluate(val_loader, graph, weights, model,
                           args, gene_idx=gene_idx)
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
    print(args)
    
    ## wandb exp name setup
    if args['GNN_simple']:
        exp_name = 'GNN_Simple'
    elif args['GNN_AE']:
        exp_name = 'GNN_AE'
        
        
    if args['wandb']:
        import wandb
        wandb.init(project=args['project_name'], entity=args['entity_name'], name=exp_name)
        wandb.config.update(args)
    
    adata = sc.read_h5ad(args['work_dir'] + 'datasets/' + args['fname'])
    gene_list = [f for f in adata.var.gene_symbols.values]

    args['gene_list'] = gene_list
    args['num_genes'] = len(gene_list)
    args['modelname'] = args['fname'].split('/')[-1].split('.h5ad')[0] + '_'\
                        + args['network_name'].split('/')[-1].split('.')[0]+'_'\
                        + str(args['top_edge_percent']) + '_' \
                        + args['data_suffix']

    print('Training '+ args['modelname'])
    print('Experiment:' + args['exp_name'])

    # Set up message passing network
    network = Network(fname=args['work_dir'] + 'graphs/' +args['network_name'],
                      gene_list=args['gene_list'],
                      percentile=args['top_edge_percent'])

    # Pertrubation dataloader
    pertdl = PertDataloader(adata, network.G, network.weights, args)

    # Compute number of features for each node
    item = [item for item in pertdl.loaders['train_loader']][0]
    args['num_node_features'] = item.x.shape[1]

    # Train a model
    all_test_pert_res = []
    # Run training 3 times to measure variability of result
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
                               pertdl.loaders['edge_index'],
                               pertdl.loaders['edge_attr'],
                               args, device=args["device"])

        test_res = evaluate(pertdl.loaders['test_loader'],
                            pertdl.loaders['edge_index'],
                            pertdl.loaders['edge_attr'],best_model, args)
        test_metrics, test_pert_res = compute_metrics(test_res)
        all_test_pert_res.append(test_pert_res)
        log = "Final best performing model" + str(itr) +\
              ": Test_DE: {:.4f}, R2 {:.4f} "
        print(log.format(test_metrics['mse_de'], test_metrics['r2_de']))
        
        if args['wandb']:
            wandb.log({'Test_DE_MSE': test_metrics['mse_de'],
                      'Test_R2': test_metrics['r2_de']})
        
    # Save model outputs and best model
    np.save(args['work_dir'] + 'saved_metrics/'+args['modelname']
            + '_'+ args['exp_name'],
            all_test_pert_res)
    np.save(args['work_dir'] + 'saved_args/'
            + args['modelname']
            + '_'+ args['exp_name'], args)
    torch.save(best_model, args['work_dir'] + 'saved_models/full_model_'
               +args['modelname']
               +'_'+ args['exp_name'])


def parse_arguments():
    """
    Argument parser
    """

    # dataset arguments
    parser = argparse.ArgumentParser(description='Perturbation response')
    parser.add_argument('--work_dir', type=str, default='/dfs/project/perturb-gnn/')
    parser.add_argument('--fname', type=str,
                        default='Norman2019_hvg+perts_combo_seen0_split.h5ad')
    parser.add_argument('--perturbation_key', type=str, default="condition")
    parser.add_argument('--species', type=str, default="human")
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--binary_pert', default=True, action='store_false')
    parser.add_argument('--edge_attr', default=True, action='store_false')
    parser.add_argument('--data_split', type=str, default=None,
            help='Split type: <combo/single>,<# seen genes>, eg: combo,0')
    parser.add_argument('--split_key', type=str, default="split")
    parser.add_argument('--save_single_graph', default=True,
                        action='store_true')

    # network arguments
    parser.add_argument('--network_name', type=str,
                        default='STRING_full_9606.csv', help='select network')
    parser.add_argument('--top_edge_percent', type=float, default=25,
                        help='percentile of top edges to retain for graph')

    # training arguments
    parser.add_argument('--device', type=str,
                        default='cuda:3')
                        #default=torch.device("cuda" if
                        # torch.cuda.is_available() else "cpu"))
    parser.add_argument('--max_epochs', type=int, default=20)
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
    parser.add_argument('--edge_weights', action='store_true', default=False,
                        help='whether to include linear edge weights during '
                             'GNN training')
    parser.add_argument('--GNN', type=str, default='GraphConv',
                        help='Which GNN to use form GCN/GraphConv/GATConv')
    parser.add_argument('--loss_type', type=str, default='micro',
                        help='micro averaged or not')
    parser.add_argument('--encode', default=False, action='store_true',
                        help='whether to use AE after GNN, GNN_AE must be True')

    # Only one of these can be True
    parser.add_argument('--GNN_simple', default=True, action='store_true',
                        help='Use simple GNN')
    parser.add_argument('--GNN_AE',default=False, action='store_true',
                        help='Use GNN followed by AE')


    # Dataloader related
    parser.add_argument('--pert_feats', default=True, action='store_false',
                        help='Separate feature to indicate perturbation')
    parser.add_argument('--pert_delta', default=False, action='store_true',
                        help='Represent perturbed cells using delta gene '
                             'expression')
    parser.add_argument('--edge_filter', default=False, action='store_true',
                        help='Filter edges based on applied perturbation')
    parser.add_argument('--data_suffix', type=str, default='pert_feats',
                        help='Suffix to add to dataloader file and modelname')
    
    parser.add_argument('--wandb', default=False, action='store_true',
                    help='Use wandb or not')
    parser.add_argument('--project_name', type=str, default='pert_gnn_v1',
                        help='project name')
    parser.add_argument('--entity_name', type=str, default='kexinhuang',
                        help='entity name')
    
    return dict(vars(parser.parse_args()))


if __name__ == "__main__":
    trainer(parse_arguments())
