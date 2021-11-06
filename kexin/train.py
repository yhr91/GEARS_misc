from copy import deepcopy
import argparse
from time import time
import sys
sys.path.append('/dfs/user/yhr/cell_reprogram/model/')

import scanpy as sc
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from model import linear_model, simple_GNN, simple_GNN_AE, GNN_Disentangle, AE, No_Perturb, No_GNN
from data import PertDataloader, Network, GeneSimNetwork, GeneCoexpressNetwork
from inference import evaluate, compute_metrics, deeper_analysis, GI_subgroup
from utils import loss_fct, uncertainty_loss_fct, parse_any_pert

torch.manual_seed(0)


def train(model, train_loader, val_loader, graph, weights, args, device="cpu", gene_idx=None):
    best_model = deepcopy(model)
    if args['wandb']:
        import wandb
    optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    scheduler = StepLR(optimizer, step_size=args['lr_decay_step_size'], gamma=args['lr_decay_factor'])

    min_val = np.inf
    
    print('Start Training...')

    for epoch in range(args["max_epochs"]):
        total_loss = 0
        model.train()
        num_graphs = 0

        for step, batch in enumerate(train_loader):

            batch.to(device)
            graph = graph.to(device)
            if weights is not None:
                weights = weights.to(device)
            
            model.to(device)
            optimizer.zero_grad()
            y = batch.y
            if args['uncertainty']:
                pred, logvar = model(batch, graph, weights)
                loss = uncertainty_loss_fct(pred, logvar, y, batch.pert,
                                  loss_mode = args['loss_mode'], 
                                  gamma = args['focal_gamma'],
                                  reg = args['uncertainty_reg'],
                                  reg_core = args['uncertainty_reg_core'])
            else:
                pred = model(batch, graph, weights)
                # Compute loss
                loss = loss_fct(pred, y, batch.pert, args['pert_loss_wt'], 
                                  loss_mode = args['loss_mode'], 
                                  gamma = args['focal_gamma'],
                                  loss_type = args['loss_type'])
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            num_graphs += batch.num_graphs
            
            if args['wandb']:
                wandb.log({'training_loss': loss.item()})
                
            if step % args["print_progress_steps"] == 0:
                log = "Epoch {} Step {} Train Loss: {:.4f}" 
                print(log.format(epoch + 1, step + 1, loss.item()))
        scheduler.step()
        # Evaluate model performance on train and val set
        total_loss /= num_graphs
        train_res = evaluate(train_loader, graph, weights, model, args, gene_idx=gene_idx)
        val_res = evaluate(val_loader, graph, weights, model, args, gene_idx=gene_idx)
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
            metrics = ['mse', 'mae', 'spearman', 'pearson', 'r2']
            for m in metrics:
                wandb.log({'train_' + m: train_metrics[m],
                           'val_'+m: val_metrics[m],
                           'train_de_' + m: train_metrics[m + '_de'],
                           'val_de_'+m: val_metrics[m + '_de']})
        
        
        # Print epoch performance for DE genes
        log = "DE_Train: {:.4f}, R2 {:.4f} " \
              "DE_Validation: {:.4f}. R2 {:.4f} "
        print(log.format(train_metrics['mse_de'], train_metrics['r2_de'],
                         val_metrics['mse_de'], val_metrics['r2_de']))
    
            
        # Select best model
        if val_metrics['mse_de'] < min_val:
            min_val = val_metrics['mse_de']
            best_model = deepcopy(model)

    return best_model


def trainer(args):
    print('---- Printing Arguments ----')
    for i, j in args.items():
        print(i + ': ' + str(j))
    print('----------------------------')
        
    ## exp name setup
    exp_name = args['model']  + '_' + args['model_backend'] + '_' + args['network_name'] + '_' + str(args['top_edge_percent']) + '_' + str(args['node_hidden_size']) + '_' + str(args['gnn_num_layers']) + '_' + args['loss_mode'] + '_' + args['dataset']
    
    if args['loss_mode'] == 'l3':
        exp_name += '_gamma' + str(args['focal_gamma'])

    if args['shared_weights']:
        exp_name += '_shared'
    
    if args['ctrl_remove_train']:
        exp_name += '_no_ctrl'
    
    if args['gene_emb']:
        exp_name += '_gene_emb'
        
    if args['gene_pert_agg'] == 'concat+w':
        exp_name += '_concat+w'
    
    if args['delta_predict']:
        exp_name += '_delta_predict'

    if args['delta_predict_with_gene']:
        exp_name += '_delta_predict_with_gene'        
        
    if args['pert_emb']:
        exp_name += '_pert_emb_'    
        exp_name += args['pert_emb_agg']

    if args['lambda_emission']:
        exp_name += '_lambda_emission'
    
    if args['sim_gnn']:
        exp_name += '_sim_gnn'
    
    if args['test_perts'] != 'N/A':
        exp_name += '_' + args['test_perts']

    if args['randomize_network']:
        exp_name += '_randomize'
        
    if args['uncertainty']:
        exp_name += '_uncertainty'
        exp_name += '_' + str(args['uncertainty_reg'])
        exp_name += '_' + str(args['uncertainty_reg_core'])
    
    args['model_name'] = exp_name
    
    if args['wandb']:
        import wandb 
        if not args['wandb_sweep']:
            if args['exp_name'] != 'N/A':
                wandb.init(project=args['project_name'] + '_' + args['split'] + '_' + args['dataset'], entity=args['entity_name'], name=args['exp_name'])
            else:
                wandb.init(project=args['project_name'] + '_' + args['split'] + '_' + args['dataset'], entity=args['entity_name'], name=exp_name)
            wandb.config.update(args)
        
    if args['network_name'] == 'string':
        args['network_path'] = '/dfs/project/perturb-gnn/graphs/STRING_full_9606.csv'
    elif args['network_name'] == 'co-expression':
        args['network_path'] = 'co_expression.csv'
        
    if args['dataset'] == 'Norman2019':
        data_path = '/dfs/project/perturb-gnn/datasets/Norman2019/Norman2019_hvg+perts_more_de.h5ad'
    elif args['dataset'] == 'Adamson2016':
        data_path = '/dfs/project/perturb-gnn/datasets/Adamson2016_hvg+perts_more_de.h5ad'
    elif args['dataset'] == 'Dixit2016':
        data_path = '/dfs/project/perturb-gnn/datasets/Dixit2016_hvg+perts_more_de.h5ad'
    
    s = time()
    adata = sc.read_h5ad(data_path)
    if 'gene_symbols' not in adata.var.columns.values:
        adata.var['gene_symbols'] = adata.var['gene_name']
    gene_list = [f for f in adata.var.gene_symbols.values]
    args['gene_list'] = gene_list
    args['num_genes'] = len(gene_list)
    
    try:
        args['num_ctrl_samples'] = adata.uns['num_ctrl_samples']
    except:
        args['num_ctrl_samples'] = 1

    print('Training '+ args['model_name'])
    print('Building cell graph... ')

    # Set up message passing network
    network = Network(fname=args['network_path'], gene_list=args['gene_list'],
                      percentile=args['top_edge_percent'], randomize = args['randomize_network'])

    # Pertrubation dataloader
    pertdl = PertDataloader(adata, network.G, network.weights, args)
    
    if args['sim_gnn']:
        if args['sim_graph'] == 'knn_go_pathway':
            fname = '/dfs/user/kexinh/perturb_GNN/kexin/gene_sim_knn.csv'
        elif args['sim_graph'] == 'go_pathway':
            fname = '/dfs/user/kexinh/perturb_GNN/kexin/gene_sim_jc_filter.csv'
        sim_network = GeneSimNetwork(fname, args['gene_list'], node_map = pertdl.node_map)
        args['G_sim'] = sim_network.edge_index
        args['G_sim_weight'] = sim_network.edge_weight
        
        #fname = 'co_expression.csv'
        #genexp_network = GeneCoexpressNetwork(fname, args['gene_list'], node_map = pertdl.node_map)
        #args['G_coexpress'] = genexp_network.edge_index
        #args['G_coexpress_weight'] = genexp_network.edge_weight
        
    if args['lambda_emission']:
        set2conditions = pertdl.set2conditions
        gene2occurence = {}
        for i in gene_list:
            gene2occurence[i] = 0

        for i in set2conditions['train']:
            if i != 'ctrl':
                for j in parse_any_pert(i):
                    gene2occurence[j] = 1

        args['occurence_bit'] = gene2occurence
        args['inv_node_map'] = {j:i for i, j in pertdl.node_map.items()}
    
    if args['pert_emb_agg'] == 'occurence':
        set2conditions = pertdl.set2conditions
        gene2occurence = {}
        for i in gene_list:
            gene2occurence[i] = 0

        for i in set2conditions['train']:
            if i != 'ctrl':
                for j in parse_any_pert(i):
                    gene2occurence[j] += 1

        ratio = np.mean(np.sort(np.array(list(gene2occurence.values())))[-10:])
        gene2occurence = {i: j/ratio + 0.05 for i,j in gene2occurence.items()}
        args['gene_occurence'] = gene2occurence
        args['inv_node_map'] = {j:i for i, j in pertdl.node_map.items()}
        
    # Compute number of features for each node
    item = [item for item in pertdl.loaders['train_loader']][0]
    args['num_node_features'] = item.x.shape[1]
    print('Finished data setup, in total takes ' + str((time() - s)/60)[:5] + ' min')
    
    print('Initializing model... ')
    
    # Train a model
    # Define model
    if args['model'] == 'GNN_simple':
        model = simple_GNN(args['num_node_features'],
                           args['num_genes'],
                           args['node_hidden_size'],
                           args['node_embed_size'],
                           args['edge_weights'],
                           args['loss_type'])

    elif args['model'] == 'GNN_AE':
        model = simple_GNN_AE(args['num_node_features'],
                           args['num_genes'],
                           args['node_hidden_size'],
                           args['node_embed_size'],
                           args['edge_weights'],
                           args['ae_num_layers'],
                           args['ae_hidden_size'],
                           args['loss_type'])
    elif args['model'] == 'GNN_Disentangle':
        model = GNN_Disentangle(args, args['num_node_features'],
                           args['num_genes'],
                           args['node_hidden_size'],
                           args['node_embed_size'],
                           args['edge_weights'],
                           args['ae_num_layers'],
                           args['ae_hidden_size'],
                           args['loss_type'],
                           ae_decoder = False,
                           shared_weights = args['shared_weights'],
                           model_backend = args['model_backend'],
                           num_layers = args['gnn_num_layers'])
    elif args['model'] == 'No_GNN':
        model = No_GNN(args, args['num_node_features'],
                           args['num_genes'],
                           args['node_hidden_size'],
                           args['node_embed_size'],
                           args['edge_weights'],
                           args['ae_num_layers'],
                           args['ae_hidden_size'],
                           args['loss_type'],
                           ae_decoder = False,
                           shared_weights = args['shared_weights'],
                           model_backend = args['model_backend'],
                           num_layers = args['gnn_num_layers'])
    elif args['model'] == 'GNN_Disentangle_AE':
        model = GNN_Disentangle(args, args['num_node_features'],
                           args['num_genes'],
                           args['node_hidden_size'],
                           args['node_embed_size'],
                           args['edge_weights'],
                           args['ae_num_layers'],
                           args['ae_hidden_size'],
                           args['loss_type'],
                           ae_decoder = True,
                           shared_weights = args['shared_weights'],
                           model_backend = args['model_backend'],
                           num_layers = args['gnn_num_layers'])
    elif args['model'] == 'AE':
        model = AE(args['num_node_features'],
                       args['num_genes'],
                       args['node_hidden_size'],
                       args['node_embed_size'],
                       args['ae_num_layers'],
                       args['ae_hidden_size'],
                       args['loss_type'])
        
    elif args['model'] == 'No_Perturb':
        model = No_Perturb()
    
    
    if args['model'] == 'No_Perturb': 
        best_model = model
    else:
        best_model = train(model, pertdl.loaders['train_loader'],
                              pertdl.loaders['val_loader'],
                              pertdl.loaders['edge_index'],
                              pertdl.loaders['edge_attr'],
                              args, device=args["device"])
    
    print('Saving model....')
        
    # Save model outputs and best model
    np.save('./saved_args/'+ args['model_name'], args)
    torch.save(best_model, './saved_models/' +args['model_name'])
    
    
    print('Start testing....')
    test_res = evaluate(pertdl.loaders['test_loader'],
                            pertdl.loaders['edge_index'],
                            pertdl.loaders['edge_attr'],best_model, args)
    
    test_metrics, test_pert_res = compute_metrics(test_res)
    np.save('./saved_metrics/'+args['model_name'],test_pert_res)
    
    log = "Final best performing model: Test_DE: {:.4f}, R2 {:.4f} "
    print(log.format(test_metrics['mse_de'], test_metrics['r2_de']))
    if args['wandb']:
        metrics = ['mse', 'mae', 'spearman', 'pearson', 'r2']
        for m in metrics:
            wandb.log({'test_' + m: test_metrics[m],
                       'test_de_'+m: test_metrics[m + '_de']
                       #'test_de_macro_'+m: test_metrics[m + '_de_macro'],
                       #'test_macro_'+m: test_metrics[m + '_macro'],                       
                      })
    
    out = deeper_analysis(adata, test_res)
    GI_out = GI_subgroup(out)
    
    
    metrics = ['frac_in_range', 'frac_in_range_45_55', 'frac_in_range_40_60', 'frac_in_range_25_75', 'mean_sigma', 'std_sigma', 'frac_sigma_below_1', 'frac_sigma_below_2', 'pearson_delta',
               'pearson_delta_de', 'fold_change_gap_all', 'pearson_delta_top200_hvg', 'fold_change_gap_upreg_3', 
               'fold_change_gap_downreg_0.33', 'fold_change_gap_downreg_0.1', 'fold_change_gap_upreg_10', 
               'pearson_top200_hvg', 'pearson_top200_de', 'pearson_top20_de', 'pearson_delta_top200_de', 
               'pearson_top100_de', 'pearson_delta_top100_de', 'pearson_delta_top50_de', 'pearson_top50_de', 'pearson_delta_top20_de',
               'mse_top200_hvg', 'mse_top100_de', 'mse_top200_de', 'mse_top50_de', 'mse_top20_de']
    
    
    if args['wandb']:
        for m in metrics:
            wandb.log({'test_' + m: np.mean([j[m] for i,j in out.items() if m in j])})
    
    if args['split'] == 'simulation':
        subgroup = pertdl.subgroup
        subgroup_analysis = {}
        for name in subgroup['test_subgroup'].keys():
            subgroup_analysis[name] = {}
            for m in list(list(test_pert_res.values())[0].keys()):
                subgroup_analysis[name][m] = []

        for name, pert_list in subgroup['test_subgroup'].items():
            for pert in pert_list:
                for m, res in test_pert_res[pert].items():
                    subgroup_analysis[name][m].append(res)

        for name, result in subgroup_analysis.items():
            for m in result.keys():
                subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
                if args['wandb']:
                    wandb.log({'test_' + name + '_' + m: subgroup_analysis[name][m]})

                print('test_' + name + '_' + m + ': ' + str(subgroup_analysis[name][m]))
        
        ## deeper analysis
        subgroup_analysis = {}
        for name in subgroup['test_subgroup'].keys():
            subgroup_analysis[name] = {}
            for m in metrics:
                subgroup_analysis[name][m] = []

        for name, pert_list in subgroup['test_subgroup'].items():
            for pert in pert_list:
                for m, res in out[pert].items():
                    subgroup_analysis[name][m].append(res)

        for name, result in subgroup_analysis.items():
            for m in result.keys():
                subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
                if args['wandb']:
                    wandb.log({'test_' + name + '_' + m: subgroup_analysis[name][m]})

                print('test_' + name + '_' + m + ': ' + str(subgroup_analysis[name][m]))
    
    for i,j in GI_out.items():
        for m in  ['mean_sigma', 'frac_in_range_45_55', 'frac_in_range_40_60', 'frac_in_range_25_75', 
               'fold_change_gap_all', 'pearson_delta_top200_de', 'pearson_delta_top100_de',  'pearson_delta_top50_de',
               'mse_top200_de', 'mse_top100_de', 'mse_top50_de', 'mse_top20_de', 'pearson_delta_top20_de']:
            if args['wandb']:
                wandb.log({'test_' + i + '_' + m: j[m]})


    print('Done!')


def parse_arguments():
    """
    Argument parser
    """

    # dataset arguments
    parser = argparse.ArgumentParser(description='Perturbation response')
    
    parser.add_argument('--dataset', type=str, choices = ['Norman2019', 'Adamson2016', 'Dixit2016'], default="Norman2019")
    parser.add_argument('--split', type=str, choices = ['simulation', 'simulation_single', 'combo_seen0', 'combo_seen1', 'combo_seen2', 'single', 'single_only'], default="combo_seen0")
    parser.add_argument('--seed', type=int, default=1)    
    parser.add_argument('--test_set_fraction', type=float, default=0.1)
    parser.add_argument('--train_gene_set_size', type=float, default=0.75)
    parser.add_argument('--combo_seen2_train_frac', type=float, default=0.75)
    parser.add_argument('--test_perts', type=str, default='N/A')
    parser.add_argument('--only_test_set_perts', default=False, action='store_true')
    
    parser.add_argument('--perturbation_key', type=str, default="condition")
    parser.add_argument('--species', type=str, default="human")
    parser.add_argument('--binary_pert', default=True, action='store_false')
    parser.add_argument('--edge_attr', default=False, action='store_true')
    parser.add_argument('--ctrl_remove_train', default=False, action='store_true')
    parser.add_argument('--edge_weights', action='store_true', default=False,
                        help='whether to include linear edge weights during '
                             'GNN training')
    # Dataloader related
    parser.add_argument('--pert_feats', default=True, action='store_false',
                        help='Separate feature to indicate perturbation')
    parser.add_argument('--pert_delta', default=False, action='store_true',
                        help='Represent perturbed cells using delta gene '
                             'expression')
    parser.add_argument('--edge_filter', default=False, action='store_true',
                        help='Filter edges based on applied perturbation')
    
    # network arguments
    parser.add_argument('--network_name', type=str, default = 'string', choices = ['string', 'co-expression'])
    parser.add_argument('--top_edge_percent', type=float, default=10,
                        help='percentile of top edges to retain for graph')
    parser.add_argument('--randomize_network', default = False, action = 'store_true')

    
    # training arguments
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--lr_decay_step_size', type=int, default=1)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--print_progress_steps', type=int, default=50)
                        
    # model arguments
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
    
    parser.add_argument('--model', choices = ['GNN_simple', 'GNN_AE', 'GNN_Disentangle', 'GNN_Disentangle_AE', 'AE', 'No_Perturb', 'No_GNN'], 
                        type = str, default = 'GNN_AE', help='model name')
    parser.add_argument('--model_backend', choices = ['GCN', 'GAT', 'DeepGCN'], 
                        type = str, default = 'GAT', help='model name')    
    parser.add_argument('--shared_weights', default=False, action='store_true',
                    help='Separate feature to indicate perturbation')                    
    parser.add_argument('--gene_specific', default=False, action='store_true',
                    help='Separate feature to indicate perturbation')                    
    parser.add_argument('--gene_emb', default=False, action='store_true',
                help='Separate feature to indicate perturbation')   
    parser.add_argument('--pert_emb', default=False, action='store_true',
                help='Separate feature to indicate perturbation')   
    parser.add_argument('--gene_pert_agg', default='sum', choices = ['sum', 'concat+w'], type = str)
    parser.add_argument('--delta_predict', default=False, action='store_true')   
    parser.add_argument('--delta_predict_with_gene', default=False, action='store_true')
    parser.add_argument('--pert_emb_lambda', type=float, default=0.2)
    parser.add_argument('--pert_emb_agg', type=str, default='constant', choices = ['constant', 'learnable', 'occurence'])
    parser.add_argument('--lambda_emission', default=False, action='store_true')
    parser.add_argument('--sim_gnn', default=False, action='store_true')
    parser.add_argument('--sim_graph', default='knn_go_pathway', type = str, choices = ['knn_go_pathway', 'go_pathway'])
    parser.add_argument('--gat_num_heads', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batchnorm', default=False, action='store_true')
    parser.add_argument('--bn_eps', type=float, default=1e-5)
    parser.add_argument('--bn_mom', type=float, default=0.1)
    parser.add_argument('--activation', type=str, default = 'relu', choices=['relu', 'parametric-relu'])
    parser.add_argument('--skipsum', default=False, action='store_true')
    parser.add_argument('--uncertainty', default=False, action='store_true')
    parser.add_argument('--uncertainty_reg', type=float, default=1)
    parser.add_argument('--uncertainty_reg_core', type=float, default=1)
    parser.add_argument('--no_gnn', default=False, action='store_true')
    
    # ablation analysis
    
    parser.add_argument('--no_pert_emb', default=False, action='store_true')
    parser.add_argument('--no_disentangle', default=False, action='store_true')
    
    # loss
    parser.add_argument('--pert_loss_wt', type=int, default=1,
                        help='weights for perturbed cells compared to control cells')
    parser.add_argument('--loss_type', type=str, default='macro', choices = ['macro', 'micro'],
                        help='micro averaged or not')
    parser.add_argument('--loss_mode', choices = ['l2', 'l3'], type = str, default = 'l2')
    parser.add_argument('--focal_gamma', type=int, default=2)    
    
    # wandb related
    parser.add_argument('--wandb', default=False, action='store_true',
                    help='Use wandb or not')
    parser.add_argument('--wandb_sweep', default=False, action='store_true',
                help='Use wandb or not')
    parser.add_argument('--project_name', type=str, default='pert_gnn',
                        help='project name')
    parser.add_argument('--entity_name', type=str, default='kexinhuang',
                        help='entity name')
    parser.add_argument('--exp_name', type=str, default='N/A',
                        help='entity name')
    return dict(vars(parser.parse_args()))


if __name__ == "__main__":
    trainer(parse_arguments())
