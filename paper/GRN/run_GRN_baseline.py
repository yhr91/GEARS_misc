import numpy as np
import pandas as pd
import scanpy as sc
import pickle 
from GRN_model import linear_model
from utils import parse_any_pert
from tqdm import tqdm
from data import PertDataloader
from pertdata import PertData
import torch

import warnings
warnings.filterwarnings("ignore")
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--dataset_name', type=str, default='norman', choices = ['norman', 'adamson2016', 'dixit2016', 'jost2020_hvg', 'tian2021_crispri_hvg', 'tian2021_crispra_hvg', 'replogle2020_hvg', 'replogle_rpe1_gw_hvg', 'replogle_k562_gw_hvg', 'replogle_k562_essential_hvg', 'tian2019_neuron_hvg', 'tian2019_ipsc_hvg', 'replogle_rpe1_gw_filtered_hvg', 'replogle_k562_essential_filtered_hvg'])
parser.add_argument('--graph_type', type=str, default='grnboost', choices = ['coexpression', 'grnboost', 'go'])

args = parser.parse_args()

data_path = '/dfs/project/perturb-gnn/datasets/data/'
split_num = args.seed
graph_type = args.graph_type
filter_ = 'top50'
method = 'linear'
dataset = args.dataset_name

if dataset in ['tian2019_neuron_hvg', 'tian2019_ipsc_hvg', 'jost2020_hvg']:
    dataset += '_small_graph'

if dataset == 'Norman2019':
    data_path = '/dfs/project/perturb-gnn/datasets/Norman2019/Norman2019_hvg+perts_more_de.h5ad'
elif dataset == 'replogle_k562_essential_filtered_hvg':
    data_path = '/dfs/project/perturb-gnn/datasets/data/replogle_k562_essential_filtered_hvg/perturb_processed_linear.h5ad'
else:
    data_path = '/dfs/project/perturb-gnn/datasets/data/' + dataset + '/perturb_processed.h5ad'
    
adata = sc.read(data_path)
if 'gene_symbols' not in adata.var.columns.values:
    adata.var['gene_symbols'] = adata.var['gene_name']
    
if dataset == 'Norman2019':    
    split_path = '../data/Norman2019/splits/Norman2019_simulation_' + str(split_num) + '_0.1.pkl'
else:
    split_path = '/dfs/project/perturb-gnn/datasets/data/' + dataset + '/splits/' + dataset + '_simulation_' + str(split_num) + '_0.75.pkl'
    
split = pickle.load(open(split_path, 'rb'))

condition2set = {}
    
for i,j in split.items():
    for k in j:
        condition2set[k] = i
        
#adata.obs['split_status'] = [condition2set[i] for i in adata.obs.condition.values]
#adata_train = adata[adata.obs['split_status'] == 'train']

if dataset == 'norman':
    
    if graph_type == 'go':
        graph_path = '/dfs/project/perturb-gnn/graphs/linear/grnboost/go_essential_norman_filter_' + str(split_num) + '_linear.csv'
        weights_path = '/dfs/project/perturb-gnn/graphs/linear/grnboost/go_essential_norman_filter_' + str(split_num) + '_linear_learntweights.csv'
    else:
        graph_path = '/dfs/project/perturb-gnn/graphs/linear/' + graph_type + '/norman_' + str(split_num) + '_' + str(filter_) + '.csv'
        weights_path = '/dfs/project/perturb-gnn/graphs/linear/' + graph_type + '/norman_' + str(split_num) + '_' + str(filter_) + '_' + method + '_learntweights.csv'
    
    
elif dataset == 'replogle_rpe1_gw_filtered_hvg':
    graph_path = '/dfs/project/perturb-gnn/graphs/linear/grnboost/replogle_rpe1_gw_filtered_hvg_' + str(split_num) + '_subsample_top50.csv'
    weights_path = '/dfs/project/perturb-gnn/graphs/linear/grnboost/replogle_rpe1_gw_filtered_hvg_' + str(split_num) + '_subsample_top50_linear_learntweights.csv' 
elif dataset == 'replogle_k562_essential_filtered_hvg':
    graph_path = '/dfs/project/perturb-gnn/graphs/linear/grnboost/replogle_k562_essential_filtered_hvg_' + str(split_num) + '_subsample_top50.csv'
    weights_path = '/dfs/project/perturb-gnn/graphs/linear/grnboost/replogle_k562_essential_filtered_hvg_' + str(split_num) + '_subsample_top50_linear_learntweights.csv'
else:
    graph_path = '/dfs/project/perturb-gnn/graphs/linear/grnboost/' + args.dataset_name + '_' + str(split_num) + '_' + str(filter_) + '.csv'
    weights_path = '/dfs/project/perturb-gnn/graphs/linear/grnboost/' + args.dataset_name + '_' + str(split_num) + '_' + str(filter_) + '_' + method + '_learntweights.csv'
    
    
gene_list = adata.var.gene_name.values

model = linear_model(graph_path=graph_path, 
             weights_path=weights_path, 
             gene_list = gene_list,
             binary=False, 
             pos_edges=False, 
             hops=1,
             species='human')

#args = np.load('./saved_args/pertnet_uncertainty_ori.npy', allow_pickle = True).item()

#args['dataset'] = dataset
#args['seed'] = split_num
#args['test_pert_genes'] = 'N/A'

#pertdl = PertDataloader(adata, args)


if args.dataset_name == 'tian2019_neuron_hvg':
    gene_path = '/dfs/user/kexinh/gears2/data/essential_all_data_pert_genes_tian2019_neuron.pkl'
elif args.dataset_name == 'tian2019_ipsc_hvg':
    gene_path = '/dfs/user/kexinh/gears2/data/essential_all_data_pert_genes_tian2019_ipsc.pkl'
elif args.dataset_name == 'jost2020_hvg':
    gene_path = '/dfs/user/kexinh/gears2/data/essential_all_data_pert_genes_jost2020.pkl'
else:
    gene_path = None
    
pert_data = PertData('/dfs/project/perturb-gnn/datasets/data', gene_path = gene_path) # specific saved folder
pert_data.load(data_path = '/dfs/project/perturb-gnn/datasets/data/' + dataset) 

pert_data.prepare_split(split = 'simulation', seed = split_num)
pert_data.get_dataloader(batch_size = 128, test_batch_size = 128)

pred_delta = {pert: model.simulate_pert(parse_any_pert(pert)) for pert in split['test']}
adata_ctrl = adata[adata.obs.condition == 'ctrl']


pert_cat = []
pred = []
truth = []
pred_de = []
truth_de = []
results = {}

for batch in tqdm(pert_data.dataloader['test_loader']):
    
    pert_cat.extend(batch.pert)
    p = np.array([pred_delta[i]+adata_ctrl.X[np.random.randint(0, adata_ctrl.shape[0])].toarray().reshape(-1,) for i in batch.pert])
    t = batch.y

    pred.extend(p)
    truth.extend(t.cpu())

    # Differentially expressed genes
    for itr, de_idx in enumerate(batch.de_idx):
        pred_de.append(p[itr, de_idx])
        truth_de.append(t[itr, de_idx])
        
# all genes
results['pert_cat'] = np.array(pert_cat)

pred = np.stack(pred)
truth = torch.stack(truth)
results['pred']= pred
results['truth']= truth.detach().cpu().numpy()

pred_de = np.stack(pred_de)
truth_de = torch.stack(truth_de)
results['pred_de']= pred_de
results['truth_de']= truth_de.detach().cpu().numpy()


from inference_new import evaluate, compute_metrics, deeper_analysis, GI_subgroup, non_dropout_analysis

test_metrics, test_pert_res = compute_metrics(results)
test_res = results

import wandb 

wandb.init(project='linear_model', name= '_'.join([args.dataset_name, str(split_num)]))

args = {}
args['wandb'] = True

out = deeper_analysis(adata, test_res)
out_non_dropout = non_dropout_analysis(adata, test_res)

metrics = ['pearson_delta']
metrics_non_dropout = ['frac_opposite_direction_top20_non_dropout', 'frac_sigma_below_1_non_dropout', 'mse_top20_de_non_dropout']
    
if args['wandb']:
    for m in metrics:
        wandb.log({'test_' + m: np.mean([j[m] for i,j in out.items() if m in j])})

    for m in metrics_non_dropout:
        wandb.log({'test_' + m: np.mean([j[m] for i,j in out_non_dropout.items() if m in j])})        

if dataset == 'Norman2019':
    subgroup_path = './splits/' + dataset_name + '_simulation_' + str(split_num) + '_0.1_subgroup.pkl'
else:
    subgroup_path = '/dfs/project/perturb-gnn/datasets/data/' + dataset + '/splits/'+ dataset + '_simulation_' + str(split_num) + '_0.75_subgroup.pkl'

subgroup = pickle.load(open(subgroup_path, "rb"))

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
        wandb.log({'test_' + name + '_' + m: subgroup_analysis[name][m]})

## deeper analysis
subgroup_analysis = {}
for name in subgroup['test_subgroup'].keys():
    subgroup_analysis[name] = {}
    for m in metrics:
        subgroup_analysis[name][m] = []

    for m in metrics_non_dropout:
        subgroup_analysis[name][m] = []

for name, pert_list in subgroup['test_subgroup'].items():
    for pert in pert_list:
        for m in metrics:
            subgroup_analysis[name][m].append(out[pert][m])

        for m in metrics_non_dropout:
            subgroup_analysis[name][m].append(out_non_dropout[pert][m])

for name, result in subgroup_analysis.items():
    for m in result.keys():
        subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
        wandb.log({'test_' + name + '_' + m: subgroup_analysis[name][m]})

