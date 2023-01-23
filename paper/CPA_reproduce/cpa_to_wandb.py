from pprint import pprint
from compert.train import train_compert
from compert.data import load_dataset_splits
from compert.api import ComPertAPI
import pandas as pd
import scanpy as sc
import numpy as np

from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import pandas as pd
import numpy as np
import networkx as nx
import statsmodels.stats.api as sms
from scipy.stats import ncx2
from os.path import isfile
import scanpy as sc
from tqdm import tqdm
import sys
sys.path.append('/dfs/user/yhr/cell_reprogram/model/')
from flow import get_graph, get_expression_data,\
            add_weight, I_TF, get_TFs, solve,\
            solve_parallel, get_expression_lambda

# replogle2020_hvg tian2019_ipsc_hvg tian2019_neuron_hvg jost2020_hvg
dataset_name = 'replogle_rpe1_gw_filtered_hvg'
kg_mode = False
if kg_mode:
    kg_str = '_kg'
    emb_model = 'kg'
else:
    kg_str = ''
    emb_model = 'one_hot'
    
for split_num in range(1, 6):

    import wandb
    import pickle
    if dataset_name == 'Norman2019':
        wandb.init(project='pert_gnn_simulation_norman2019', entity='kexinhuang', name='CPA' + kg_str + '_seed' + str(split_num))
    else:
        wandb.init(project=dataset_name, entity='kexinhuang', name='CPA' + kg_str + '_seed' + str(split_num))

    import torch
    from compert.train import prepare_compert
    print(1)
    model_name = './compert/' + dataset_name + '_split' + str(split_num) + '/model' + kg_str + '_seed=0_epoch=499.pt'
    #state, args, history = torch.load(model_name, map_location=torch.device('cpu'))
    state  = torch.load(model_name, map_location=torch.device('cpu'))
    print(2)
    fname = '/dfs/project/perturb-gnn/datasets/data/' + dataset_name + '_simulation_cpa.h5ad'
    autoencoder_params = { 'adversary_depth': 4,
                              'adversary_lr': 0.0001875455179637405,
                              'adversary_steps': 3,
                              'adversary_wd': 0.00019718137187038062,
                              'adversary_width': 128,
                              'autoencoder_depth': 4,
                              'autoencoder_lr': 0.0011021870411382655,
                              'autoencoder_wd': 1.1455862519513426e-05,
                              'autoencoder_width': 512,
                              'batch_size': 128,
                              'dim': 256,
                              'dosers_depth': 2,
                              'dosers_lr': 0.00026396192072937485,
                              'dosers_wd': 7.165810318386074e-07,
                              'dosers_width': 64,
                              'penalty_adversary': 8.735507132389051,
                              'reg_adversary': 69.6011204833175,
                              'step_size_lr': 25}

    args = {'dataset_path': fname, # full path to the anndata dataset 
            'cell_type_key': 'cell_type', # necessary field for cell types. Fill it with a dummy variable if no celltypes present.
            'split_key': 'split' + str(split_num), # necessary field for train, test, ood splits.
            'perturbation_key': 'condition', # necessary field for perturbations
            'dose_key': 'dose_val', # necessary field for dose. Fill in with dummy variable if dose is the same. 
            'checkpoint_freq': 20, # checkoint frequencty to save intermediate results
            'hparams': "", #autoencoder_params, # autoencoder architecture
            'max_epochs': 200, # maximum epochs for training
            'max_minutes': 300, # maximum computation time
            'patience': 20, # patience for early stopping
            'loss_ae': 'gauss', # loss (currently only gaussian loss is supported)
            'doser_type': 'sigm', # non-linearity for doser function
            'save_dir': './', # directory to save the model
            'decoder_activation': 'linear', # last layer of the decoder
            'seed': 0, # random seed
            'sweep_seeds': 0,
            'emb': emb_model,
            'dataset': dataset_name}
    args['cuda'] = 0
    
    dataset = args["dataset"]
    
    import pandas as pd

    if dataset == 'tian2019_neuron_hvg':
        go_path = '/dfs/user/kexinh/gears2/data/go_essential_tian2020_neuron.csv'
    elif dataset == 'tian2019_ipsc_hvg':
        go_path = '/dfs/user/kexinh/gears2/data/go_essential_tian2020_ipsc.csv'
    elif dataset == 'jost2020_hvg':
        go_path = '/dfs/user/kexinh/gears2/data/go_essential_jost2020.csv'
    else:
        go_path = '/dfs/user/kexinh/gears2/go_essential_all.csv'
    
    df = pd.read_csv(go_path)
    df = df.groupby('target').apply(lambda x: x.nlargest(20 + 1,['importance'])).reset_index(drop = True)
    
    if dataset == 'tian2019_neuron_hvg':
        gene_path = '/dfs/user/kexinh/gears2/data/essential_all_data_pert_genes_tian2019_neuron.pkl'
    elif dataset == 'tian2019_ipsc_hvg':
        gene_path = '/dfs/user/kexinh/gears2/data/essential_all_data_pert_genes_tian2019_ipsc.pkl'
    elif dataset == 'jost2020_hvg':
        gene_path = '/dfs/user/kexinh/gears2/data/essential_all_data_pert_genes_jost2020.pkl'
    else:
        gene_path = '/dfs/user/kexinh/gears2/data/essential_all_data_pert_genes.pkl'
    
    
    def get_map(pert):
        tmp = np.zeros(len(gene_list))
        tmp[np.where(np.in1d(gene_list, df[df.target == pert].source.values))[0]] = 1
        return tmp    
    
    import pickle
    with open(gene_path, 'rb') as f:
        gene_list = pickle.load(f)
    args['num_pert_in_graph'] = len(gene_list)
    
    
    # load the dataset and model pre-trained weights
    autoencoder, datasets = prepare_compert(args, state_dict=state)
    print(3)
    autoencoder.load_state_dict(state)
    print(4)
    
    pert_dict = datasets["training"].perts_dict
    
    pert2neighbor =  {i: get_map(i) for i in list(pert_dict.keys())}    
    pert_dict_rev = {'+'.join([str(x) for x in np.where(j == 1)[0]]): i for i,j  in pert_dict.items()}
    autoencoder.pert2neighbor = pert2neighbor
    autoencoder.pert_dict_rev = pert_dict_rev
    
    
    from inference import evaluate, compute_metrics, deeper_analysis, GI_subgroup, non_dropout_analysis, non_zero_analysis
    compert_api = ComPertAPI(datasets, autoencoder)
    print(5)
    adata = sc.read(fname)
    print(6)
    condition2num_of_cells = dict(adata.obs.cov_drug_dose_name.value_counts())
    name_map = dict(adata.obs[['cov_drug_dose_name', 'condition']].drop_duplicates().values)

    dataset = datasets['ood']

    compert_api.model.eval()
    print(7)
    scores = pd.DataFrame(columns=[compert_api.covars_key,
                                    compert_api.perturbation_key,
                                    compert_api.dose_key,
                                    'R2_mean', 'R2_mean_DE', 'R2_var',
                                    'R2_var_DE', 'num_cells'])

    total_cells = len(dataset)

    icond = 0

    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    truth_de = []
    results = {}
    print(8)
    for pert_category in tqdm(np.unique(dataset.pert_categories)):
        # pert_category category contains: 'celltype_perturbation_dose' info
        de_idx = np.where(
            dataset.var_names.isin(
                np.array(dataset.de_genes[pert_category])))[0]

        idx = np.where(dataset.pert_categories == pert_category)[0]

        genes_control = datasets['training_control'].genes[np.random.randint(0,
                                            len(datasets['training_control'].genes), condition2num_of_cells[pert_category]), :]
        num, dim = genes_control.size(0), genes_control.size(1)

        pert_cat.extend([name_map[pert_category]] * condition2num_of_cells[pert_category])

        if len(idx) > 0:
            emb_drugs = dataset.drugs[idx][0].view(
                1, -1).repeat(num, 1).clone()
            emb_cts = dataset.cell_types[idx][0].view(
                1, -1).repeat(num, 1).clone()

            genes_predict = compert_api.model.predict(
                genes_control, emb_drugs, emb_cts).detach().cpu()

            y_pred = genes_predict[:, :dim]
            y_true = dataset.genes[idx, :].numpy()

            y_true_de = y_true[:, de_idx]
            y_pred_de = y_pred[:, de_idx]
            pred.extend(y_pred)
            truth.extend(y_true)

            pred_de.extend(y_pred_de)
            truth_de.extend(y_true_de)

    results['pert_cat'] = np.array(pert_cat)

    pred = torch.stack(pred)
    truth = np.stack(truth)
    results['pred']= pred
    results['truth']= truth

    results['pred_de']= torch.stack(pred_de)
    results['truth_de']= np.stack(truth_de)

    test_metrics, test_pert_res = compute_metrics(results)

    test_res = results
    results['pred'] = results['pred'].numpy()
    results['pred_de'] = results['pred_de'].numpy()
    args = {}
    args['wandb'] = True

    metrics = ['mse', 'mae', 'spearman', 'pearson', 'r2']
    for m in metrics:
        wandb.log({'test_' + m: test_metrics[m],
                   'test_de_'+m: test_metrics[m + '_de']                     
                  })
    print(10)
    out = deeper_analysis(adata, test_res, de_column_prefix = 'top_non_dropout_de_20')
    out_non_dropout = non_dropout_analysis(adata, test_res)
    out_non_zero = non_zero_analysis(adata, test_res)
    GI_out = GI_subgroup(out)
    GI_out_non_dropout = GI_subgroup(out_non_dropout)
    GI_out_non_zero = GI_subgroup(out_non_zero)

    metrics = ['frac_in_range', 'frac_in_range_45_55', 'frac_in_range_40_60', 'frac_in_range_25_75', 'mean_sigma', 'std_sigma', 'frac_sigma_below_1', 'frac_sigma_below_2', 'pearson_delta',
               'pearson_delta_de', 'fold_change_gap_all', 'pearson_delta_top200_hvg', 'fold_change_gap_upreg_3', 
               'fold_change_gap_downreg_0.33', 'fold_change_gap_downreg_0.1', 'fold_change_gap_upreg_10', 
               'pearson_top200_hvg', 'pearson_top200_de', 'pearson_top20_de', 'pearson_delta_top200_de', 
               'pearson_top100_de', 'pearson_delta_top100_de', 'pearson_delta_top50_de', 'pearson_top50_de', 'pearson_delta_top20_de',
               'mse_top200_hvg', 'mse_top100_de', 'mse_top200_de', 'mse_top50_de', 'mse_top20_de', 'frac_correct_direction_all', 'frac_correct_direction_20', 'frac_correct_direction_50', 'frac_correct_direction_100', 'frac_correct_direction_200', 'frac_correct_direction_20_nonzero']

    metrics_non_dropout = ['frac_correct_direction_top20_non_dropout', 'frac_opposite_direction_top20_non_dropout', 'frac_0/1_direction_top20_non_dropout', 'frac_correct_direction_non_zero', 'frac_correct_direction_non_dropout', 'frac_in_range_non_dropout', 'frac_in_range_45_55_non_dropout', 'frac_in_range_40_60_non_dropout', 'frac_in_range_25_75_non_dropout', 'mean_sigma_non_dropout', 'std_sigma_non_dropout', 'frac_sigma_below_1_non_dropout', 'frac_sigma_below_2_non_dropout', 'pearson_delta_top20_de_non_dropout', 'pearson_top20_de_non_dropout', 'mse_top20_de_non_dropout', 'frac_opposite_direction_non_dropout', 'frac_0/1_direction_non_dropout', 'frac_opposite_direction_non_zero', 'frac_0/1_direction_non_zero']

    metrics_non_zero = ['frac_correct_direction_top20_non_zero', 'frac_opposite_direction_top20_non_zero', 'frac_0/1_direction_top20_non_zero', 'frac_in_range_non_zero', 'frac_in_range_45_55_non_zero', 'frac_in_range_40_60_non_zero', 'frac_in_range_25_75_non_zero', 'mean_sigma_non_zero', 'std_sigma_non_zero', 'frac_sigma_below_1_non_zero', 'frac_sigma_below_2_non_zero', 'pearson_delta_top20_de_non_zero', 'pearson_top20_de_non_zero', 'mse_top20_de_non_zero']

    if args['wandb']:
        for m in metrics:
            wandb.log({'test_' + m: np.mean([j[m] for i,j in out.items() if m in j])})

        for m in metrics_non_dropout:
            wandb.log({'test_' + m: np.mean([j[m] for i,j in out_non_dropout.items() if m in j])})        

        for m in metrics_non_zero:
            wandb.log({'test_' + m: np.mean([j[m] for i,j in out_non_zero.items() if m in j])})        
    
    
    if dataset_name == 'Norman2019':    
        subgroup_path = '/dfs/user/kexinh/perturb_GNN/pertnet/splits/' + dataset_name + '_simulation_' + str(split_num) + '_0.1_subgroup.pkl'
    else:
        subgroup_path = '/dfs/project/perturb-gnn/datasets/data/' + dataset_name +'/splits/' + dataset_name + '_simulation_' + str(split_num) + '_0.75_subgroup.pkl'
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
            if args['wandb']:
                wandb.log({'test_' + name + '_' + m: subgroup_analysis[name][m]})

            print('test_' + name + '_' + m + ': ' + str(subgroup_analysis[name][m]))

    ## deeper analysis
    subgroup_analysis = {}
    for name in subgroup['test_subgroup'].keys():
        subgroup_analysis[name] = {}
        for m in metrics:
            subgroup_analysis[name][m] = []

        for m in metrics_non_dropout:
            subgroup_analysis[name][m] = []

        for m in metrics_non_zero:
            subgroup_analysis[name][m] = []

    for name, pert_list in subgroup['test_subgroup'].items():
        for pert in pert_list:
            for m, res in out[pert].items():
                subgroup_analysis[name][m].append(res)

            for m, res in out_non_dropout[pert].items():
                subgroup_analysis[name][m].append(res)

            for m, res in out_non_zero[pert].items():
                subgroup_analysis[name][m].append(res)


    for name, result in subgroup_analysis.items():
        for m in result.keys():
            subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
            if args['wandb']:
                wandb.log({'test_' + name + '_' + m: subgroup_analysis[name][m]})

            print('test_' + name + '_' + m + ': ' + str(subgroup_analysis[name][m]))
