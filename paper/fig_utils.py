import glob
import numpy as np
import torch
import scanpy as sc
import pandas as pd 
import copy
import sys
sys.path.append('../../')

import os
#from data import PertDataloader
#from inference import evaluate, compute_metrics
from gears_misc.gears.inference import GIs
import matplotlib.patches as mpatches

# Linear model fitting functions
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from dcor import distance_correlation, partial_distance_correlation
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
GI_names = [k.lower() for k in GIs.keys()]
home_dir = './'
device = 'cuda:0'


# Plot generation functions
def get_t_p_seen1(metric, dict_names, return_names=False, uncert_filter=True):
    res_seen1 = {}
    res_p = {}
    res_t = {}
    res_p_names = {}
    res_t_names = {}
    seen1_dict_names = [d for d in dict_names if '+' not in d.split('/')[-1]]

    # Set up output dictionaries
    for GI in GI_names:
        res_p[GI] = []
        res_t[GI] = []

    for GI_sel in GI_names:
        res_p_names[GI_sel] = []
        res_t_names[GI_sel] = []

        # For a given GI what are all the relevant perturbations
        all_perts_gi = GIs[GI_sel.upper()]

        # What are all the relevant single gene perturbations
        all_perts_gi = [v.split('+') for v in all_perts_gi] 
        seen1_perts_gi = np.unique([item for sublist in all_perts_gi for item in sublist])

        # Iterate over all models trained with these single genes held out
        for gene in np.sort(seen1_perts_gi):
            for fpath in seen1_dict_names:
                if gene in fpath:
                    
                    res_seen1[gene] = np.load(fpath, allow_pickle=True).item()

                    # Get all keys for single pert model predictions that are relevant
                    if uncert_filter==True:
                        allowed_keys = get_low_unc_perts(gene)
                    else:
                        allowed_keys = res_seen1[gene].keys()
                    
                    keys_ = [k for k in allowed_keys if k in GIs[GI_sel.upper()]]
                    res_t_names[GI_sel].extend(keys_)
                    res_p_names[GI_sel].extend(keys_)

                    try:
                        t_vals = [res_seen1[gene][k]['truth'][metric] for k in keys_]
                        p_vals = [res_seen1[gene][k]['pred'][metric] for k in keys_]

                        res_p[GI_sel].extend(p_vals)
                        res_t[GI_sel].extend(t_vals)
                    except:
                        pass
                    
                  
                    
    res_p['synergy'] = res_p['synergy_similar_pheno'] + res_p['synergy_dissimilar_pheno'] +\
                       res_p['potentiation']
    res_t['synergy'] = res_t['synergy_similar_pheno'] + res_t['synergy_dissimilar_pheno'] +\
                       res_t['potentiation']
    res_p_names['synergy'] = res_p_names['synergy_similar_pheno'] +\
                             res_p_names['synergy_dissimilar_pheno'] +\
                             res_p_names['potentiation']
    res_t_names['synergy'] = res_t_names['synergy_similar_pheno'] +\
                             res_t_names['synergy_dissimilar_pheno'] +\
                             res_t_names['potentiation']
    
    if return_names:
        return res_p, res_t, res_p_names, res_t_names
    
    else:
        return res_p, res_t


def get_t_p_seen2_naive(metric, dict_name, return_names=False):

    # Seen 2
    res_p = {}
    res_t = {}
    res_p_names = {}
    res_t_names = {}
    loaded = np.load(dict_name, allow_pickle=True).item()

    for GI in GI_names:
        res_p[GI] = []
        res_t[GI] = []
        res_p_names[GI] = []
        res_t_names[GI] = []

    for GI in GI_names:
        GI_interactions = GIs[GI.upper()]
        for GI_interaction in GI_interactions:
            if GI_interaction in loaded.keys():
                res_t[GI].append(loaded[GI_interaction]['truth'][metric])
                res_p[GI].append(loaded[GI_interaction]['pred'][metric])
                res_t_names[GI].append(GI_interaction)
                res_p_names[GI].append(GI_interaction)
                
    res_p['synergy'] = res_p['synergy_similar_pheno'] + res_p['synergy_dissimilar_pheno'] +\
                       res_p['potentiation']
    res_t['synergy'] = res_t['synergy_similar_pheno'] + res_t['synergy_dissimilar_pheno'] +\
                       res_t['potentiation']
    res_p_names['synergy'] = res_p_names['synergy_similar_pheno'] +\
                             res_p_names['synergy_dissimilar_pheno'] +\
                             res_p_names['potentiation']
    res_t_names['synergy'] = res_t_names['synergy_similar_pheno'] +\
                             res_t_names['synergy_dissimilar_pheno'] +\
                             res_t_names['potentiation']

    if return_names:
        return res_p, res_t, res_p_names, res_t_names
    
    else:
        return res_p, res_t

    
def get_t_p_seen2_gears(metric, all_res_path, return_names=False):

    # Seen 2
    res_p = {}
    res_t = {}
    res_p_names = {}
    res_t_names = {}

    for GI in GI_names:
        res_p[GI] = []
        res_t[GI] = []
        res_p_names[GI] = []
        res_t_names[GI] = []

    all_res = np.load(all_res_path, allow_pickle=True).item()

    for GI in GI_names:
        for pert in GIs[GI.upper()]:
            
            try:
                loaded = all_res[pert]
                res_t[GI].append(loaded['true'][metric])
                res_p[GI].append(loaded['pred'][metric])
                res_t_names[GI].append(pert)
                res_p_names[GI].append(pert)

            except:
                pass

    res_p['synergy'] = res_p['synergy_similar_pheno'] + res_p['synergy_dissimilar_pheno'] +\
                       res_p['potentiation']
    res_t['synergy'] = res_t['synergy_similar_pheno'] + res_t['synergy_dissimilar_pheno'] +\
                       res_t['potentiation']
    res_p_names['synergy'] = res_p_names['synergy_similar_pheno'] +\
                             res_p_names['synergy_dissimilar_pheno'] +\
                             res_p_names['potentiation']
    res_t_names['synergy'] = res_t_names['synergy_similar_pheno'] +\
                             res_t_names['synergy_dissimilar_pheno'] +\
                             res_t_names['potentiation']

    if return_names:
        return res_p, res_t, res_p_names, res_t_names

    else:
        return res_p, res_t
    
    
# Plot generation functions
def get_t_p_seen1_gears(metric, all_res_path, return_names=False, uncert_filter=True):
    res_seen1 = {}
    res_p = {}
    res_t = {}
    res_p_names = {}
    res_t_names = {}
    
    all_res = np.load(all_res_path, allow_pickle=True).item()

    # Set up output dictionaries
    for GI_sel in GI_names:
        res_p[GI_sel] = []
        res_t[GI_sel] = []
        res_p_names[GI_sel] = []
        res_t_names[GI_sel] = []

        # For a given GI what are all the relevant perturbations
        all_perts_gi = GIs[GI_sel.upper()]

        # What are all the relevant single gene perturbations
        all_perts_gi = [v.split('+') for v in all_perts_gi] 
        seen1_perts_gi = np.unique([item for sublist in all_perts_gi for item in sublist])

        # Iterate over all models trained with these single genes held out
        
        # Single gene involved in known GIs
        for unseen_gene in np.sort(seen1_perts_gi):
            
            # Single gene models trained using LOO on that gene
            for loo_gene in all_res:
                
                # If there's a match then record seen1 results
                if unseen_gene == loo_gene:

                    # Get all keys for single pert model predictions that are relevant
                    if uncert_filter==True:
                        allowed_combos = get_low_unc_perts(unseen_gene)
                    else:
                        allowed_combos = res_seen1[unseen_gene].keys()
                        
                    
                    
                    # Pick the combos that show the desired GI
                    combos_ = [k for k in allowed_combos if k in GIs[GI_sel.upper()]]
                    
                    res_t_names[GI_sel].extend(combos_)
                    res_p_names[GI_sel].extend(combos_)

                    try:
                    #breakpoint()
                        t_vals = [all_res[unseen_gene]['true'][k][metric] for k in combos_]
                        p_vals = [all_res[unseen_gene]['pred'][k][metric] for k in combos_]

                        res_p[GI_sel].extend(p_vals)
                        res_t[GI_sel].extend(t_vals)
                    except:
                        pass
                    
    res_p['synergy'] = res_p['synergy_similar_pheno'] + res_p['synergy_dissimilar_pheno'] +\
                       res_p['potentiation']
    res_t['synergy'] = res_t['synergy_similar_pheno'] + res_t['synergy_dissimilar_pheno'] +\
                       res_t['potentiation']
    res_p_names['synergy'] = res_p_names['synergy_similar_pheno'] +\
                             res_p_names['synergy_dissimilar_pheno'] +\
                             res_p_names['potentiation']
    res_t_names['synergy'] = res_t_names['synergy_similar_pheno'] +\
                             res_t_names['synergy_dissimilar_pheno'] +\
                             res_t_names['potentiation']
    
    if return_names:
        return res_p, res_t, res_p_names, res_t_names
    
    else:
        return res_p, res_t

def get_t_p_seen2(metric, dict_names, return_names=False):

    # Seen 2
    res_p = {}
    res_t = {}
    res_p_names = {}
    res_t_names = {}
    seen2_dict_names = [d for d in dict_names if '+' in d.split('/')[-1]]

    for GI in GI_names:
        res_p[GI] = []
        res_t[GI] = []
        res_p_names[GI] = []
        res_t_names[GI] = []

    for GI in GI_names:
        for d in seen2_dict_names:
            if GI in d:
                loaded = list(np.load(d, allow_pickle=True).item().values())[0]

                try:
                    res_t[GI].append(loaded['truth'][metric])
                    res_p[GI].append(loaded['pred'][metric])
                    d = d.split('_')[-1].split('.npy')[0]
                    res_t_names[GI].append(d)
                    res_p_names[GI].append(d)
                    
                except:
                    pass

    res_p['synergy'] = res_p['synergy_similar_pheno'] + res_p['synergy_dissimilar_pheno'] +\
                       res_p['potentiation']
    res_t['synergy'] = res_t['synergy_similar_pheno'] + res_t['synergy_dissimilar_pheno'] +\
                       res_t['potentiation']
    res_p_names['synergy'] = res_p_names['synergy_similar_pheno'] +\
                             res_p_names['synergy_dissimilar_pheno'] +\
                             res_p_names['potentiation']
    res_t_names['synergy'] = res_t_names['synergy_similar_pheno'] +\
                             res_t_names['synergy_dissimilar_pheno'] +\
                             res_t_names['potentiation']
    
    if return_names:
        return res_p, res_t, res_p_names, res_t_names
    
    else:
        return res_p, res_t

def set_up_box_scatter(GI_sel, res_p, res_t, metric, GI_sel_first=False):

    xs = []
    preds = []
    trues = []
    labels = []

    GI_names = [k.lower() for k in GIs.keys()]
    param = metric
    GI_interest = GI_sel.lower()

    if GI_sel_first:
        GI_names = [g for g in GI_names if g != GI_interest]
        GI_names = [GI_interest] + GI_names

    idx = 0
    for GI_name in res_p:

        preds.append(res_p[GI_name])
        trues.append(res_t[GI_name])
        
        xs.append(np.random.normal(idx, 0.02, max(len(trues[-1]), len(preds[-1])))) 
        labels.append(GI_name)
            
        idx += 1
        
    return xs, preds, trues, labels

def remove_nan(arr):
    return [x for x in arr if x==x]


# 0.3868
def get_low_unc_perts(gene, thresh_factor=0.5):
    
    logvar_df_file = './GI_gene_mse/dfs/norman_umi_go_' + gene + '_logvar'
    logvar_df = pd.read_csv(logvar_df_file)
    logvar_df = logvar_df.set_index('condition')
    thresh  = logvar_df.loc['thresh_mean_train'][0] + thresh_factor*logvar_df.loc['thresh_std_train'][0] 
    #return logvar_df.index.values
    return logvar_df[logvar_df['0']<thresh].index.values

def get_z_scores_logvar(logvar_df):
    mean = logvar_df.loc['thresh_mean_train'][0]
    std = logvar_df.loc['thresh_std_train'][0]
    pert_inds = [ind for ind in logvar_df.index 
                 if ind not in ['thresh_mean_train', 'thresh_std_train']]

    z_score_df = logvar_df.loc[pert_inds,:].apply(lambda x: (x-mean)/std)
    z_score_df = z_score_df.rename(columns={'0':'z-score'})
    return z_score_df

def make_box_scatter(xs, preds, trues, labels, n_preds=1, title=None, xticks=True,
                     ylabel=None, scatter=True, GIs=None, seen=2,
                    pred_color=None, shade_color=None, ticklabels=None, 
                     skip_last_scatter=False, legend=True, naives=False, two_metrics=False):
    
    num = 2
    plt.figure(figsize=[1.3,5])
    medianprops = dict(linewidth=0)
    
    ### TODO fix the nan
    GI_idxs = []   
   
    t = trues[0]
    p = preds[0]


    if labels[0] in GIs:
        GI_idxs.append(0)

    """
    box = plt.boxplot(remove_nan(p), 
                      positions= [3*idx],
                      labels=[''], 
                      medianprops=medianprops, 
                      showfliers=False,
                      widths=0.8, 
                      meanline=False, 
                      patch_artist=True)
    plt.setp(box['boxes'], facecolor=pred_color, alpha=0.4, 
             linewidth=0.5)   

    box = plt.boxplot(remove_nan(t), 
                      positions= [3*idx-1],
                      labels=[ticklabels[idx]], 
                      medianprops=medianprops, 
                      showfliers=False,
                      widths=0.8, 
                      meanline=False, 
                      patch_artist=True)
    plt.setp(box['boxes'], facecolor='gray', alpha=0.4, 
             linewidth=0.5)  
    """
    if scatter:
        plt.scatter(3*xs[0]-0.25, p, marker='o', color=pred_color, alpha=0.7)
        plt.scatter(3*xs[0]-1.25, t, marker='o', color='black', alpha=0.7)
        

        if two_metrics == True:
            t2 = trues[2]
            p2 = preds[2]

            plt.scatter(3*xs[2]+0.25, p2, marker='x', color=pred_color, alpha=0.7)
            plt.scatter(3*xs[2]-0.75, t2, marker='x', color='black', alpha=0.7)            
            
    if seen ==2:
        red_patch = mpatches.Patch(color=pred_color, label='Predicted (2 genes seen)')
    elif seen==1:
        red_patch = mpatches.Patch(color=pred_color, label='Predicted (1 gene seen)')
    blue_patch = mpatches.Patch(color='gray', label='Truth')

    if legend:
        plt.legend(handles=[red_patch, blue_patch])
    
    ax = plt.gca()
    #ax.set_title(title, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.xaxis.set_ticklabels([])
    ax.xaxis.grid(True, linestyle='--')
    
    for idx in GI_idxs:
        ax.axvspan(3*idx-2, 3*idx+1, alpha=0.15, color=shade_color)    
        
    plt.xticks(rotation=90, fontsize=16)
    
    if naives == True:
        all_preds = remove_nan(preds[0]+preds[1])
        all_trues = remove_nan(trues[0]+trues[1])
        
        pred_mean = np.mean(all_preds)
        true_mean = np.mean(all_trues)
        pred_max = np.max(all_preds)
        pred_min = np.min(all_preds)
        true_max = np.max(all_trues)
        true_min = np.min(all_trues)
        
        plt.plot([idx-0.5,idx-1.5],[true_mean,true_mean], color='black', linewidth=4.0)
        plt.plot([idx-0.5,idx-1.5],[true_max,true_max], linestyle='--', color='black', linewidth=2.0)
        plt.plot([idx-0.5,idx-1.5],[true_min,true_min], linestyle='--',  color='black', linewidth=2.0)
        
        plt.plot([idx+0.5,idx-.5],[pred_mean,pred_mean], color='black', linewidth=4.0)
        plt.plot([idx+0.5,idx-.5],[pred_max,pred_max], linestyle='--',  color='black', linewidth=2.0)
        plt.plot([idx+0.5,idx-.5],[pred_min,pred_min], linestyle='--', color='black', linewidth=2.0)
        
        #plt.plot([-2,3*7+1],[naives[metric],naives[metric]], '--', color='black')
    #if not xticks:
    #    ax.set_xticklabels([])
    plt.yticks(fontsize=16)
    plt.xlim([-2,3*(num-2)+1])
    #plt.xlim([-2,3*(num-1)+1])
    #plt.ylim([0, 2.4])

    
def make_comp_box_scatter(xs, preds, trues, labels, n_preds=1, title=None, xticks=True,
                     ylabel=None, scatter=True, GIs=None, seen=2, GI=None,
                    pred_color=None, shade_color=None, ticklabels=None, 
                     skip_last_scatter=False, legend=True, extremes=None, means=None):
    
    num = len(preds)
    if len(preds)==1:
        plt.figure(figsize=[int(num*(12/6)),5])
    else:
        plt.figure(figsize=[int(num*(12/8)),5])
    medianprops = dict(linewidth=0)
    step = 2
    
    ### TODO fix the nan
    GI_idxs = []   
    for idx in range(len(preds)):
        t = trues[idx]
        p = preds[idx]
    
        if labels[idx] in GIs:
            GI_idxs.append(idx)
        
        #box = plt.boxplot(remove_nan(p), positions= [step*idx],
        #              labels=[ticklabels[idx+1]], medianprops=medianprops, 
        #              widths=0.8, meanline=False, patch_artist=True)
        #plt.setp(box['boxes'], facecolor=pred_color, alpha=0.4, 
        #         linewidth=0.5)   
        
        if scatter:
            x_ = xs[idx][:len(p[0])]
            plt.scatter((step*idx)+x_, p, color=pred_color, alpha=1)
        
        if idx == 0:
            #box = plt.boxplot(remove_nan(t), positions= [step*idx-2],
            #              labels=[ticklabels[idx]], medianprops=medianprops, 
            #              widths=0.8, meanline=False, patch_artist=True)
            #plt.setp(box['boxes'], facecolor='gray', alpha=0.4, 
            #         linewidth=0.5)    
            if scatter:
                x_ = xs[idx][:len(t[0])]
                plt.scatter(x_-2, t, color='gray', alpha=1)
                
            loc = -1
            plt.plot([loc-0.5,loc-1.5],[means[0],means[0]], color='black', linewidth=4.0)
            plt.plot([loc-0.5,loc-1.5],[extremes[0][0],extremes[0][0]], linestyle='--', color='black', linewidth=2.0)
            plt.plot([loc-0.5,loc-1.5],[extremes[0][1],extremes[0][1]], linestyle='--',  color='black', linewidth=2.0)
                
        loc = (idx*step)+1
        plt.plot([loc-0.5,loc-1.5],[means[idx+1],means[idx+1]], color='black', linewidth=4.0)
        plt.plot([loc-0.5,loc-1.5],[extremes[idx+1][0],extremes[idx+1][0]], linestyle='--', color='black', linewidth=2.0)
        plt.plot([loc-0.5,loc-1.5],[extremes[idx+1][1],extremes[idx+1][1]], linestyle='--',  color='black', linewidth=2.0)
            
    if seen ==2:
        red_patch = mpatches.Patch(color=pred_color, label='Predicted (2 genes seen)')
    elif seen==1:
        red_patch = mpatches.Patch(color=pred_color, label='Predicted (1 gene seen)')
    blue_patch = mpatches.Patch(color='gray', label='Truth')

    if legend:
        plt.legend(handles=[red_patch, blue_patch])
    
    ax = plt.gca()
    ax.set_title(title, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.xaxis.grid(True, linestyle='--')
    
    #for idx in GI_idxs:
    #    ax.axvspan(step*idx-2, step*idx+1, alpha=0.15, color=shade_color)    
        
    plt.xticks(rotation=90, fontsize=16)
    
    if GI=='additive':
        plt.plot([-3,step*7+1],[thresh[GI][0],thresh[GI][0]], '--', color=pred_color)
        plt.plot([-3,step*7+1],[thresh[GI][1],thresh[GI][1]], '--', color=pred_color)
    elif thresh[GI] is not None:
        plt.plot([-3,step*7+1],[thresh[GI],thresh[GI]], '--', color=pred_color)       
        
    if not xticks:
        if len(preds)==1:
            ax.set_xticks([-2,0])
        else:
            ax.set_xticks([-2,0,2,4])
        ax.set_xticklabels(ticklabels)
    plt.yticks(fontsize=16)
    plt.xlim([-3,step*(num-1)+1])
    #plt.ylim([0, 2.0])
    
def get_eq_contr(res):
    return np.min([res['corr_first'], res['corr_second']])/\
    np.max([res['corr_first'], res['corr_second']])

def get_truth(pert):
    # Get truth metrics for CPA predictions
    path_ = [f for f in 
             glob.glob('/dfs/user/yhr/snap/perturb_GNN/pertnet-cli/GI_subtypes_out_v2/*crossgene*.npy') if pert in f]
    
    loaded_ = np.load(path_[0], allow_pickle=True).item()
    truth = list(loaded_.values())[0]['truth']
    return truth



# Some useful dictionaries for automatic plotting
# metric names in latex
latex_names = {'dcor':r'($corr([a,b],ab)$)',
               'dcor_singles':r'$corr(a,b)$',   
               'corr_fit':r'corr($c_1a + c_2b,ab$)',
               'mag':r'$\sqrt{c_1^2 + c_2^2}$',
               'dominance':r'$|\log_{10}(c_1/c_2)|$',
               'eq_contr':r'$\frac{\min(dcor(a,ab), dcor(b,ab))}{\max(dcor(a,ab), dcor(b,ab)))}$'}

title_map = {'REDUNDANT': 'Redundance',
             'NEOMORPHIC': 'Model Fit',
             'ADDITIVE': 'Additivity', 
             'EPISTASIS': 'Dominance',
             'POTENTIATION': 'Potentiation',
             'SYNERGY_SIMILAR_PHENO': 'Synergy (Similar Phenotypes)',
             'SYNERGY_DISSIMILAR_PHENO': 'Synergy (Dissimilar Phenotypes)',
             'SUPPRESSOR': 'Suppression',
             'SYNERGY': 'Synergy'}

title_map2 = {'dcor': 'Redundancy',
             'corr_fit': 'Neomorphism',
             'dominance': 'Epistasis',
             'mag': 'Synergy',
             'eq_contr': 'Epistasis',}

tick_labels_map = {'neomorphic':'Neomorphic',
 'additive':'Additive',
 'epistasis':'Epistatic',
 'redundant':'Redundant',
 'potentiation':'Potentiating',
 'synergy':'Synergistic',
 'synergy_similar_pheno':'Synergistic\n (Similar Phenotype)',
 'synergy_dissimilar_pheno':'Synergistic\n (Dissimilar Phenotype)',
 'suppressor':'Suppressing'
}

metric_map = {'REDUNDANT': 'dcor',
             'NEOMORPHIC': 'corr_fit',
             'ADDITIVE': 'mag', 
             'EPISTASIS': 'dominance',
             'POTENTIATION': 'mag',
             'SYNERGY': 'mag',
             'SYNERGY_SIMILAR_PHENO': 'mag',
             'SYNERGY_DISSIMILAR_PHENO': 'mag',
             'SUPPRESSOR': 'mag'}

yranges = {'dcor': [0.3,1],
          'dcor_singles':[0.35,1],   
               'corr_fit':[0.35,1],
               'mag':[0.25,2.25],
               'dominance':[0,2]} 

metric_GI_map = {'dcor':['redundant'],
                 'corr_fit':['neomorphic'],
                 'dominance':['epistasis', 'potentiation'],
                 'eq_contr':['epistasis', 'potentiation'],
                 'mag':['potentiation', 'synergy', 'synergy_similar_pheno', 'synergy_dissimilar_pheno',
                       'suppressor', 'additive']    
}
    
colors={'dcor':['forestgreen', 'palegreen'],
       'corr_fit':['royalblue', 'skyblue'],
       'dominance':['darkviolet', 'plum'],
       'eq_contr':['darkviolet', 'plum'],
       'mag':['crimson', 'mistyrose']}

thresh={'neomorphic':0.88,
 'additive':[0.75,1.25],
 'epistasis':0.28,
 'redundant':0.85,
 'potentiation':1.15,
 'synergy_similar_pheno':1.15,
 'synergy_dissimilar_pheno':1.15,
 'suppressor':1.0,
 'synergy':1.15}


def get_idx_in_bounds(arr, upper, lower):
    idx_u = np.where(arr<upper)[0]
    idx_l = np.where(arr>lower)[0]
    return list(set(idx_u).intersection(set(idx_l)))

def get_prec_recall(preds, trues, upper, lower):
    result = {}
    
    preds_ = np.hstack([list(l) for l in preds.values()])
    trues_ = np.hstack([list(l) for l in trues.values()])
    
    preds_idx = get_idx_in_bounds(preds_, upper, lower)
    trues_idx = get_idx_in_bounds(trues_, upper, lower)
    TP = len(set(preds_idx).intersection(set(trues_idx)))
    
    if len(preds_idx)==0:
        result['precision'] = 0
    else:
        result['precision'] = TP/len(preds_idx)
        
    result['recall'] = TP/len(trues_idx)
    
    return result

def get_topk_acc(preds, trues, p_names, t_names, min_max, k=5, random=False):
    result = {}

    #preds = np.hstack([list(l) for l in p.values()])
    #trues = np.hstack([list(l) for l in t.values()])
    #preds_names = np.hstack([list(l) for l in p_names.values()])
    #trues_names = np.hstack([list(l) for l in t_names.values()])
    
    preds = [item for sublist in list(preds.values()) for item in sublist]
    trues = [item for sublist in list(trues.values()) for item in sublist]
    preds_names = [item for sublist in list(p_names.values()) for item in sublist]
    trues_names = [item for sublist in list(t_names.values()) for item in sublist]


    df = pd.DataFrame([preds, trues, preds_names, trues_names]).T
    df.columns = ['pred', 'true', 'pred_name', 'true_name']
    df = df.drop_duplicates('pred_name')
    if random:
        df = df.sample(len(df))
    else:
        df = df.sample(len(df), random_state=0)

    if random:
        model_P = set(df['pred_name'][-k:].values)
    if min_max == 'max':
        if not random:
            model_P = set(df.sort_values('pred')['pred_name'][-k:].values)
        true_P = set(df.sort_values('true')['true_name'][-k:].values)
    elif min_max == 'min':
        if not random:
            model_P = set(df.sort_values('pred')['pred_name'][:k].values)
        true_P = set(df.sort_values('true')['true_name'][:k].values)
    
    TP = model_P.intersection(true_P)
    return len(TP)/k
        
def get_prec_atk(preds, trues, p_names, t_names, GI_type, k=10, random=False):
    result = {}

    #preds = np.hstack([list(l) for l in p.values()])
    #trues = np.hstack([list(l) for l in t.values()])
    #preds_names = np.hstack([list(l) for l in p_names.values()])
    #trues_names = np.hstack([list(l) for l in t_names.values()])
    
    preds = [item for sublist in list(preds.values()) for item in sublist]
    trues = [item for sublist in list(trues.values()) for item in sublist]
    preds_names = [item for sublist in list(p_names.values()) for item in sublist]
    trues_names = [item for sublist in list(t_names.values()) for item in sublist]

    df = pd.DataFrame([preds, trues, preds_names, trues_names]).T
    df.columns = ['pred', 'true', 'pred_name', 'true_name']
    df = df.drop_duplicates('pred_name')
    if random:
        df = df.sample(len(df))
    else:
        df = df.sample(len(df), random_state=0)

    if random:
        model_P = set(df['pred_name'][-k:].values)
    if GI_type == 'synergy':
        if not random:
            model_P = set(df.sort_values('pred')['pred_name'][-k:].values)
        true_P = set(df[df['true']>1.15]['true_name'].values)
    elif GI_type == 'suppressor':
        if not random:
            model_P = set(df.sort_values('pred')['pred_name'][:k].values)
        true_P = set(df[df['true']<1.0]['true_name'].values)    
    elif GI_type == 'redundant':
        if not random:
            model_P = set(df.sort_values('pred')['pred_name'][-k:].values)
        true_P = set(df[df['true']>0.85]['true_name'].values)    
    elif GI_type == 'neomorphic':
        if not random:
            model_P = set(df.sort_values('pred')['pred_name'][:k].values)
        true_P = set(df[df['true']<0.88]['true_name'].values)   
    elif GI_type == 'epistasis':
        if not random:
            model_P = set(df.sort_values('pred')['pred_name'][:k].values)  ## eq_contr
            #model_P = set(df.sort_values('pred')['pred_name'][-k:].values)  ## dominance
        true_P = set(df[df['true']<0.28]['true_name'].values) ## eq_contr
        #true_P = set(df[df['true']>0.35]['true_name'].values)  ## dominance 
    elif GI_type == 'additive':
        df['additive_pred'] = (df['pred']-1).abs()
        df['additive_true'] = (df['true']-1).abs()
        if not random:
            model_P = set(df.sort_values('additive_pred')['pred_name'][:k].values)
        true_P = set(df[df['additive_true']<0.25]['true_name'].values)   
    
    TP = model_P.intersection(true_P)
    return len(TP)/k, df
    #return len(TP)/k, model_P, true_P, TP

def get_all_pr_scores(dict_names, seen=2):
    
    prec_recalls = {}
    if seen == 2:
        res_p_seen, res_t_seen = get_t_p_seen2('mag', dict_names)
    elif seen == 1:
        res_p_seen, res_t_seen = get_t_p_seen1('mag', dict_names)
    prec_recalls['suppressor'] = get_prec_recall(res_p_seen, res_t_seen, 1, 0)
    prec_recalls['additive'] = get_prec_recall(res_p_seen, res_t_seen, 1.25, 0.75)
    prec_recalls['synergy'] = get_prec_recall(res_p_seen, res_t_seen, 100, 1.15)
    
    if seen == 2:
        res_p_seen, res_t_seen = get_t_p_seen2('dcor', dict_names)
    elif seen == 1:
        res_p_seen, res_t_seen = get_t_p_seen1('dcor', dict_names)
    prec_recalls['redundant'] = get_prec_recall(res_p_seen, res_t_seen, 100, 0.85)
    
    if seen == 2:
        res_p_seen, res_t_seen = get_t_p_seen2('corr_fit', dict_names)
    elif seen == 1:
        res_p_seen, res_t_seen = get_t_p_seen1('corr_fit', dict_names)
    prec_recalls['neomorphic'] = get_prec_recall(res_p_seen, res_t_seen, 0.88, 0.0)
    
    if seen == 2:
        res_p_seen, res_t_seen = get_t_p_seen2('eq_contr', dict_names)
    elif seen == 1:
        res_p_seen, res_t_seen = get_t_p_seen1('eq_contr', dict_names)
    prec_recalls['epistatic'] = get_prec_recall(res_p_seen, res_t_seen, 0.28, 0)
    
    """
    if seen == 2:
        res_p_seen, res_t_seen = get_t_p_seen2('dominance', dict_names)
    elif seen == 1:
        res_p_seen, res_t_seen = get_t_p_seen1('dominance', dict_names)
    prec_recalls['epistatic'] = get_prec_recall(res_p_seen, res_t_seen, 5, 0.3)
    """
    
    return prec_recalls

"""
def get_all_pr_at_k(dict_names, seen=2):
    ## NOT USED
    pr_at_k = {}
    if seen == 2:
        res_p_seen, res_t_seen = get_t_p_seen2('mag', dict_names)
    elif seen == 1:
        res_p_seen, res_t_seen = get_t_p_seen1('mag', dict_names)
    prec_recalls['suppressor'] = get_prec_recall(res_p_seen, res_t_seen, 1, 0)
    prec_recalls['additive'] = get_prec_recall(res_p_seen, res_t_seen, 1.25, 0.75)
    prec_recalls['synergy'] = get_prec_recall(res_p_seen, res_t_seen, 100, 1)
    
    if seen == 2:
        res_p_seen, res_t_seen = get_t_p_seen2('dcor', dict_names)
    elif seen == 1:
        res_p_seen, res_t_seen = get_t_p_seen1('dcor', dict_names)
    prec_recalls['redundant'] = get_prec_recall(res_p_seen, res_t_seen, 1.2, 0.8)
    
    if seen == 2:
        res_p_seen, res_t_seen = get_t_p_seen2('corr_fit', dict_names)
    elif seen == 1:
        res_p_seen, res_t_seen = get_t_p_seen1('corr_fit', dict_names)
    prec_recalls['neomorphic'] = get_prec_recall(res_p_seen, res_t_seen, 0.9, 0.0)
    
    if seen == 2:
        res_p_seen, res_t_seen = get_t_p_seen2('eq_contr', dict_names)
    elif seen == 1:
        res_p_seen, res_t_seen = get_t_p_seen1('eq_contr', dict_names)
    prec_recalls['epistatic'] = get_prec_recall(res_p_seen, res_t_seen, 0.4, 0)
    
    return prec_recalls
"""


def model_setup(model_name, home_dir=home_dir, device=device):
    """
    Set up trained model, weights, dataloader
    """
    args = np.load(home_dir+'saved_args/'+model_name+'.npy', allow_pickle = True).item()
    # args['device'] = device
    device = args['device']

    if args['dataset'] == 'Norman2019':
        data_path = '/dfs/project/perturb-gnn/datasets/Norman2019/Norman2019_hvg+perts_more_de.h5ad'
    elif args['dataset'] == 'Adamson2016':
        data_path = '/dfs/project/perturb-gnn/datasets/Adamson2016_hvg+perts_more_de_in_genes.h5ad'
    elif args['dataset'] == 'Dixit2016':
        data_path = '/dfs/project/perturb-gnn/datasets/Dixit2016_hvg+perts_more_de.h5ad'
    elif args['dataset'] == 'Norman2019_Adamson2016':
        data_path = '/dfs/project/perturb-gnn/datasets/trans_norman_adamson/norman2019.h5ad'
    elif args['dataset'] == 'Norman2019_umi':
        data_path = '/dfs/project/perturb-gnn/datasets/Norman2019/Norman2019_hi_umi+hvg.h5ad'

    adata = sc.read_h5ad(data_path)
    if 'gene_symbols' not in adata.var.columns.values:
        adata.var['gene_symbols'] = adata.var['gene_name']
    gene_list = [f for f in adata.var.gene_symbols.values]

    # Pertrubation dataloader
    print(args['device'])
    pertdl = PertDataloader(adata, args)

    model = torch.load(home_dir+'saved_models/' + model_name)
    model.args = args 
    
    if 'G_sim' in vars(model):
        if isinstance(model.G_sim, dict):
            for i,j in model.G_sim.items():
                model.G_sim[i] = j.to(model.args['device'])

            for i,j in model.G_sim_weight.items():
                model.G_sim_weight[i] = j.to(model.args['device'])
        else:
            model.G_sim = model.G_sim.to(model.args['device'])
            model.G_sim_weight = model.G_sim_weight.to(model.args['device'])

    return model, pertdl, args, adata

def run_inference(pertdl, model, args):
    
    print('Train Evaluation')
    train_res = evaluate(pertdl.loaders['train_loader'],model, args)
    
    print('Val Evaluation')
    val_res = evaluate(pertdl.loaders['val_loader'],model, args)
    
    print('Test Evaluation')
    test_res = evaluate(pertdl.loaders['test_loader'],model, args)
    
    return train_res, val_res, test_res

# Get control baseline

# Set up control means
def control_setup(adata):
    adata.var = adata.var.set_index('gene_name', drop=False)
    mean_ctrl_exp = adata[adata.obs['condition'] == 'ctrl'].to_df().mean()
    return mean_ctrl_exp

#gene_name_dict = adata.var.loc[:,'gene_name'].to_dict()

# Get DE means
def get_de_exp(pert, mean_ctrl_exp):
    de_genes = get_covar_genes(pert, gene_name_dict=gene_name_dict)
    return mean_ctrl_exp[de_genes]

def get_single_name(g, adata):
    name = g+'+ctrl'
    if name in adata.obs.condition.values:
        return name
    else:
        return 'ctrl+'+g
    
def get_test_set_results_seen2(res, sel_GI_type):
    # Get relevant test set results
    test_pert_cats = [p for p in np.unique(res['pert_cat']) if p in GIs[sel_GI_type] or 'ctrl' in p]
    pred_idx = np.where([t in test_pert_cats for t in res['pert_cat']])
    out = {}
    for key in res:
        out[key] = res[key][pred_idx]
    
    return out


def get_test_set_results_seen1(test_res, model_name):
    # Get relevant test set results
    all_perts = [item for sublist in GIs.values() for item in sublist]
    test_pert_cats = [p for p in set(test_res['pert_cat']) if p in all_perts or 'ctrl' in p]
    pred_idx = np.where([t in test_pert_cats for t in test_res['pert_cat']])
    for key in test_res:
        test_res[key] = test_res[key][pred_idx]
    
    return test_res

def combine_train_val(train_res, val_res):
    for key in train_res:
        train_res[key] = np.concatenate([train_res[key], val_res[key]])   
    return train_res


def get_ctrl_pert(x, all_conds):
    # Sometime ctrl is first, sometimes its second, so this finds the right match
    return [c for c in all_conds if x in c and 'ctrl' in c][0]

def get_DE_genes(x):
    return adata.uns['rank_genes_groups_cov']['A549_'+x+'_1+1']

def get_combo_adata_true(combo, adata, DE_genes=None):
    """
    Given a combo perturbation, returns expression profiles for component 
    single perturbations and combo perturbation. Restriced to union of DE genes
    """
    
    x,y = combo.split('+')
    x_pert = get_ctrl_pert(x)
    y_pert = get_ctrl_pert(y)
    x_y_pert = gi
    
    pert_adata = adata[adata.obs['condition'].isin([x_pert, y_pert, x_y_pert])]
    
    if DE_genes is None:
        DE_genes = set(get_DE_genes(x_pert))
        DE_genes = DE_genes.union(set(get_DE_genes(y_pert)))
        DE_genes = list(DE_genes.union(set(get_DE_genes(x_y_pert))))
        pert_adata = pert_adata[:,DE_genes]
        pert_adata.var = pert_adata.var.set_index('gene_name')
    else:
        pert_adata.var = pert_adata.var.set_index('gene_name')
        pert_adata = pert_adata[:,DE_genes]

    return pert_adata

def get_pert_heatmap(pert_adata, control_adata):
    control = control_adata[:,pert_adata.var.index]
    
    pert_df = pert_adata.to_df()
    pert_df['condition'] = pert_adata.obs.condition
    pert_df = pert_df.groupby('condition').mean()
    
    pert_df = pert_df - control.to_df().mean(0)
    return pert_df

def create_output_df(res, split='truth'):
    out = pd.DataFrame(res[split])
    out.columns = adata.var.gene_name
    out['condition'] = res['pert_cat']
    return out

def get_combo_adata_pred(combo, out_df, DE_genes, control_adata):
    """
    Given a combo perturbation, returns expression profiles for component 
    single perturbations and combo perturbation. Restriced to union of DE genes
    """
    control = control_adata[:,pert_adata.var.index]
    
    x,y = combo.split('+')
    x_pert = get_ctrl_pert(x)
    y_pert = get_ctrl_pert(y)
    x_y_pert = combo
    
    pert_df = out_df.loc[[x_pert,y_pert,x_y_pert]]
    pert_df = pert_df - control.to_df().mean(0) #- out_df.loc['ctrl']
    return pert_df.loc[:,DE_genes]

def get_percentiles_gene(out_df, gene):
    out_df = out_df.loc[:,[gene, 'condition']]
    perc_hi = out_df.groupby('condition').apply(lambda x:np.percentile(x, 75))
    perc_50 = out_df.groupby('condition').apply(lambda x:np.percentile(x, 50))
    perc_low = out_df.groupby('condition').apply(lambda x:np.percentile(x, 25))
    
    return (perc_low, perc_50, perc_hi)

def check_outlier_gene(percentiles, x_pert, y_pert, x_y_pert):
    """
    Checks if the combination expression for a given gene is outside the expected IQR
    """
    
    perc_low, perc_50, perc_hi = percentiles
    if perc_low[x_pert] > perc_50[x_y_pert] or perc_hi[x_pert] < perc_50[x_y_pert]:
        return True
    
    if perc_low[y_pert] > perc_50[x_y_pert] or perc_hi[y_pert] < perc_50[x_y_pert]:
        return True
    
    return False

def get_outlier_genes_for_combo(combo, data_df):

    outlier_genes = []
    print(combo)
    for it, g in enumerate(adata.var.gene_name):

        print(it,g)
        x,y = combo.split('+')
        x_pert = get_ctrl_pert(x)
        y_pert = get_ctrl_pert(y)
        x_y_pert = combo

        percentiles = get_percentiles_gene(data_df, g)
        if check_outlier_gene(percentiles, x_pert, y_pert, x_y_pert):
            outlier_genes.append(g)
            
    return outlier_genes

def get_k_high_diff(k, vecs):
    k=10
    k_high_diff_genes = np.argsort((vecs['double_pred'] - vecs['double_truth'])**2)[-k:]
    return k_high_diff_genes

def get_all_exp_values(idx):
    values = {}
    for key in all_vectors:
        values[key] = all_vectors[key][idx]
    return values



## Identify most unexpect expressed genes

def get_combo_single_df(gi, split='truth'):
    """
    Returns a df with combination and single perturbations
    """
    
    x,y = gi.split('+')
    x_pert = get_ctrl_pert(x)
    y_pert = get_ctrl_pert(y)
    perts = [gi, x_pert, y_pert]
    
    ## Create df for combo+singles
    train_df = create_output_df(train_res, split)
    train_df = train_df[train_df['condition'].isin(perts)]
    
    test_df = create_output_df(test_res, split)
    test_df = test_df[test_df['condition'].isin(perts)]
    
    truth_df = pd.concat([train_df, test_df])
    
    return truth_df


def mean_subtract(data_df, mean_control):
    return data_df - mean_control

def add_naive_combo(data_df, gi):
    """
    Returns set of genes that show unexpected expression
    """
    x,y = gi.split('+')
    x_pert = get_ctrl_pert(x)
    y_pert = get_ctrl_pert(y)
    
    naive_sum = data_df.loc[x_pert,:] + data_df.loc[y_pert,:]
    naive_sum.name = 'Naive'

    data_df = data_df.append(naive_sum) 
    return data_df

def get_unexpect_genes(data_df, gi, k=25):
    
    data_df = add_naive_combo(data_df, gi)
    abs_diff = (data_df.loc[gi]-data_df.loc['Naive']).apply(np.abs)
    unexp_genes = abs_diff.sort_values(ascending=False).index
    
    return data_df, unexp_genes[:k]

def get_unexpect_mse(gi, mean_control, return_plot_df = False, k=20):
    results = {}
    
    truth_df = get_combo_single_df(gi, split='truth')
    truth_df = truth_df.groupby('condition').mean()
    truth_df = mean_subtract(truth_df, mean_control)
    truth_df, unexp_genes = get_unexpect_genes(truth_df, gi, k=k)
    truth_df = truth_df.loc[:,unexp_genes]


    pred_df = get_combo_single_df(gi, split='pred')
    pred_df = pred_df.groupby('condition').mean()
    pred_df = mean_subtract(pred_df, mean_control)
    pred_df = pred_df.loc[:,unexp_genes]
    pred_df.index = pred_df.index+'_p'
    
    results['pred_mse'] = np.mean((pred_df.loc[gi+'_p',:] - truth_df.loc[gi,:])**2)
    results['pred_pearson'] = pearsonr(pred_df.loc[gi+'_p',:],truth_df.loc[gi,:])
    results['naive_mse'] = np.mean((truth_df.loc['Naive',:] - truth_df.loc[gi,:])**2)
    results['naive_pearson'] = pearsonr(truth_df.loc['Naive',:],truth_df.loc[gi,:])
    
    if return_plot_df:
        plot_df = truth_df.append(pred_df.loc[gi+'_p'])
        return results, plot_df
    
    else:
        return results

def get_unexpect_precision(gi, mean_control, k=10, k_real=None):
    results = {}
    
    truth_df = get_combo_single_df(gi, split='truth')
    truth_df = truth_df.groupby('condition').mean()
    truth_df = mean_subtract(truth_df, mean_control)

    pred_df_ = get_combo_single_df(gi, split='pred')
    pred_df_ = pred_df_.groupby('condition').mean()
    pred_df_ = mean_subtract(pred_df_, mean_control)
    
    # Replace single perturbation in pred_df with true values   
    pred_df = truth_df.copy()
    pred_df.loc[gi,:] = pred_df_.loc[gi, :]
    
    if k_real is None:
        k_real = k
    truth_naive_df, unexp_genes_truth = get_unexpect_genes(truth_df, gi, k=k_real)
    
    pred_naive_df, unexp_genes_pred = get_unexpect_genes(pred_df, gi, k=k)
    
    naive_topk = ((truth_naive_df.loc['Naive',:])**2).sort_values(ascending=False)[:k].keys()

    results['naive_precision_at_'+str(k)+'_'+str(k_real)] = len(set(unexp_genes_truth).intersection(set(naive_topk)))/k
    results['precision_at_'+str(k)+'_'+str(k_real)] = len(set(unexp_genes_truth).intersection(set(unexp_genes_pred)))/k
    
    return results