import torch
import numpy as np
import pandas as pd

## helper function
def parse_single_pert(i):
    a = i.split('+')[0]
    b = i.split('+')[1]
    if a == 'ctrl':
        pert = b
    else:
        pert = a
    return pert

def parse_combo_pert(i):
    return i.split('+')[0], i.split('+')[1]

def parse_any_pert(p):
    if ('ctrl' in p) and (p != 'ctrl'):
        return [parse_single_pert(p)]
    elif 'ctrl' not in p:
        out = parse_combo_pert(p)
        return [out[0], out[1]]

def np_pearson_cor(x, y):
    xv = x - x.mean(axis=0)
    yv = y - y.mean(axis=0)
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)
    result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))
    # bound the values to -1 to 1 in the event of precision issues
    return np.maximum(np.minimum(result, 1.0), -1.0)


def get_similarity_network(network_type, dataset, adata, pertdl, args, threshold = 0.4, k = 10):
    
    if network_type == 'co-expression_train':
        df_out = get_coexpression_network_from_train(adata, pertdl, args, threshold, k)
    elif network_type == 'gene_ontology':
        if dataset == 'Norman2019':
            df_jaccard = pd.read_csv('/dfs/user/kexinh/perturb_GNN/kexin/gene_sim_jc_filter_norman.csv')
        elif dataset == 'Adamson2016':
            df_jaccard = pd.read_csv('/dfs/user/kexinh/perturb_GNN/kexin/gene_sim_jc_filter_adamson.csv')
        df_out = df_jaccard.groupby('target').apply(lambda x: x.nlargest(k + 1,['importance'])).reset_index(drop = True)
    elif network_type == 'string_ppi':
        df_string =  pd.read_csv('/dfs/project/perturb-gnn/graphs/STRING_full_9606.csv')
        gene_list = adata.var.gene_name.values
        df_out = get_string_ppi(df_string, gene_list, k)
        
    return df_out
        
def get_string_ppi(df_string, gene_list, k):        
    df_string = df_string[df_string.source.isin(gene_list)]
    df_string = df_string[df_string.target.isin(gene_list)]
    df_string = df_string.sort_values('importance', ascending=False)
    df_string = df_string.groupby('target').apply(lambda x: x.nlargest(k + 1,['importance'])).reset_index(drop = True)
    return df_string

def get_coexpression_network_from_train(adata, pertdl, args, threshold = 0.4, k = 10):
    import os
    import pandas as pd
    
    fname = './saved_networks/' + args['dataset'] + '_' + args['split'] + '_' + str(args['seed']) + '_' + str(args['test_set_fraction']) + '_' + str(threshold) + '_' + str(k) + '_co_expression_network.csv'
    
    if os.path.exists(fname):
        return pd.read_csv(fname)
    else:
        gene_list = [f for f in adata.var.gene_symbols.values]
        idx2gene = dict(zip(range(len(gene_list)), gene_list)) 
        X = adata.X
        train_perts = pertdl.set2conditions['train']
        X_tr = X[np.isin(adata.obs.condition, [i for i in train_perts if 'ctrl' in i])]
        gene_list = adata.var['gene_name'].values

        X_tr = X_tr.toarray()
        out = np_pearson_cor(X_tr, X_tr)
        out[np.isnan(out)] = 0
        out = np.abs(out)

        out_sort_idx = np.argsort(out)[:, -(k + 1):]
        out_sort_val = np.sort(out)[:, -(k + 1):]

        df_g = []
        for i in range(out_sort_idx.shape[0]):
            target = idx2gene[i]
            for j in range(out_sort_idx.shape[1]):
                df_g.append((idx2gene[out_sort_idx[i, j]], target, out_sort_val[i, j]))

        df_g = [i for i in df_g if i[2] > threshold]
        df_co_expression = pd.DataFrame(df_g).rename(columns = {0: 'source', 1: 'target', 2: 'importance'})
        df_co_expression.to_csv(fname, index = False)
        return df_co_expression
    
def weighted_mse_loss(input, target, weight):
    """
    Weighted MSE implementation
    """
    sample_mean = torch.mean((input - target) ** 2, 1)
    return torch.mean(weight * sample_mean)


def uncertainty_loss_fct(pred, logvar, y, perts, loss_mode = 'l2', gamma = 1, reg = 0.1, reg_core = 1, loss_direction = False, ctrl = None, direction_lambda = 1e-3, filter_status = False, dict_filter = None):
    perts = np.array(perts)
    losses = torch.tensor(0.0, requires_grad=True).to(pred.device)
    for p in set(perts):
        if (p!= 'ctrl') and filter_status:
            retain_idx = dict_filter[p]
            pred_p = pred[np.where(perts==p)[0]][:, retain_idx]
            y_p = y[np.where(perts==p)[0]][:, retain_idx]
            logvar_p = logvar[np.where(perts==p)[0]][:, retain_idx]
        else:
            pred_p = pred[np.where(perts==p)[0]]
            y_p = y[np.where(perts==p)[0]]
            logvar_p = logvar[np.where(perts==p)[0]]
        
        if loss_mode == 'l2':
            losses += torch.sum(0.5 * torch.exp(-logvar_p) * (pred_p - y_p)**2 + 0.5 * logvar_p)/pred_p.shape[0]/pred_p.shape[1]
        elif loss_mode == 'l3':
            #losses += torch.sum(0.5 * torch.exp(-logvar_p) * (pred_p - y_p)**(2 + gamma) + 0.01 * logvar_p)/pred_p.shape[0]/pred_p.shape[1]
            #losses += torch.sum((pred_p - y_p)**(2 + gamma) + 0.1 * torch.exp(-logvar_p) * (pred_p - y_p)**(2 + gamma) + 0.1 * logvar_p)/pred_p.shape[0]/pred_p.shape[1]
            losses += reg_core * torch.sum((pred_p - y_p)**(2 + gamma) + reg * torch.exp(-logvar_p) * (pred_p - y_p)**(2 + gamma))/pred_p.shape[0]/pred_p.shape[1]
        if loss_direction:
            if (p!= 'ctrl') and filter_status:
                losses += torch.sum(direction_lambda * (torch.sign(y_p - ctrl[retain_idx]) - torch.sign(pred_p - ctrl[retain_idx]))**2)/pred_p.shape[0]/pred_p.shape[1]

            else:
                losses += torch.sum(direction_lambda * (torch.sign(y_p - ctrl) - torch.sign(pred_p - ctrl))**2)/pred_p.shape[0]/pred_p.shape[1]
            
    return losses/(len(set(perts)))


def loss_fct(pred, y, perts, weight=1, loss_type = 'macro', loss_mode = 'l2', gamma = 1, loss_direction = False, ctrl = None, direction_lambda = 1e-3, filter_status = False, dict_filter = None):

        # Micro average MSE
        if loss_type == 'macro':
            mse_p = torch.nn.MSELoss()
            perts = np.array(perts)
            losses = torch.tensor(0.0, requires_grad=True).to(pred.device)
            for p in set(perts):
                if (p!= 'ctrl') and filter_status:
                    retain_idx = dict_filter[p]
                    pred_p = pred[np.where(perts==p)[0]][:, retain_idx]
                    y_p = y[np.where(perts==p)[0]][:, retain_idx]
                else:
                    pred_p = pred[np.where(perts==p)[0]]
                    y_p = y[np.where(perts==p)[0]]
                    
                if loss_mode == 'l2':
                    losses += torch.sum((pred_p - y_p)**2)/pred_p.shape[0]/pred_p.shape[1]
                elif loss_mode == 'l3':
                    losses += torch.sum((pred_p - y_p)**(2 + gamma))/pred_p.shape[0]/pred_p.shape[1]

                if loss_direction:
                    if (p!= 'ctrl') and filter_status:
                        losses += torch.sum(direction_lambda * (torch.sign(y_p - ctrl[retain_idx]) - torch.sign(pred_p - ctrl[retain_idx]))**2)/pred_p.shape[0]/pred_p.shape[1]
                        
                    else:
                        losses += torch.sum(direction_lambda * (torch.sign(y_p - ctrl) - torch.sign(pred_p - ctrl))**2)/pred_p.shape[0]/pred_p.shape[1]
                    #losses += torch.sum(direction_lambda * (pred_p - y_p)**(2 + gamma) * (torch.sign(y_p - ctrl) - torch.sign(pred_p - ctrl))**2)/pred_p.shape[0]/pred_p.shape[1]
            return losses/(len(set(perts)))

        else:
            # Weigh the loss for perturbations (unweighted by default)
            #weights = np.ones(len(pred))
            #non_ctrl_idx = np.where([('ctrl' != p) for p in perts])[0]
            #weights[non_ctrl_idx] = weight
            #loss = weighted_mse_loss(pred, y, torch.Tensor(weights).to(pred.device))
            if loss_mode == 'l2':
                loss = torch.sum((pred - y)**2)/pred.shape[0]/pred.shape[1]
            elif loss_mode == 'l3':
                loss = torch.sum((pred - y)**(2 + gamma))/pred.shape[0]/pred.shape[1]

            return loss      