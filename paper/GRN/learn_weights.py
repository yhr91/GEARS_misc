import sys
import pandas as pd
import numpy as np
import scanpy as sc
import networkx as nx
sys.path.append('../model/')
from flow import get_graph
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import glob
import argparse
import pdb

no_model_count = 0

def nonzero_idx(mat):
    mat=pd.DataFrame(mat)
    return mat[(mat > 0).sum(1) > 0].index.values

def get_split_adata(adata, split_dir, split_id):
    split_files = [f for f in glob.glob(split_dir + '/*') if 'subgroup' not in f]
    split_fname = [f for f in split_files if 'simulation_'+str(split_id) in f][0]
    split_dict = pd.read_pickle(split_fname)
    
    return adata[adata.obs['condition'].isin(split_dict['train'])]

def data_split(X, y, size=0.1):
    nnz = list(set(nonzero_idx(X)).intersection(set(nonzero_idx(y))))

    if len(nnz) <= 2:
        global no_model_count
        no_model_count += 1

        return -1,-1

    train_split, val_split = train_test_split(nnz, test_size=size,
                                              random_state=42)
    return train_split, val_split

def train_regressor(X, y, kind, alpha=0):

    if kind == 'linear':
        model = linear_model.LinearRegression()
    elif kind == 'lasso':
        model = linear_model.Lasso(alpha=alpha)
    elif kind == 'elasticnet':
        model = linear_model.ElasticNet(alpha=alpha, l1_ratio=0.5,
                                        max_iter=1000)
    elif kind == 'ridge':
        model = linear_model.Ridge(alpha=alpha, max_iter=1000)
    elif kind == 'MLP':
        model = MLPRegressor(hidden_layer_sizes=(20,10), max_iter=1000)

    reg = model.fit(X, y)
    loss = np.sqrt(np.mean((y - model.predict(X))**2))
    return reg, loss, reg.score(X, y)


def evaluate_regressor(model, X, y):
    y_cap = model.predict(X)
    loss = np.sqrt(np.mean((y - y_cap)**2))

    return loss, y, y_cap

def init_dict():
    d = {}
    d['linear'] = []
    d['lasso'] = []
    d['ridge'] = []
    d['MLP'] = []
    return d

# Looks at the median of max expression across cells/not genes
def max_median_norm(df):
    return df/df.max().median()

def get_weights(adj_mat, exp_adata, nodelist, method='linear', lim=50000):
    models = init_dict()
    adj_list = {}

    adj_list['TF'] = []; adj_list['target'] = []; adj_list['importance'] = [];

    adj_mat_idx = np.arange(len(adj_mat))
    np.random.shuffle(adj_mat_idx)
    count = 0


    def trainer(kind, feats, y):
        model, _, _ = train_regressor(
                                        feats, y, kind=kind)

        # Store results
        try: models[kind].append(model.coef_);
        except: pass;


    def trainer_split(kind, feats, y, train_split, val_split):
        models_ = []
        val_losses_ = []
        
        for alpha in [1e-6, 1e-4, 1e-2, 1e-1]:
             model, _, _ = train_regressor(
                                        feats[train_split,:],
                                        y[train_split], kind=kind,
                                        alpha=alpha)
             val_loss, _, _ = evaluate_regressor(model,
                                       feats[val_split, :],
                                       y[val_split])

             models_.append(model)
             val_losses_.append(val_loss)
        
        best_model = models_[np.argmin(val_losses_)]

        # Store results
        try: models[kind].append(best_model.coef_);
        except: pass;


    print('T genes: ', str(len(adj_mat_idx)))
    for itr in adj_mat_idx:
        i = adj_mat[itr]
        if i.sum() > 0:
            idx = np.where(i > 0)[1]
            TFs = np.array(nodelist)[idx]
            target = np.array(nodelist)[itr]

            feats = exp_adata[:, TFs].X.toarray()
            y = exp_adata[:, target].X.toarray()
            
            if method=='linear': 
                trainer('linear', feats, y)
            else:
                train_split, val_split = data_split(feats, y, size=0.1)
                if train_split==-1: continue;
                trainer_split(method, feats, y, train_split, val_split)            

            # Add row to new weight matrix
            for j,k in enumerate(TFs):
                adj_list['TF'].append(k)
                adj_list['target'].append(target)
                try:
                    adj_list['importance'].append(models[method][-1][0][j])
                except:
                    adj_list['importance'].append(models[method][-1][j])

            print(count)
            count += 1

        if count >= lim:
            break
    return models, adj_list


def main(args):
    try:
       split_id = int(args.graph_name.split('_')[-2])
    except:
       split_id = int(args.graph_name.split('_')[-3])

    adata = sc.read_h5ad(args.split_dir.split('splits')[0]+'perturb_processed.h5ad')
    
    # Remove genes with duplicaet names
    genes_to_keep = adata.var.drop_duplicates('gene_name').index
    adata = adata[:, genes_to_keep]

    exp_adata = get_split_adata(adata, args.split_dir, split_id)
    exp_adata.var = exp_adata.var.set_index('gene_name', drop=False)    

    G = pd.read_csv(args.graph_name, header=None)
    G = nx.from_pandas_edgelist(G, source=0,
                        target=1, create_using=nx.DiGraph())
    adj_mat = nx.linalg.graphmatrix.adjacency_matrix(G).todense().T
    nodelist = [n for n in G.nodes()]

    # Remove self-edges
    np.fill_diagonal(adj_mat, 0)
    models, adj_list = get_weights(adj_mat, exp_adata, 
                                    nodelist, method=args.method, lim=20000)

    out_name = args.graph_name.split('/')[-1].split('.')[0]
    pd.DataFrame(adj_list).to_csv(args.out_dir + out_name + 
                                  '_' + args.method + '_learntweights.csv')

    # Convert coefficients into new weight matrix
    print('Done')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Set model hyperparametrs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #torch.cuda.set_device(4)

    parser.add_argument('--split_dir', type=str,
                        help='Directory for splits')
    parser.add_argument('--graph_name', type=str,
                        help='Graph filename')
    parser.add_argument('--out_dir', type=str,
                        help='Output filename')
    parser.add_argument('--method', type=str,
                        help='Regression method')


    parser.set_defaults(
    split_dir ='../data/norman2019/splits',
    graph_name='norman2019_1_top50.csv',
    method='linear',
    out_dir='./')

    args = parser.parse_args()
    main(args)
