import sys
import pandas as pd
import numpy as np
import scanpy as sc
import networkx as nx
from flow import get_graph
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import argparse

no_model_count = 0

def nonzero_idx(mat):
    mat=pd.DataFrame(mat)
    return mat[(mat > 0).sum(1) > 0].index.values

def data_split(X, y, size=0.1):
    nnz = list(set(nonzero_idx(X)).intersection(set(nonzero_idx(y))))

    if len(nnz) <= 5:
        global no_model_count
        no_model_count += 1

        return -1,-1

    train_split, val_split = train_test_split(nnz, test_size=size,
                                              random_state=42)
    return train_split, val_split

def train_regressor(X, y, kind='linear'):

    if kind == 'linear' or kind == 'linear_TM':
        model = linear_model.LinearRegression()
    elif kind == 'lasso':
        model = linear_model.Lasso(alpha=10e4)
    elif kind == 'elasticnet' or kind == 'elasticnet_TM':
        model = linear_model.ElasticNet(alpha=10, l1_ratio=0.5,
                                        max_iter=1000)
    elif kind == 'ridge':
        model = linear_model.Ridge(alpha=10e4, max_iter=1000)
    elif kind == 'MLP' or kind == 'MLP_TM':
        model = MLPRegressor(hidden_layer_sizes=(20,10), max_iter=1000)

    reg = model.fit(X, y)
    loss = np.sqrt(np.mean((y - model.predict(X.values))**2))
    return reg, loss, reg.score(X, y)


def evaluate_regressor(model, X, y):
    y_cap = model.predict(X.values)
    loss = np.sqrt(np.mean((y - y_cap)**2))

    return loss, y.values, y_cap

def init_dict():
    d = {}
    d['linear'] = []
    d['linear_TM'] = []
    d['ones'] = []
    d['lasso'] = []
    d['elasticnet'] = []
    d['elasticnet_TM'] = []
    d['ridge'] = []
    d['MLP'] = []
    d['MLP_TM'] = []
    return d

# Looks at the median of max expression across cells/not genes
def max_median_norm(df):
    return df/df.max().median()

def set_up_TM_data(X, norm=True):
    TM_data = sc.read_h5ad('/dfs/user/yhr/CPA/datasets/TMSenis_brain_myeloid.h5ad')
    TM_data = TM_data.to_df()
    if (norm):
        TM_data = max_median_norm(TM_data)
        X = max_median_norm(X.T)
    X_TM = X.merge(TM_data, left_index=True, right_index=True,
                     suffixes=[None,"_TM"])
    X = X_TM.loc[:,[c for c in X_TM.columns if '_TM' not in c]]
    return X.T, X_TM.T

def get_weights(adj_mat, X, nodelist, lim=20000):
    models = init_dict()
    val_loss = init_dict()
    train_loss = init_dict()
    train_score = init_dict()
    preds = init_dict()
    trues = init_dict()
    adj_list = {}
    test_splits = {}

    adj_list['TF'] = []; adj_list['target'] = []; adj_list['importance'] = [];
    #X, X_TM = set_up_TM_data(X)
    #TM_rows = [c for c in X_TM.index if '_TM' in c]

    adj_mat_idx = np.arange(len(adj_mat))
    np.random.shuffle(adj_mat_idx)
    count = 0

    def trainer(kind, feats, y, train_split, val_split):
        model, train_loss_, train_score_ = train_regressor(
                                        feats.loc[train_split,:],
                                        y.loc[train_split], kind=kind)
        val_loss_, true, pred = evaluate_regressor(model,
                                       feats.loc[val_split, :],
                                       y.loc[val_split])

        # Store results
        val_loss[kind].append(val_loss_)
        train_loss[kind].append(train_loss_)
        train_score[kind].append(train_score_)
        trues[kind].extend(true)
        preds[kind].extend(pred)
        try: models[kind].append(model.coef_);
        except: pass;

    print('Total genes: ', str(len(adj_mat_idx)))
    for itr in adj_mat_idx:
        i = adj_mat[itr]
        if i.sum() > 0:
            idx = np.where(i > 0)[1]
            TFs = np.array(nodelist)[idx]
            target = np.array(nodelist)[itr]

            try:
                feats = X.loc[:, TFs]
                y = X.loc[:, target]
            except:
                continue
            train_split, test_split = data_split(feats, y, size=0.1)
            if train_split==-1: continue;

            feats = feats.loc[train_split,:]
            train_split, val_split = data_split(feats, y, size=0.1)
            if train_split==-1: continue;

            print(count)

            # Add data from TM
            #feats_TM = X_TM.loc[:, TFs]
            #y_TM = X_TM.loc[:, target]

            # Linear Regression models
            trainer('linear', feats, y, train_split, val_split)
            #trainer('linear_TM', feats_TM, y_TM, train_split+TM_rows, val_split)
            #trainer('ridge', feats, y, train_split, val_split)
            #trainer('MLP', feats, y, train_split, val_split)
            #trainer('MLP_TM', feats_TM, y_TM, train_split+TM_rows, val_split)
            #trainer('elasticnet', feats, y, train_split, val_split)
            #trainer('elasticnet_TM', feats_TM, y_TM, train_split+TM_rows,
            # val_split)

            # All edges are 1
            model = linear_model.LinearRegression()
            model.coef_ = np.ones(len(idx))
            model.intercept_ = 0
            val_loss_, true, pred = evaluate_regressor(model, feats.loc[
                                                          val_split,:],
                                                  y.loc[val_split])
            val_loss['ones'].append(val_loss_)
            trues['ones'].extend(true)
            preds['ones'].extend(pred)

            # Add row to new weight matrix
            for j,k in enumerate(TFs):
                adj_list['TF'].append(k)
                adj_list['target'].append(target)
                adj_list['importance'].append(models['linear'][-1][j])

            # Save the test split for use later
            test_splits[target] = test_split

            count += 1

        if count >= lim:
            break
    return models, adj_list, test_splits


def main(args):
    X = pd.read_csv(args.exp_matrix, delimiter='\t',
                                 index_col=0)
    regs = pd.read_csv(args.regulons_path, header=2)

    G = get_graph(name = args.graph_name, TF_only=False)
    G = nx.from_pandas_edgelist(G, source=0,
                        target=1, create_using=nx.DiGraph())
    adj_mat = nx.linalg.graphmatrix.adjacency_matrix(G).todense().T
    nodelist = [n for n in G.nodes()]

    # Remove self-edges
    np.fill_diagonal(adj_mat, 0)
    X = X.T

    models, adj_list, test_splits = get_weights(adj_mat, X, nodelist, lim=1000)

    # Save final results
    #np.save('train_loss', specs[])
    np.save('test_splits', test_splits)

    if args.out_name is None:
       args.out_name = args.graph_name
    pd.DataFrame(adj_list).to_csv(args.out_name + '_learntweights.csv')

    # Convert coefficients into new weight matrix
    print('Done')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Set model hyperparametrs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #torch.cuda.set_device(4)

    parser.add_argument('--regulons_path', type=str,
                        help='Regulons data path')
    parser.add_argument('--exp_matrix', type=str,
                        help='Expression matrix')
    parser.add_argument('--graph_name', type=str,
                        help='Graph filename')
    parser.add_argument('--out_name', type=str,
                        help='Output filename')


    parser.set_defaults(
    regulons_path = '../Notebooks/regulons_Norman2019_ctrl_only.csv',
    exp_matrix = '../Notebooks/Norman2019_ctrl_only.txt',
    graph_name='Norman2019_ctrl_only',
    out_name=None)

    args = parser.parse_args()
    main(args)
