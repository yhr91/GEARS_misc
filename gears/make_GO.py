## Script for creating Gene Ontology graph from a custom set of genes

import pickle, os
import pandas as pd

data_name = 'dixit'

with open(os.path.join('./data/', 'gene2go_all.pkl'), 'rb') as f:
    gene2go = pickle.load(f)
    
with open('./data/essential_' + data_name + '.pkl', 'rb') as f:
    essential_genes = pickle.load(f)
    
gene2go = {i: gene2go[i] for i in essential_genes if i in gene2go}

import tqdm
from multiprocessing import Pool
import numpy as np

def get_edge_list(g1):
    edge_list = []
    for g2 in gene2go.keys():
        score = len(gene2go[g1].intersection(gene2go[g2]))/len(gene2go[g1].union(gene2go[g2]))
        if score > 0.1:
            edge_list.append((g1, g2, score))
    return edge_list

with Pool(40) as p:
    all_edge_list = list(tqdm.tqdm(p.imap(get_edge_list, list(gene2go.keys())), total=len(gene2go.keys())))
    
edge_list = []
for i in all_edge_list:
    edge_list = edge_list + i

del all_edge_list

df_edge_list = pd.DataFrame(edge_list).rename(columns = {0: 'gene1', 1: 'gene2', 2: 'score'})

df_edge_list = df_edge_list.rename(columns = {'gene1': 'source', 'gene2': 'target', 'score': 'importance'})
df_edge_list.to_csv('./data/go_essential_' + data_name + '.csv', index = False)
