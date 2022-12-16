import sys
sys.path.append('../../gears_misc/')
from gears import PertData, GEARS

import numpy as np
from multiprocessing import Pool
import tqdm
import scanpy as sc

data_name = 'norman_umi'
model_name = 'gears_misc_umi_no_test_leavecluster_strict'
pert_data = PertData('/dfs/project/perturb-gnn/datasets/data')
pert_data.load(data_path = '/dfs/project/perturb-gnn/datasets/data/'+data_name)
pert_data.prepare_split(split = 'no_test', seed = 1)
pert_data.get_dataloader(batch_size = 32, test_batch_size = 128)

gears_model = GEARS(pert_data, device = 'cuda:7', 
                        weight_bias_track = False, 
                        proj_name = 'gears', 
                        exp_name = model_name)
gears_model.load_pretrained('./model_ckpt/'+model_name)


## ---- GI Predictions

def get_reverse(perts):
    return [t.split('+')[-1]+'+'+t.split('+')[0] for t in perts]

def remove_reverse(perts):
    return list(set(perts).difference(set(get_reverse(perts))))

def remove_duplicates_list(list_):
    import itertools
    list_.sort()
    return list(k for k,_ in itertools.groupby(list_))

# ---- SEEN 2

norman_adata = sc.read_h5ad('/dfs/project/perturb-gnn/datasets/data/'+data_name+'/perturb_processed.h5ad')

genes_of_interest = set([c.strip('+ctrl') for c in norman_adata.obs['condition'] 
                         if ('ctrl+' in c) or ('+ctrl' in c)])
genes_of_interest = [g for g in genes_of_interest if g in list(pert_data.pert_names)]


all_possible_combos = []

for g1 in genes_of_interest:
    for g2 in genes_of_interest:
        if g1==g2:
            continue
        all_possible_combos.append(sorted([g1,g2]))
        
all_possible_combos = remove_duplicates_list(all_possible_combos)

## First run inference on all combos using GPU 

# Predict all singles
for c in genes_of_interest:
     print('Single prediction: ',c)
     gears_model.predict([[c]])

# Predict all combos
for it, c in enumerate(all_possible_combos):
    print('Combo prediction: ',it)
    gears_model.predict([c])

# Then use a CPU-based model for computing GI scores parallely
np.save(model_name+'_all_preds', gears_model.saved_pred)
gears_model_cpu = GEARS(pert_data, device = 'cpu')
gears_model_cpu.saved_pred = gears_model.saved_pred

def Map(F, x, workers):
    """
    wrapper for map()
    Spawn workers for parallel processing
    
    """
    with Pool(workers) as pool:
        ret = list(tqdm.tqdm(pool.imap(F, x), total=len(x)))
    return ret

def mapper(c):
    return gears_model_cpu.GI_predict(c)

all_GIs = Map(mapper, all_possible_combos, workers=10)

# Construct final dictionary and save
all_GIs = {str(key):val for key, val in zip(all_possible_combos, all_GIs)}
np.save(model_name+'_allGI', all_GIs)
np.save(model_name+'_alluncs', gears_model.saved_logvar_sum)
