import argparse
import sys
sys.path.append('../')

from gears import PertData, GEARS
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--dataset', type=str, default='norman2019', choices = ['norman2019', 'jost2020_hvg', 'tian2021_crispri_hvg', 'tian2021_crispra_hvg', 'replogle2020_hvg', 'replogle_rpe1_gw_hvg', 'replogle_k562_gw_hvg', 'replogle_k562_essential_hvg', 'tian2019_neuron_hvg', 'tian2019_ipsc_hvg', 'replogle_rpe1_gw_filtered_hvg', 'replogle_k562_essential_filtered_hvg', 'norman'])
parser.add_argument('--model', type=str, default='gears', choices = ['gears', 'no_perturb'])
parser.add_argument('--batch_size', type=int, default=32)

args = parser.parse_args()
seed = args.seed

if args.model == 'no_perturb':
    epoch = 0
    no_perturb = True
else:
    epoch = 15
    no_perturb = False

## Set this to local dataloader directory
data_path = './data/'

if args.dataset == 'tian2019_neuron_hvg':
    gene_path = './data/essential_all_data_pert_genes_tian2019_neuron.pkl'
elif args.dataset == 'tian2019_ipsc_hvg':
    gene_path = './data/essential_all_data_pert_genes_tian2019_ipsc.pkl'
elif args.dataset == 'jost2020_hvg':
    gene_path = './data/essential_all_data_pert_genes_jost2020.pkl'
elif args.dataset == 'norman2019':
    gene_path = './data/essential_norman.pkl'
else:
    gene_path = None

if args.dataset in ['tian2019_neuron_hvg', 'tian2019_ipsc_hvg', 'jost2020_hvg']:
    add_small_graph = True
else:
    add_small_graph = False
    
pert_data = PertData(data_path[:-1], gene_path = gene_path) # specific saved folder
pert_data.load(data_path = data_path + args.dataset) # load the processed data, the path is saved folder + dataset_name
pert_data.prepare_split(split = 'simulation', seed = seed)
pert_data.get_dataloader(batch_size = args.batch_size, test_batch_size = args.batch_size)
from gears import GEARS
gears_model = GEARS(pert_data, device = 'cuda:' + str(args.device), 
                    weight_bias_track = True, 
                        proj_name = args.dataset, 
                        exp_name = str(args.model) + '_seed' + str(seed))

if args.dataset == 'tian2019_neuron_hvg':
    go_path = './data/go_essential_tian2020_neuron.csv'
elif args.dataset == 'tian2019_ipsc_hvg':
    go_path = './data/go_essential_tian2020_ipsc.csv'
elif args.dataset == 'jost2020_hvg':
    go_path = './data/go_essential_jost2020.csv'
elif args.dataset == 'norman2019':
    go_path = './data/go_essential_norman.csv'
else:
    go_path = None
    
gears_model.model_initialize(hidden_size = 64, no_perturb = no_perturb, go_path = go_path)

gears_model.train(epochs = epoch)
if args.model != 'no_perturb':
    gears_model.save_model('./model_ckpt/' + args.dataset + '_' + args.model + '_run' + str(seed))
