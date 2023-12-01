import argparse
import sys
#sys.path.append('../')

from gears import PertData, GEARS
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--device', type=int, default=3)
parser.add_argument('--dataset', type=str, default='norman', choices = ['norman', 'adamson', 'dixit', 
                         'replogle_k562_essential', 
                         'replogle_rpe1_essential'])
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

if args.dataset in ['norman', 'adamson', 'dixit']:
    pert_data = PertData('./data', default_pert_graph=False)
else:
    pert_data = PertData('./data', default_pert_graph=True)

pert_data.load(data_name = args.dataset)
pert_data.prepare_split(split = 'simulation', seed = seed) # get data split with seed
pert_data.get_dataloader(batch_size = args.batch_size, test_batch_size = args.batch_size)

gears_model = GEARS(pert_data, device = 'cuda:' + str(args.device),  
                    weight_bias_track = True, 
                    proj_name = args.dataset, 
                    exp_name = str(args.model) + '_seed' + str(seed))

gears_model.model_initialize(hidden_size = 64, no_perturb = no_perturb)
gears_model.train(epochs = epoch)
if args.model != 'no_perturb':
    gears_model.save_model('./model_ckpt/' + args.dataset + '_' + args.model + '_run' + str(seed))
