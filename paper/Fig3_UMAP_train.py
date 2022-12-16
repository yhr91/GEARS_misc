import sys
sys.path.append('../../gears_misc/')

from gears import PertData, GEARS

pert_data = PertData('/dfs/project/perturb-gnn/datasets/data')
pert_data.load(data_path = '/dfs/project/perturb-gnn/datasets/data/norman_umi')
pert_data.prepare_split(split = 'no_test', seed = 1)
pert_data.get_dataloader(batch_size = 32, test_batch_size = 128)


gears_model = GEARS(pert_data, device = 'cuda:6', 
                        weight_bias_track = True, 
                        proj_name = 'gears', 
                        exp_name = 'gears_misc_positional_umi_no_test')
gears_model.model_initialize(hidden_size = 64,
                               uncertainty=False)

gears_model.train(epochs = 20, lr = 1e-3)

gears_model.save_model('gears_misc_positional_umi_no_test')
