import wandb
from train import trainer
import argparse

config_default = dict(
    top_edge_percent = 20,
    node_hidden_size = 64,
    gnn_num_layers = 2,
    delta_predict = 'False',
    batchnorm = 'False',
    skipsum = 'False',
    gat_num_heads = 1,
    dropout = 0.0,
    activation = 'relu'    
)


wandb.init(config=config_default)
config = wandb.config

batchnorm = config.batchnorm == 'True'
delta_predict = config.delta_predict == 'True'
skipsum = config.skipsum == 'True'

# dataset arguments
parser = argparse.ArgumentParser(description='Perturbation response')

parser.add_argument('--dataset', type=str, choices = ['Norman2019'], default="Norman2019")
parser.add_argument('--split', type=str, choices = ['simulation', 'combo_seen0', 'combo_seen1', 'combo_seen2', 'single', 'single_only'], default="simulation")
parser.add_argument('--seed', type=int, default=1)    
parser.add_argument('--test_set_fraction', type=float, default=0.1)
parser.add_argument('--train_gene_set_size', type=float, default=0.75)
parser.add_argument('--combo_seen2_train_frac', type=float, default=0.75)
parser.add_argument('--test_perts', type=str, default='N/A')
parser.add_argument('--only_test_set_perts', default=False, action='store_true')

parser.add_argument('--perturbation_key', type=str, default="condition")
parser.add_argument('--species', type=str, default="human")
parser.add_argument('--binary_pert', default=True, action='store_false')
parser.add_argument('--edge_attr', default=True, action='store_false')
parser.add_argument('--ctrl_remove_train', default=False, action='store_true')
parser.add_argument('--edge_weights', action='store_true', default=False,
                    help='whether to include linear edge weights during '
                         'GNN training')
# Dataloader related
parser.add_argument('--pert_feats', default=True, action='store_false',
                    help='Separate feature to indicate perturbation')
parser.add_argument('--pert_delta', default=False, action='store_true',
                    help='Represent perturbed cells using delta gene '
                         'expression')
parser.add_argument('--edge_filter', default=False, action='store_true',
                    help='Filter edges based on applied perturbation')

# network arguments
parser.add_argument('--network_name', type=str, default = 'string')
parser.add_argument('--top_edge_percent', type=float, default=10,
                    help='percentile of top edges to retain for graph')

# training arguments
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lr_decay_step_size', type=int, default=3)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--print_progress_steps', type=int, default=50)

# model arguments
parser.add_argument('--node_hidden_size', type=int, default=2,
                    help='hidden dimension for GNN')
parser.add_argument('--node_embed_size', type=int, default=1,
                    help='final node embedding size for GNN')
parser.add_argument('--ae_hidden_size', type=int, default=512,
                    help='hidden dimension for AE')
parser.add_argument('--gnn_num_layers', type=int, default=2,
                    help='number of layers in GNN')
parser.add_argument('--ae_num_layers', type=int, default=2,
                    help='number of layers in autoencoder')

parser.add_argument('--model', choices = ['GNN_simple', 'GNN_AE', 'GNN_Disentangle', 'GNN_Disentangle_AE', 'AE', 'No_Perturb'], 
                    type = str, default = 'GNN_Disentangle', help='model name')
parser.add_argument('--model_backend', choices = ['GCN', 'GAT', 'DeepGCN'], 
                    type = str, default = 'GAT', help='model name')    
parser.add_argument('--shared_weights', default=False, action='store_true',
                help='Separate feature to indicate perturbation')                    
parser.add_argument('--gene_specific', default=False, action='store_true',
                help='Separate feature to indicate perturbation')                    
parser.add_argument('--gene_emb', default=True, action='store_false',
            help='Separate feature to indicate perturbation')   
parser.add_argument('--pert_emb', default=True, action='store_false',
            help='Separate feature to indicate perturbation')   
parser.add_argument('--gene_pert_agg', default='sum', choices = ['sum', 'concat+w'], type = str)
parser.add_argument('--delta_predict', default=config.delta_predict, action='store_true')   

parser.add_argument('--pert_emb_lambda', type=float, default=0.2)
parser.add_argument('--pert_emb_agg', type=str, default='constant', choices = ['constant', 'learnable', 'occurence'])
parser.add_argument('--lambda_emission', default=False, action='store_true')
parser.add_argument('--sim_gnn', default=True, action='store_false')
parser.add_argument('--sim_graph', default='knn_go_pathway', type = str, choices = ['knn_go_pathway', 'go_pathway'])

parser.add_argument('--gat_num_heads', type=int, default=config.gat_num_heads)
parser.add_argument('--dropout', type=float, default=config.dropout)
parser.add_argument('--batchnorm', default=config.batchnorm, action='store_true')
parser.add_argument('--bn_eps', type=float, default=1e-5)
parser.add_argument('--bn_mom', type=float, default=0.1)
parser.add_argument('--activation', type=str, default = config.activation, choices=['relu', 'parametric-relu'])
parser.add_argument('--skipsum', default=config.skipsum, action='store_true')
parser.add_argument('--uncertainty', default=False, action='store_true')

# ablation analysis

parser.add_argument('--no_pert_emb', default=False, action='store_true')
parser.add_argument('--no_disentangle', default=False, action='store_true')

# loss
parser.add_argument('--pert_loss_wt', type=int, default=1,
                    help='weights for perturbed cells compared to control cells')
parser.add_argument('--loss_type', type=str, default='macro', choices = ['macro', 'micro'],
                    help='micro averaged or not')
parser.add_argument('--loss_mode', choices = ['l2', 'l3'], type = str, default = 'l3')
parser.add_argument('--focal_gamma', type=int, default=2)    

# wandb related
parser.add_argument('--wandb', default=True, action='store_true',
                help='Use wandb or not')
parser.add_argument('--wandb_sweep', default=True, action='store_true',
            help='Use wandb or not')
parser.add_argument('--project_name', type=str, default='pert_gnn',
                    help='project name')
parser.add_argument('--entity_name', type=str, default='kexinhuang',
                    help='entity name')
parser.add_argument('--exp_name', type=str, default='N/A',
                    help='entity name')

args = dict(vars(parser.parse_args('')))

trainer(args)