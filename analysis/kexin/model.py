import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

from torch_geometric.nn import GINConv, GCNConv, GATConv, GraphConv, SGConv
from torch_geometric.nn import GENConv, DeepGCNLayer

from torch.nn import Sequential, Linear, ReLU, LayerNorm, PReLU
import pandas as pd

import sys

sys.path.append('/dfs/user/yhr/cell_reprogram/model/')
from flow import get_graph, get_expression_data, \
    add_weight, I_TF, get_TFs, solve, \
    solve_parallel, get_expression_lambda


class linear_model():
    """
    Linear model object for reading in background network and assigning
    weights. Only considers directed edges from TFs
    """
    def __init__(self, species, regulon_name, gene_list, adjacency):
        self.TFs = get_TFs(species)

        # Set up graph structure
        G_df = get_graph(name=regulon_name,
                         TF_only=False)
        print('Edges: ' + str(len(G_df)))
        self.G = nx.from_pandas_edgelist(G_df, source=0,
                                         target=1, create_using=nx.DiGraph())

        # Add nodes without edges but with expression to the graph
        for n in gene_list:
            if n not in self.G.nodes():
                self.G.add_node(n)

        # Add edge weights
        self.read_weights = pd.read_csv(adjacency, index_col=0)
        self.gene_list = gene_list
        try:
            self.read_weights = self.read_weights.set_index('TF')
        except:
            pass

    def create_adj_mat(self):
        # Create a df version of the graph for merging
        G_df = pd.DataFrame(self.G.edges(), columns=['TF', 'target'])

        # Merge it with the weights DF
        weighted_G_df = self.read_weights.merge(G_df, on=['TF', 'target'])
        for w in weighted_G_df.iterrows():
            add_weight(self.G, w[1]['TF'], w[1]['target'], w[1]['importance'])

        # Get an adjacency matrix based on the gene ordering from the DE list
        return nx.linalg.graphmatrix.adjacency_matrix(
            self.G, nodelist=self.gene_list).todense()


class MLP(torch.nn.Module):
    """
    A multilayer perceptron with ReLU activations and optional BatchNorm.
    From Lotfollahi et al. 2021
    """

    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 2 else None,
                torch.nn.ReLU()
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        if self.activation == "linear":
            pass
        elif self.activation == "ReLU":
            self.relu = torch.nn.ReLU()
        else:
            raise ValueError("last_layer_act must be one of 'linear' or 'ReLU'")

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        if self.activation == "ReLU":
            x = self.network(x)
            dim = x.size(1) // 2
            return torch.cat((self.relu(x[:, :dim]), x[:, dim:]), dim=1)
        return self.network(x)


class simple_GNN(torch.nn.Module):
    """
    shallow GNN architecture with no AE
    """

    def __init__(self, num_feats, num_genes, hidden_size, node_embed_size,
                 incl_edge_weight, loss_type='micro'):
        super(simple_GNN, self).__init__()

        self.num_genes = num_genes
        self.node_embed_size = node_embed_size
        self.conv1 = GATConv(num_feats, hidden_size)
        self.conv2 = GATConv(hidden_size, hidden_size)
        self.lin = Linear(hidden_size, node_embed_size)
        self.loss_type = loss_type

        if incl_edge_weight:
            self.incl_edge_weight = True
        else:
            self.incl_edge_weight = False

    def forward(self, data, graph, weights):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, \
                                          data.edge_attr, data.batch

        if edge_index is None:
            num_graphs = len(data.batch.unique())
            edge_index = graph.repeat(1, num_graphs)

        x = self.conv1(x, edge_index=edge_index)
        #x = self.conv1(x, edge_index=edge_index, edge_weight=edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index=edge_index)
        x = x.relu()
        out = self.lin(x)

        out = torch.split(torch.flatten(out), self.num_genes *
                          self.node_embed_size)
        return torch.stack(out)




class GNN_Disentangle(torch.nn.Module):
    """
    GNN_Disentangle
    """

    def __init__(self, args, num_feats, num_genes, hidden_size, node_embed_size,
                 incl_edge_weight, ae_num_layers, ae_hidden_size,
                 single_gene_out=False, loss_type='micro', ae_decoder = False, 
                 shared_weights = False, num_layers = 2, model_backend = 'GAT'):
        super(GNN_Disentangle, self).__init__()

        self.num_genes = num_genes
        self.shared_weights = shared_weights
        self.num_layers = num_layers
        self.model_backend = model_backend
        self.gene_specific = args['gene_specific']
        self.gene_emb = args['gene_emb']
        self.lambda_emission = args['lambda_emission']
        self.sim_gnn = args['sim_gnn']
        self.sim_gnn_gene = args['sim_gnn_gene']
        self.gene_sim_pos_emb = args['gene_sim_pos_emb']
        
        if self.sim_gnn:
            self.G_sim = args['G_sim'].to(args['device'])
            self.G_sim_weight = args['G_sim_weight'].to(args['device'])
            
        if self.sim_gnn_gene:
            self.G_coexpress = args['G_coexpress'].to(args['device'])
            self.G_coexpress_weight = args['G_coexpress_weight'].to(args['device'])
            
        self.uncertainty = args['uncertainty']
        self.args = args
        
        if 'pert_emb' in args:
            self.pert_emb = args['pert_emb']
            self.pert_emb_lambda = args['pert_emb_lambda']
            self.pert_emb_agg = args['pert_emb_agg']
            if self.pert_emb_agg == 'occurence':
                self.gene_occurence = args['gene_occurence']
                self.inv_node_map = args['inv_node_map']
                
            if self.lambda_emission:
                self.occurence_bit = args['occurence_bit']
                self.inv_node_map = args['inv_node_map']
        else:
            self.pert_emb = False
            
        self.pert_emb = args['pert_emb']
        self.gene_pert_agg = args['gene_pert_agg']
        
        if 'delta_predict' in args:
            self.delta_predict = args['delta_predict']
        else:
            self.delta_predict = False
            
            
        if 'delta_predict_with_gene' in args:
            self.delta_predict_with_gene = args['delta_predict_with_gene']
        else:
            self.delta_predict_with_gene = False
        
        self.args = args
        
        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            if self.model_backend == 'GAT':
                self.layers.append(GATConv(hidden_size, hidden_size, heads = args['gat_num_heads'], dropout = args['dropout']))
            elif self.model_backend == 'GCN':
                self.layers.append(GCNConv(hidden_size, hidden_size))

            
        #self.conv1 = GATConv(hidden_size, hidden_size)
        #self.conv2 = GATConv(hidden_size, hidden_size)
        
        self.pert_w = nn.Linear(1, hidden_size)
        self.gene_basal_w = nn.Linear(1, hidden_size)
        
        if self.gene_pert_agg == 'concat+w':
            self.pert_base_trans_w = nn.Linear(hidden_size * 2, hidden_size)
        
        if self.gene_emb:
            self.emb = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
            if self.delta_predict:
                self.emb_trans = nn.Linear(hidden_size, hidden_size)
            else:
                self.emb_trans = nn.Linear(hidden_size * 2, hidden_size)
        
        if self.gene_sim_pos_emb:
            self.emb_pos = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
            self.layers_emb_pos = torch.nn.ModuleList()
            for i in range(1, num_layers + 1):
                if self.model_backend == 'GAT':
                    self.layers_emb_pos.append(GATConv(hidden_size, hidden_size, heads = args['gat_num_heads'], dropout = args['dropout']))
                elif self.model_backend == 'GCN':
                    self.layers_emb_pos.append(GCNConv(hidden_size, hidden_size))
            
            
        if self.sim_gnn:
            self.sim_layers = torch.nn.ModuleList()
            for i in range(1, 1 + 1):
                #if self.model_backend == 'GAT':
                #    self.sim_layers.append(GATConv(hidden_size, hidden_size))
                #elif self.model_backend == 'GCN':
                self.sim_layers.append(SGConv(hidden_size, hidden_size, 1))
                #self.sim_layers.append(GCNConv(hidden_size, hidden_size))
            #self.sim_trans = nn.Linear(hidden_size * 2, hidden_size)
            
        if self.sim_gnn_gene:
            self.sim_layers_gene = torch.nn.ModuleList()
            for i in range(1, 1 + 1):
                self.sim_layers.append(SGConv(hidden_size, hidden_size, 1))

        if self.pert_emb:
            #self.pert_emb_trans = nn.Linear(hidden_size, hidden_size)
            self.pert_emb_trans = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
            if self.pert_emb_agg == 'learnable':
                self.pert_lambda_pred = MLP([hidden_size, hidden_size, 1], last_layer_act='ReLU')
        
        if self.args['batchnorm']:
            self.bn = nn.BatchNorm1d(hidden_size,
                                 eps=args['bn_eps'],
                                 momentum=args['bn_mom'])
        
        if self.args['activation'] == 'relu':
            self.act = ReLU()
        elif self.args['activation'] == 'parametric-relu':
            self.act = PReLU()
            
        if self.gene_specific:
            pass
        else:
            #self.recovery_w = nn.Linear(hidden_size, 1)
            self.recovery_w = MLP([hidden_size, hidden_size*2, hidden_size, 1], last_layer_act='linear')
            
        if self.uncertainty:
            self.uncertainty_w = MLP([hidden_size, hidden_size*2, hidden_size, 1], last_layer_act='linear')
        
        if self.args['no_gnn']:
            if self.args['mlp_mixer']:
                self.transform = nn.Sequential(
                    *[
                        MixerLayer(hidden_size, self.num_genes, 2, self.args['mlp_mixer_dropout'])
                        for _ in range(self.args['mlp_mixer_num_layers'])
                    ]
                )

            else:
                self.transform = MLP([hidden_size, hidden_size], last_layer_act='ReLU')
        
        self.loss_type = loss_type
        self.ae_decoder = ae_decoder
        if self.ae_decoder:
            if single_gene_out==1:
                ae_output_size = 1
            else:
                ae_output_size = num_genes
            
            ae_input_size = node_embed_size * num_genes
            self.encoder = MLP(
                [ae_input_size] + [ae_hidden_size] * ae_num_layers +
                [ae_hidden_size])
            self.decoder = MLP(
                [ae_hidden_size] + [ae_hidden_size] * ae_num_layers +
                [ae_output_size], last_layer_act='linear')
            
    def forward(self, data, graph, weights):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, \
                                          data.edge_attr, data.batch
        
        # Check whether graph is included in batch, transform the grpah
        # object to match the PyG format
        if edge_index is None:
            num_graphs = len(data.batch.unique())
            edge_index = graph.repeat(1, num_graphs)

        pert = x[:, 1].reshape(-1,1)
        pert_emb = self.pert_w(pert)
        
        if self.delta_predict:
            if not self.gene_emb:
                raise ValueError('delta_predict mode has to turn on gene_emb!')
            emb = self.emb(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))
            base_emb = self.emb_trans(emb)
            
        else:
            gene_base = x[:, 0].reshape(-1,1)
            base_emb = self.gene_basal_w(gene_base)
        
            if self.gene_emb:
                emb = self.emb(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))
                
                if self.sim_gnn_gene:
                    emb_ori = emb
                    for idx, layer in enumerate(self.sim_layers_gene):
                        emb = layer(emb, self.G_coexpress, self.G_coexpress_weight)
                        if idx < self.num_layers - 1:
                            emb = emb.relu()
                    emb = emb_ori + 0.2 * emb
                    
                base_emb = torch.cat((emb, base_emb), axis = 1)
                base_emb = self.emb_trans(base_emb)
       
        
        if self.pert_emb:
            pert_index = torch.where(pert.reshape(*data.y.shape) == 1)
            pert_global_emb = self.pert_emb_trans(torch.LongTensor(list(range(self.num_genes))).to(self.args['device']))

            if self.sim_gnn:
                for idx, layer in enumerate(self.sim_layers):
                    pert_global_emb = layer(pert_global_emb, self.G_sim, self.G_sim_weight)
                    if idx < self.num_layers - 1:
                        pert_global_emb = pert_global_emb.relu()
                
                #pert_global_emb = self.sim_trans(torch.cat((pert_global_emb, self.pert_emb_trans(torch.LongTensor(list(range(self.num_genes))).to(self.args['device']))), axis = 1))
                
            base_emb = base_emb.reshape(num_graphs, self.num_genes, -1)
            
            if self.pert_emb_agg == 'learnable':
                pert_emb_lambda = self.pert_lambda_pred(pert_global_emb[pert_index[1]])
                
            for i, j in enumerate(pert_index[0]):
                
                if self.pert_emb_agg == 'learnable':
                    lambda_i = pert_emb_lambda[i] 
                elif self.pert_emb_agg == 'constant':
                    lambda_i = self.pert_emb_lambda 
                elif self.pert_emb_agg == 'occurence':
                    lambda_i = self.gene_occurence[self.inv_node_map[pert_index[1][i].item()]]
                if self.lambda_emission:    
                    if self.training:
                        # during training
                        emi_p = np.random.binomial(1, 0.75, 1)[0]
                        if emi_p == 0:
                            ## emit
                            lambda_i = 0
                    else:
                        # during testing
                        if self.occurence_bit[self.inv_node_map[pert_index[1][i].item()]] == 0:
                            lambda_i = 0
                base_emb[j] += lambda_i * pert_global_emb[pert_index[1][i]]
                    
            base_emb = base_emb.reshape(num_graphs * self.num_genes, -1)
        
        if self.gene_pert_agg == 'concat+w':
            def pert_base_trans(pert_emb, base_emb):
                return self.pert_base_trans_w(torch.cat((pert_emb, base_emb), axis = 1))
        elif self.gene_pert_agg == 'sum':
            def pert_base_trans(pert_emb, base_emb):
                return pert_emb+base_emb
            
        if self.args['no_pert_emb']:  
            pert_base_emb = base_emb
        else:
            pert_base_emb = pert_base_trans(pert_emb, base_emb)
        
        
        if self.args['batchnorm']:
            pert_base_emb = self.bn(pert_base_emb)
        
        
        if self.args['no_gnn']:
            pert_base_emb = pert_base_emb.reshape(num_graphs, self.num_genes, -1)
            pert_base_emb = self.transform(pert_base_emb)
            pert_base_emb = pert_base_emb.reshape(num_graphs * self.num_genes, -1)
            
        else:
            if self.args['no_disentangle']:
                for layer in self.layers:
                    pert_base_emb = layer(pert_base_emb, edge_index)
                    pert_base_emb = pert_base_emb.relu()
            else:
                x = pert_base_emb
                if self.shared_weights:
                    for i in range(self.num_layers):                        
                        pert_emb = self.layer(pert_base_emb, edge_index)
                        pert_emb = self.act(pert_emb)
                        pert_base_emb = pert_base_trans(pert_emb, base_emb)

                        if self.args['skipsum']:
                            pert_base_emb = pert_base_emb + x
                else:
                    for layer in self.layers:
                        pert_emb = layer(pert_base_emb, edge_index)
                        pert_emb = self.act(pert_emb)
                        pert_base_emb = pert_base_trans(pert_emb, base_emb)

                    if self.args['skipsum']:
                        pert_base_emb = pert_base_emb + x
                        
        if self.delta_predict:
            out = self.recovery_w(pert_base_emb) + x[:, 0].reshape(-1,1)
            out = torch.split(torch.flatten(out), self.num_genes)
        elif self.delta_predict_with_gene:
            out = self.recovery_w(pert_base_emb) + x[:, 0].reshape(-1,1)
            out = torch.split(torch.flatten(out), self.num_genes)
        else:
            out = self.recovery_w(pert_base_emb)
            out = torch.split(torch.flatten(out), self.num_genes)
        
        if self.uncertainty:
            out_logvar = self.uncertainty_w(pert_base_emb)
            out_logvar = torch.split(torch.flatten(out_logvar), self.num_genes)
            return torch.stack(out), torch.stack(out_logvar)
        
        if self.ae_decoder:
            out = torch.stack(out)
            encoded = self.encoder(out)
            decoded = self.decoder(encoded)
            return decoded.squeeze()
        else:
            return torch.stack(out)
        

        
class simple_GNN_AE(torch.nn.Module):
    """
    shallow GNN + AE
    """

    def __init__(self, num_feats, num_genes, hidden_size, node_embed_size,
                 incl_edge_weight, ae_num_layers, ae_hidden_size,
                 single_gene_out=False, loss_type='micro'):
        super(simple_GNN_AE, self).__init__()

        self.num_genes = num_genes
        self.node_embed_size = node_embed_size
        self.conv1 = GATConv(num_feats, hidden_size)
        self.conv2 = GATConv(hidden_size, hidden_size)
        self.lin = Linear(hidden_size, node_embed_size)
        self.loss_type = loss_type
        if single_gene_out==1:
            ae_output_size = 1
        else:
            ae_output_size = num_genes

        if incl_edge_weight:
            self.incl_edge_weight = True
        else:
            self.incl_edge_weight = False

        ae_input_size = node_embed_size * num_genes
        self.encoder = MLP(
            [ae_input_size] + [ae_hidden_size] * ae_num_layers +
            [ae_hidden_size])
        self.decoder = MLP(
            [ae_hidden_size] + [ae_hidden_size] * ae_num_layers +
            [ae_output_size], last_layer_act='linear')

    def forward(self, data, graph, weights):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, \
                                          data.edge_attr, data.batch

        # Check whether graph is included in batch, transform the grpah
        # object to match the PyG format
        if edge_index is None:
            num_graphs = len(data.batch.unique())
            edge_index = graph.repeat(1, num_graphs)

        x = self.conv1(x, edge_index=edge_index)
        #x = self.conv1(x, edge_index=edge_index, edge_weight=edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index=edge_index)
        x = x.relu()
        out = self.lin(x)

        out = torch.split(torch.flatten(out), self.num_genes *
                          self.node_embed_size)
        out = torch.stack(out)
        encoded = self.encoder(out)
        decoded = self.decoder(encoded)

        return decoded.squeeze()

    
        
class AE(torch.nn.Module):
    """
    AE for post training
    """

    def __init__(self, num_feats, num_genes, hidden_size, node_embed_size,
                 ae_num_layers, ae_hidden_size, loss_type='micro'):
        super(AE, self).__init__()

        self.num_genes = num_genes
        self.loss_type = loss_type
        ae_input_size = node_embed_size * num_genes
        self.encoder = MLP(
            [ae_input_size] + [ae_hidden_size] * ae_num_layers + [
                ae_hidden_size])
        self.decoder = MLP(
            [ae_hidden_size] + [ae_hidden_size] * ae_num_layers + [num_genes],
            last_layer_act='linear')

    def forward(self, data, g, w):
        
        x = data.x
        x = x[:, 0].reshape(*data.y.shape)
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded
    
class No_Perturb(torch.nn.Module):
    """
    No Perturbation
    """

    def __init__(self):
        super(No_Perturb, self).__init__()        

    def forward(self, data, g, w):
        
        x = data.x
        x = x[:, 0].reshape(*data.y.shape)
        
        return x
   
        
        
class simple_AE(torch.nn.Module):
    """
    AE for post training
    """

    def __init__(self, num_feats, num_genes, hidden_size, node_embed_size,
                 ae_num_layers, ae_hidden_size, loss_type='micro'):
        super(simple_AE, self).__init__()

        self.num_genes = num_genes
        self.loss_type = loss_type
        ae_input_size = node_embed_size * num_genes
        self.encoder = MLP(
            [ae_input_size] + [ae_hidden_size] * ae_num_layers + [
                ae_hidden_size])
        self.decoder = MLP(
            [ae_hidden_size] + [ae_hidden_size] * ae_num_layers + [num_genes],
            last_layer_act='linear')

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

    
class MLP_2(nn.Module):
    def __init__(self, num_features, expansion_factor, dropout):
        super().__init__()
        num_hidden = num_features * expansion_factor
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(num_hidden, num_features)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(F.gelu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x


class TokenMixer(nn.Module):
    def __init__(self, num_features, num_patches, expansion_factor, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
        self.mlp = MLP_2(num_patches, expansion_factor, dropout)

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features)
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)
        # x.shape == (batch_size, num_features, num_patches)
        x = self.mlp(x)
        x = x.transpose(1, 2)
        # x.shape == (batch_size, num_patches, num_features)
        out = x + residual
        return out


class ChannelMixer(nn.Module):
    def __init__(self, num_features, num_patches, expansion_factor, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
        self.mlp = MLP_2(num_features, expansion_factor, dropout)

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features)
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        # x.shape == (batch_size, num_patches, num_features)
        out = x + residual
        return out


class MixerLayer(nn.Module):
    def __init__(self, num_features, num_patches, expansion_factor, dropout):
        super().__init__()
        self.token_mixer = TokenMixer(
            num_features, num_patches, expansion_factor, dropout
        )
        self.channel_mixer = ChannelMixer(
            num_features, num_patches, expansion_factor, dropout
        )

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features)
        x = self.token_mixer(x)
        x = self.channel_mixer(x)
        # x.shape == (batch_size, num_patches, num_features)
        return x