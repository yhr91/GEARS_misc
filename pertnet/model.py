import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import pdb

from torch_geometric.nn import GINConv, GCNConv, GATConv, GraphConv, SGConv
from torch.nn import Sequential, Linear, ReLU, LayerNorm, PReLU
import pandas as pd

import sys

class No_Perturb(torch.nn.Module):
    """
    No Perturbation
    """

    def __init__(self):
        super(No_Perturb, self).__init__()        

    def forward(self, data):
        
        x = data.x
        x = x[:, 0].reshape(*data.y.shape)
        
        return x

class MLP(torch.nn.Module):

    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 1 else None,
                torch.nn.ReLU()
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        self.network = torch.nn.Sequential(*layers)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        if self.activation == "ReLU":
            return self.relu(x)
        else:
            return self.network(x)


class PertNet(torch.nn.Module):
    """
    PertNet
    """

    def __init__(self, args):
        super(PertNet, self).__init__()

        self.num_genes = args['num_genes']
        hidden_size = args['node_hidden_size']
        self.uncertainty = args['uncertainty']
        self.num_layers = args['num_of_gnn_layers']
        self.network_type = args['network_type']
        self.indv_out_layer = args['indv_out_layer']
        self.indv_out_hidden_size = args['indv_out_hidden_size']
        self.num_mlp_layers = args['num_mlp_layers']

        # gene structure
        self.gene_sim_pos_emb = args['gene_sim_pos_emb']
        self.num_layers_gene_pos = args['gene_sim_pos_emb_num_layers']
        self.model_backend = args['model_backend']

        self.args = args        
        # lambda for aggregation between global perturbation emb + gene embedding
        self.pert_emb_lambda = args['pert_emb_lambda']
        
        # perturbation positional embedding added only to the perturbed genes
        self.pert_w = nn.Linear(1, hidden_size)

        # gene basal embedding - encoding gene expression
        self.gene_basal_w = nn.Linear(1, hidden_size)
        
        # gene/globel perturbation embedding dictionary lookup            
        self.gene_emb = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.pert_emb = nn.Embedding(self.num_genes, hidden_size, max_norm=True)

        # transformation layer
        self.emb_trans = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        self.pert_base_trans_w = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        self.gene_base_trans_w = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        self.transform = MLP([hidden_size, hidden_size, hidden_size],
                             last_layer_act='ReLU')
        self.cross_gene_MLP = args['cross_gene_MLP']

        if self.gene_sim_pos_emb:
            self.G_coexpress = args['G_coexpress'].to(args['device'])
            self.G_coexpress_weight = args['G_coexpress_weight'].to(args['device'])

            self.emb_trans_v2 = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
            self.emb_pos = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
            self.layers_emb_pos = torch.nn.ModuleList()
            for i in range(1, self.num_layers_gene_pos + 1):
                if self.model_backend == 'GAT':
                    self.layers_emb_pos.append(GATConv(hidden_size, hidden_size, heads = 1))
                elif self.model_backend == 'GCN':
                    self.layers_emb_pos.append(GCNConv(hidden_size, hidden_size))
                elif self.model_backend == 'SGC':
                    self.layers_emb_pos.append(SGConv(hidden_size, hidden_size, 1))


        if self.indv_out_layer:
            self.recovery_w = MLP([hidden_size, hidden_size*2, hidden_size], last_layer_act='linear')
            #self.recovery_w = MLP([hidden_size, hidden_size], last_layer_act='ReLU')
            self.indv_w1 = nn.Parameter(torch.rand(self.num_genes,
                                                   hidden_size, 1))
            self.indv_b1 = nn.Parameter(torch.rand(self.num_genes, 1))
            self.act = nn.ReLU()

            if self.cross_gene_MLP:
                # Cross gene state encoder
                self.cross_gene_state = MLP([self.num_genes, hidden_size,
                                             hidden_size])

                # First layer parameters
                self.indv_w2 = nn.Parameter(torch.rand(1, self.num_genes,
                                                       hidden_size+1))
                self.indv_b2 = nn.Parameter(torch.rand(1, self.num_genes))

                # Second layer parameters
                #self.indv_w3 = nn.Parameter(torch.rand(1, 8, self.num_genes))
                #self.indv_b3 = nn.Parameter(torch.rand(1, self.num_genes))

                nn.init.xavier_normal_(self.indv_w2)
                nn.init.xavier_normal_(self.indv_b2)

            #self.indv_w2 = nn.Parameter(torch.rand(hidden_size, hidden_size,1))
            #self.indv_b2 = nn.Parameter(torch.rand(hidden_size, 1))
            
            #self.indv_w_out = nn.Parameter(torch.rand(self.num_genes, hidden_size, 1))
            #self.indv_b_out = nn.Parameter(torch.rand(self.num_genes, 1))
            
            nn.init.xavier_normal_(self.indv_w1)
            nn.init.xavier_normal_(self.indv_b1)

            nn.init.kaiming_uniform_(self.indv_w1, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.indv_w1)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.indv_b1, -bound, bound)

            #nn.init.xavier_normal_(self.indv_w_out)
            #nn.init.xavier_normal_(self.indv_b_out)
        else:
            self.recovery_w = MLP([hidden_size, hidden_size*2, hidden_size, 1], last_layer_act='linear')

        if self.args['func_readout']:
            self.func_w1 = MLP([hidden_size, hidden_size * 2, hidden_size, 1],
                last_layer_act='linear')
            self.func_w2 = MLP([self.num_genes, self.num_genes * 2, self.num_genes, 1],
                last_layer_act='linear')
        
        ### perturbation embedding similarity
        if self.network_type == 'all':
            self.sim_layers_networks = {}
            self.G_sim = {}
            self.G_sim_weight = {}
            
            for i in ['string_ppi', 'co-expression_train', 'gene_ontology']:
                self.G_sim[i] = args['G_sim_' + i].to(args['device'])
                self.G_sim_weight[i] = args['G_sim_weight_' + i].to(args['device'])
                
                sim_layers = torch.nn.ModuleList()
                for l in range(1, self.num_layers + 1):
                    sim_layers.append(SGConv(hidden_size, hidden_size, 1))
                self.sim_layers_networks[i] = sim_layers
            self.sim_layers_networks = nn.ModuleDict(self.sim_layers_networks)
            
        else:   
            # perturbation similarity network
            self.G_sim = args['G_sim'].to(args['device'])
            self.G_sim_weight = args['G_sim_weight'].to(args['device'])

            self.sim_layers = torch.nn.ModuleList()
            for i in range(1, self.num_layers + 1):
                self.sim_layers.append(SGConv(hidden_size, hidden_size, 1))

        # batchnorms
        self.bn_pert_emb = nn.BatchNorm1d(hidden_size)
        self.bn_emb = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base_trans = nn.BatchNorm1d(hidden_size)
        self.bn_final = nn.BatchNorm1d(hidden_size)
        self.bn_gene_base_trans = nn.BatchNorm1d(hidden_size)
        self.bn_gene_base = nn.BatchNorm1d(hidden_size)
        self.bn_post_gnn = nn.BatchNorm1d(hidden_size)
        self.bn_base_emb = nn.BatchNorm1d(hidden_size)

        self.cross_gene_MLP=args['cross_gene_MLP']

        # uncertainty mode
        if self.uncertainty:
            self.uncertainty_w = MLP([hidden_size, hidden_size*2, hidden_size, 1], last_layer_act='linear')
        

    def forward(self, data):
        x, batch = data.x, data.batch
        num_graphs = len(data.batch.unique())
        
        ## get base gene embeddings
        emb = self.gene_emb(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))        
        emb = self.bn_emb(emb)
        base_emb = self.emb_trans(emb)
        #base_emb = self.bn_base_emb(base_emb)

        if self.gene_sim_pos_emb:
            pos_emb = self.emb_pos(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))
            for idx, layer in enumerate(self.layers_emb_pos):
                pos_emb = layer(pos_emb, self.G_coexpress, self.G_coexpress_weight)
                if idx < len(self.layers_emb_pos) - 1:
                    pos_emb = pos_emb.relu()

            #base_emb = self.emb_trans(torch.cat((emb, pos_emb), axis = 1))

            base_emb = base_emb + 0.2 * pos_emb
            base_emb = self.emb_trans_v2(base_emb)


        ## get perturbation index and embeddings
        pert = x[:, 1].reshape(-1,1)
        pert_index = torch.where(pert.reshape(num_graphs, int(x.shape[0]/num_graphs)) == 1)
        pert_global_emb = self.pert_emb(torch.LongTensor(list(range(self.num_genes))).to(self.args['device']))
        #pert_global_emb = self.bn_pert_emb(pert_global_emb)
        
        if self.network_type == 'all':
            pert_global_emb_all = 0
            for i in ['string_ppi', 'co-expression_train', 'gene_ontology']:
                sim_layers = self.sim_layers_networks[i]
                
                for idx, layer in enumerate(sim_layers):
                    pert_global_emb = layer(pert_global_emb, self.G_sim[i], self.G_sim_weight[i])
                    if idx < self.num_layers - 1:
                        pert_global_emb = pert_global_emb.relu()
                
                pert_global_emb_all += pert_global_emb
            pert_global_emb = pert_global_emb_all               
        else:
            ## augment global perturbation embedding with GNN
            for idx, layer in enumerate(self.sim_layers):
                pert_global_emb = layer(pert_global_emb, self.G_sim, self.G_sim_weight)
                if idx < self.num_layers - 1:
                    pert_global_emb = pert_global_emb.relu()
            #pert_global_emb = self.bn_post_gnn(pert_global_emb)

        ## add global perturbation embedding to each gene in each cell in the batch
        base_emb = base_emb.reshape(num_graphs, self.num_genes, -1)


        for i, j in enumerate(pert_index[0]):
            lambda_i = self.pert_emb_lambda
            base_emb[j] += lambda_i * pert_global_emb[pert_index[1][i]]
        base_emb = base_emb.reshape(num_graphs * self.num_genes, -1)
        
        '''
        ## add the gene expression positional embedding
        gene_base = x[:, 0].reshape(-1,1)
        gene_emb = self.gene_basal_w(gene_base)
        combined = gene_emb+base_emb
        combined = self.bn_gene_base_trans(combined)
        base_emb = self.gene_base_trans_w(combined)
        base_emb = self.bn_gene_base(base_emb)
        '''
        ## add the perturbation positional embedding
        pert_emb = self.pert_w(pert)
        combined = pert_emb+base_emb
        combined = self.bn_pert_base_trans(combined)
        base_emb = self.pert_base_trans_w(combined)
        base_emb = self.bn_pert_base(base_emb)
        
        ## apply the first MLP
        base_emb = self.transform(base_emb)
        func_out = None
        #base_emb = self.bn_final(base_emb)
        
        if self.indv_out_layer:
            out = self.recovery_w(base_emb)
            out = out.reshape(num_graphs, self.num_genes, -1)
            out = out.unsqueeze(-1) * self.indv_w1
            w = torch.sum(out, axis = 2)
            out = w + self.indv_b1
            #out = self.act(out)

            #out = out.unsqueeze(-1) * self.indv_w2
            #w = torch.sum(out, axis = 2)
            #out = w + self.indv_b2
            #out = self.act(out)
            
            #out = out.unsqueeze(-1) * self.indv_w_out
            #w = torch.sum(out, axis = 2)
            #out = w + self.indv_b_out

            if self.cross_gene_MLP:
                # Compute global gene embedding
                cross_gene_embed = self.cross_gene_state(out.squeeze())

                # repeat embedding num_genes times
                cross_gene_embed = cross_gene_embed.repeat(1, self.num_genes)

                # stack it under out
                cross_gene_embed = cross_gene_embed.reshape([num_graphs,self.num_genes, -1])
                cross_gene_out = torch.cat([out, cross_gene_embed], 2)

                # First pass through MLP
                cross_gene_out = cross_gene_out * self.indv_w2
                cross_gene_out = torch.sum(cross_gene_out, axis=2)
                cross_gene_out = cross_gene_out + self.indv_b2

                ## Two layer MLP (doesn't work)
                #cross_gene_out = cross_gene_out.unsqueeze(1) * self.indv_w2
                #cross_gene_out = torch.sum(cross_gene_out, axis=3)
                #cross_gene_out = cross_gene_out + self.indv_b2

                # Second pass through MLP
                # cross_gene_out = cross_gene_out * self.indv_w3
                # cross_gene_out = torch.sum(cross_gene_out, axis=1)
                # out = cross_gene_out + self.indv_b3

            out = out.reshape(num_graphs * self.num_genes, -1) + x[:, 0].reshape(-1,1)
            # Add cross gene MLP

            out = torch.split(torch.flatten(out), self.num_genes)
            
        else:
            ## apply the final MLP to predict delta only and then add back the x. 
            out = self.recovery_w(base_emb) + x[:, 0].reshape(-1,1)
            out = torch.split(torch.flatten(out), self.num_genes)

        if self.args['func_readout']:
            func_out = torch.flatten(self.func_w2(torch.stack(out)))

        ## uncertainty head
        if self.uncertainty:
            out_logvar = self.uncertainty_w(base_emb)
            out_logvar = torch.split(torch.flatten(out_logvar), self.num_genes)
            return torch.stack(out), torch.stack(out_logvar), func_out
        
        return torch.stack(out), func_out
        
