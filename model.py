import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

from torch_geometric.nn import GINConv, GCNConv, GATConv, GraphConv
from torch.nn import Sequential, Linear, ReLU
import pandas as pd

import sys

sys.path.append('/dfs/user/yhr/cell_reprogram/model/')
from flow import get_graph, get_expression_data, \
    add_weight, I_TF, get_TFs, solve, \
    solve_parallel, get_expression_lambda


# Helpers
def weighted_mse_loss(input, target, weight):
    sample_mean = torch.mean((input - target) ** 2, 1)
    return torch.mean(weight * sample_mean)


class linear_model():
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


# GNN architecture 1
class GNN_1(torch.nn.Module):
    """
    Creates node level embeddings. All nodes for each graph are concatenated
    at the end of the forward call.

    """

    def __init__(self, num_feats, num_genes, num_layers, hidden_size,
                 embed_size, GNN):
        super(GNN_1, self).__init__()
        self.mp_layers = num_layers
        self.pre_mp = nn.Linear(num_feats, hidden_size)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.embed_size = embed_size
        self.num_genes = num_genes

        for l in range(self.mp_layers):
            layer = Sequential(
                Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                Linear(hidden_size, hidden_size)
            )
            if GNN == 'GIN':
                self.convs.append(GINConv(layer))
            self.bns.append(nn.BatchNorm1d(hidden_size))
        self.post_mp = Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, embed_size),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.pre_mp(x)
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
        x = self.convs[-1](x, edge_index)
        x = self.post_mp(x)

        x = torch.split(torch.flatten(x), self.num_genes * self.embed_size)
        return torch.stack(x)


# GNN architecture 2
class GNN_2(torch.nn.Module):
    def __init__(self, num_feats, num_genes, hidden_size, embed_size, GNN):
        super(GNN_2, self).__init__()
        torch.manual_seed(12345)
        self.num_genes = num_genes
        self.embed_size = embed_size
        if GNN == 'GCN':
            self.conv1 = GCNConv(num_feats, hidden_size)
            self.conv2 = GCNConv(hidden_size, hidden_size)
            self.conv3 = GCNConv(hidden_size, hidden_size)
        elif GNN == 'GraphConv':
            self.conv1 = GraphConv(num_feats, hidden_size)
            self.conv2 = GraphConv(hidden_size, hidden_size)
        elif GNN == 'GAT':
            self.conv1 = GATConv(num_feats, hidden_size)
            self.conv2 = GATConv(hidden_size, hidden_size)
            # self.conv3 = GATConv(hidden_size, hidden_size)
        self.lin = Linear(hidden_size, embed_size)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, \
                                          data.edge_attr, data.batch

        # 1. Obtain node embeddings
        out = self.conv1(x, edge_index, edge_weight=edge_attr)
        #x = x.relu()
        #x = self.conv2(x, edge_index, edge_weight=edge_attr)
        # x = x.relu()
        # x = self.conv3(x, edge_index)
        # out = self.lin(out)

        #x = F.dropout(x, p=0.5, training=self.training)

        # Assumed differential loss
        out = out[:,0] - x[:, 0]

        out = torch.split(torch.flatten(out), self.num_genes * self.embed_size)
        return torch.stack(out)


class GNN_node_specific(torch.nn.Module):
    """
    TODO very memomry inefficient implementation that doesn't work right now
    Node specific GNN
    (i) [GNN] message passing over gene-gene graph with expression and
    perturbations together represented as two dimensional node features
    FOR EACH NODE separately
    """

    def __init__(self, num_node_features, gene_list, node_hidden_size,
                 node_embed_size, ae_num_layers, ae_hidden_size, GNN,
                 device, encode=True):
        super(GNN_node_specific, self).__init__()

        num_genes = 1
        ae_input_size = node_embed_size * num_genes

        self.node_GNN = {}
        for idx, _ in enumerate(gene_list):
            self.node_GNN[idx] = GNN_2(num_node_features, num_genes,
                                       node_hidden_size, node_embed_size,
                                       GNN=GNN).to(device)

        self.encode = encode
        self.encoder = MLP(
            [ae_input_size] + [ae_hidden_size] * ae_num_layers + [
                ae_hidden_size])
        self.decoder = MLP(
            [ae_hidden_size] + [ae_hidden_size] * ae_num_layers + [num_genes],
            last_layer_act='linear')

    def forward(self, data):
        # Run the input through all (node-specific) GNNs
        x = [self.node_GNN[idx](data) for idx in range(len(self.node_GNN))]

        # Pass gradients only through the output for each node
        # out =

        if self.encode:
            encoded = self.encoder(x)
            x = self.decoder(encoded)

        return x


class GNN_AE(torch.nn.Module):
    """
    GNN + AE Model consisting of two steps:
    (i) [GNN] message passing over gene-gene graph with expression and
    perturbations together represented as two dimensional node features
    (ii) [AE] "auto-encoder" to convert concatenated node embeddings to post
    perturbation expression state
    """

    def __init__(self, num_node_features, num_genes,
                 gnn_num_layers, node_hidden_size, node_embed_size,
                 ae_num_layers, ae_hidden_size, GNN, GNN_arch=2,
                 encode=True):
        super(GNN_AE, self).__init__()

        ae_input_size = node_embed_size * num_genes
        self.encode = encode
        if GNN_arch == 1:
            self.GNN = GNN_1(num_node_features, num_genes, gnn_num_layers,
                             node_hidden_size, node_embed_size, GNN=GNN)
        elif GNN_arch == 2:
            self.GNN = GNN_2(num_node_features, num_genes,
                             node_hidden_size, node_embed_size, GNN=GNN)

        self.encoder = MLP(
            [ae_input_size] + [ae_hidden_size] * ae_num_layers + [
                ae_hidden_size])
        self.decoder = MLP(
            [ae_hidden_size] + [ae_hidden_size] * ae_num_layers + [num_genes],
            last_layer_act='linear')

    def forward(self, x):
        x = self.GNN(x)

        if self.encode:
            encoded = self.encoder(x)
            x = self.decoder(encoded)

        return x

    def loss(self, pred, y, perts, weight=1):

        # Weigh the loss for perturbations
        weights = np.ones(len(pred))
        non_ctrl_idx = np.where([('ctrl' != p) for p in perts])[0]
        weights[non_ctrl_idx] = weight

        loss = weighted_mse_loss(pred, y, torch.Tensor(weights).to(pred.device))
        return loss


class simple_GNN(torch.nn.Module):
    """
    shallow GNN architecture
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

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, \
                                          data.edge_attr, data.batch

        # 1. Obtain node embeddings
        if self.incl_edge_weight:
            edge_weight = edge_attr
        else:
            edge_weight = None

        x = self.conv1(x, edge_index=edge_index)
        #x = self.conv1(x, edge_index=edge_index, edge_weight=edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index=edge_index)
        x = x.relu()
        out = self.lin(x)

        out = torch.split(torch.flatten(out), self.num_genes *
                          self.node_embed_size)
        return torch.stack(out)

    def loss(self, pred, y, perts, weight=1):

        # Micro average MSE
        if self.loss_type == 'micro':
            mse_p = torch.nn.MSELoss()
            perts = np.array(perts)
            losses = torch.tensor(0.0, requires_grad=True).to(pred.device)
            for p in set(perts):
                pred_p = pred[np.where(perts==p)[0]]
                y_p = y[np.where(perts==p)[0]]
                losses += mse_p(pred_p, y_p)
            return losses/(len(set(perts)))

        else:
            # Weigh the loss for perturbations
            weights = np.ones(len(pred))
            non_ctrl_idx = np.where([('ctrl' != p) for p in perts])[0]
            weights[non_ctrl_idx] = weight
            loss = weighted_mse_loss(pred, y, torch.Tensor(weights).to(pred.device))
            return loss


class simple_GNN_AE(torch.nn.Module):
    """
    shallow GNN + AE
    """

    def __init__(self, num_feats, num_genes, hidden_size, node_embed_size,
                 incl_edge_weight, ae_num_layers, ae_hidden_size,
                 loss_type='micro'):
        super(simple_GNN_AE, self).__init__()

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

        ae_input_size = node_embed_size * num_genes
        self.encoder = MLP(
            [ae_input_size] + [ae_hidden_size] * ae_num_layers + [
                ae_hidden_size])
        self.decoder = MLP(
            [ae_hidden_size] + [ae_hidden_size] * ae_num_layers + [1],
            last_layer_act='linear')

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, \
                                          data.edge_attr, data.batch

        # 1. Obtain node embeddings
        if self.incl_edge_weight:
            edge_weight = edge_attr
        else:
            edge_weight = None

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

    def loss(self, pred, y, perts, weight=1):

        # Micro average MSE
        if self.loss_type == 'micro':
            mse_p = torch.nn.MSELoss()
            perts = np.array(perts)
            losses = torch.tensor(0.0, requires_grad=True).to(pred.device)
            for p in set(perts):
                pred_p = pred[np.where(perts==p)[0]]
                y_p = y[np.where(perts==p)[0]]
                losses += mse_p(pred_p, y_p)
            return losses/(len(set(perts)))

        else:
            # Weigh the loss for perturbations
            weights = np.ones(len(pred))
            non_ctrl_idx = np.where([('ctrl' != p) for p in perts])[0]
            weights[non_ctrl_idx] = weight
            loss = weighted_mse_loss(pred, y, torch.Tensor(weights).to(pred.device))
            return loss


class simple_AE(torch.nn.Module):
    """
    GNN for debugging
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

    def loss(self, pred, y, perts, weight=1):

        # Micro average MSE
        if self.loss_type == 'micro':
            mse_p = torch.nn.MSELoss()
            perts = np.array(perts)
            losses = torch.tensor(0.0, requires_grad=True).to(pred.device)
            for p in set(perts):
                pred_p = pred[np.where(perts == p)[0]]
                y_p = y[np.where(perts == p)[0]]
                losses += mse_p(pred_p, y_p)
            return losses / (len(set(perts)))

        else:
            # Weigh the loss for perturbations
            weights = np.ones(len(pred))
            non_ctrl_idx = np.where([('ctrl' != p) for p in perts])[0]
            weights[non_ctrl_idx] = weight
            loss = weighted_mse_loss(pred, y,
                                     torch.Tensor(weights).to(pred.device))
            return loss
