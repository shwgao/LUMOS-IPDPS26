import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
from torch_geometric.utils import degree
from base_layers import L0Dense as Linear
import torch_pruning as tp

import math

### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim, droprate_init, temperature, budget, weight_decay, 
                 lamba, local_rep, device, use_reg):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(Linear(emb_dim, 2*emb_dim, droprate_init=droprate_init, temperature=temperature, 
                                              budget=budget, weight_decay=weight_decay, lamba=lamba, local_rep=local_rep, 
                                              device=device, use_reg=use_reg), 
                                       torch.nn.BatchNorm1d(2*emb_dim), 
                                       torch.nn.ReLU(), 
                                       Linear(2*emb_dim, emb_dim, droprate_init=droprate_init, temperature=temperature, 
                                              budget=budget, weight_decay=weight_decay, lamba=lamba, local_rep=local_rep, 
                                              device=device, use_reg=use_reg))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out
    
    def prune(self, in_mask, out_mask):
        med_mask = self.mlp[3].mask
        if in_mask:
            tp.prune_linear_in_channels(self.mlp[0], idxs=in_mask)
            for name, module in self.bond_encoder.named_modules():
                if isinstance(module, torch.nn.Embedding):
                    tp.prune_embedding_out_channels(module, idxs=in_mask)
            
        if med_mask:
            tp.prune_linear_in_channels(self.mlp[3], idxs=med_mask)
            tp.prune_batchnorm_in_channels(self.mlp[1], idxs=med_mask)
            tp.prune_linear_out_channels(self.mlp[0], idxs=med_mask)
        
        if out_mask:
            tp.prune_linear_out_channels(self.mlp[3], idxs=out_mask)


### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim, droprate_init, temperature, budget, weight_decay, 
                 lamba, local_rep, device, use_reg):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = Linear(emb_dim, emb_dim, droprate_init=droprate_init, temperature=temperature, budget=budget, weight_decay=weight_decay, 
                             lamba=lamba, local_rep=local_rep, device=device, use_reg=use_reg)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)
        self.use_reg = use_reg

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)
        
        # if self.use_reg:  # todo: if this is correct
        #     edge_embedding *= self.linear.m
        #     root_emb = self.root_emb.weight * self.linear.m
        # else:
        #     root_emb = self.root_emb.weight
        root_emb = self.root_emb.weight

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + root_emb) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out
    
    def prune(self, in_mask, out_mask):
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Embedding):
                if out_mask is not None:
                    tp.prune_embedding_out_channels(module, idxs=out_mask)
            elif isinstance(module, Linear):
                if in_mask is not None:
                    tp.prune_linear_in_channels(module, idxs=in_mask)
                if out_mask is not None:
                    tp.prune_linear_out_channels(module, idxs=out_mask)
                    keep_idxs = list(set(range(self.linear.out_features+len(out_mask))) - set(out_mask))
                    keep_idxs.sort()
                    self.linear.m = torch.index_select(self.linear.m, 0, torch.LongTensor(keep_idxs).contiguous())
            else:
                continue

        self.linear.m = self.linear.m.detach().cuda()  # detach the mask from the model

### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, droprate_init, temperature, budget, weight_decay, lamba, local_rep, 
                 device, use_reg, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin',
                 ):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim, droprate_init=droprate_init, temperature=temperature, budget=budget, weight_decay=weight_decay, 
                                          lamba=lamba, local_rep=local_rep, device=device, use_reg=use_reg))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim, droprate_init=droprate_init, temperature=temperature, budget=budget, weight_decay=weight_decay, 
                                          lamba=lamba, local_rep=local_rep, device=device, use_reg=use_reg))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        # x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        x, edge_index, edge_attr, batch = batched_data
        
        ### computing input node embedding
        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation


### Virtual GNN to generate node embedding
class GNN_node_Virtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, droprate_init, temperature, budget, weight_decay, 
                 lamba, local_rep, device, use_reg, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin',
                 ):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GNN_node_Virtualnode, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim, droprate_init=droprate_init, temperature=temperature, budget=budget, weight_decay=weight_decay, 
                                          lamba=lamba, local_rep=local_rep, device=device, use_reg=use_reg))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim, droprate_init=droprate_init, temperature=temperature, budget=budget, weight_decay=weight_decay, 
                                          lamba=lamba, local_rep=local_rep, device=device, use_reg=use_reg))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(Linear(emb_dim, 2*emb_dim, droprate_init=droprate_init, 
                                                                        temperature=temperature, budget=budget, weight_decay=weight_decay, 
                                                                        lamba=lamba, local_rep=local_rep, device=device, use_reg=use_reg), 
                                                                 torch.nn.BatchNorm1d(2*emb_dim), 
                                                                 torch.nn.ReLU(), 
                                                                 Linear(2*emb_dim, emb_dim, droprate_init=droprate_init, 
                                                                        temperature=temperature, budget=budget, weight_decay=weight_decay, 
                                                                        lamba=lamba, local_rep=local_rep, device=device, use_reg=use_reg), 
                                                                 torch.nn.BatchNorm1d(emb_dim), 
                                                                 torch.nn.ReLU()))


    def forward(self, batched_data):

        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layer):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            ### Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layer - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                ### transform virtual nodes using MLP

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)
                else:
                    virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation


if __name__ == "__main__":
    pass