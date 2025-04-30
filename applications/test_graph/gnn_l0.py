import torch
import os
import json
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
import torch.nn as nn
import torch_pruning as tp
from torch_geometric.nn.inits import uniform
from base_layers import BaseModel
from base_layers import L0Dense as Linear
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder


class GNN(BaseModel):

    def __init__(self, num_class, num_layer = 5, emb_dim = 300, 
                 gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5
                 , JK = "last", graph_pooling = "mean", FLAG = 'ogbg-molhiv'):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()
        
        script_dir = os.path.dirname(__file__)
        with open(f'{script_dir}/settings.json') as f:
            settings = json.load(f)
        
        weight_decay = settings["weight_decay"]
        lambas = settings["lambas"]
        device = settings["device"]
        use_reg = settings["use_reg"]
        local_rep = settings["local_rep"]
        temperature = settings["temperature"]
        droprate_init = settings["droprate_init"]
        budget = settings["budget"]
        beta_ema = settings["beta_ema"]

        self.beta_ema = beta_ema
        self.N = settings["N"]
        self.budget = budget
        self.device = device
        self.temperature = temperature

        self.num_layer = num_layer
        self.drop_ratio = droprate_init
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_class = num_class
        self.graph_pooling = graph_pooling
        
        self.FLAG = FLAG

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        
        if FLAG=='ogbg-ppa':
            from .conv_l0 import GNN_node, GNN_node_Virtualnode
        else:
            from .conv_mol_l0 import GNN_node, GNN_node_Virtualnode

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type, 
                                                 droprate_init=droprate_init, temperature=temperature, budget=budget, weight_decay=weight_decay, 
                                                 lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type,
                                     droprate_init=droprate_init, temperature=temperature, budget=budget, weight_decay=weight_decay, 
                                     lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim, droprate_init=droprate_init, 
                                                                                      temperature=temperature, budget=budget, weight_decay=weight_decay, 
                                                                                      lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg), 
                                                                      torch.nn.BatchNorm1d(2*emb_dim), 
                                                                      torch.nn.ReLU(), 
                                                                      torch.nn.Linear(2*emb_dim, 1, droprate_init=droprate_init, 
                                                                                      temperature=temperature, budget=budget, weight_decay=weight_decay, 
                                                                                      lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = Linear(2*self.emb_dim, self.num_class, droprate_init=droprate_init, temperature=temperature, 
                                            budget=budget, weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, 
                                            device=device, use_reg=use_reg)
        else:
            self.graph_pred_linear = Linear(self.emb_dim, self.num_class, droprate_init=droprate_init, temperature=temperature, 
                                            budget=budget, weight_decay=weight_decay, lamba=lambas, local_rep=local_rep, 
                                            device=device, use_reg=use_reg)
            
        self.layers = []
        for m in self.modules():
            if isinstance(m, Linear):
                self.layers.append(m)

    def forward(self, batched_data):
        batch = batched_data[-1]
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batch)

        return self.graph_pred_linear(h_graph)
    
    def build_dependency_graph(self):
        if self.FLAG=='ogbg-ppa':
            from .conv_l0 import GCNConv, GINConv
        else:
            from .conv_mol_l0 import GCNConv, GINConv
            
        dependency_dict = {}
        pre_module = None
        
        conv_count = 0
        bn_count = 0
        bn_masks_name = []
        conv_masks_name = []
        for name, module in self.named_modules():
            if isinstance(module, GCNConv):
                dependency_dict[name] = {'in_mask': None, 'out_mask': None, 'type': 'gcncov'}
                dependency_dict[name]['in_mask'] = module.linear.mask
                if conv_count==0:
                    if 'gnn_node.atom_encoder' in dependency_dict.keys():
                        dependency_dict['gnn_node.atom_encoder']['out_mask'] = module.linear.mask
                    if 'gnn_node.node_encoder' in dependency_dict.keys():
                        dependency_dict['gnn_node.node_encoder']['out_mask'] = module.linear.mask
                else:        
                    dependency_dict[conv_masks_name[conv_count-1]]['out_mask'] = module.linear.mask
                conv_masks_name.append(name)
                bn_masks_name.append(name)
                conv_count += 1
            
            if isinstance(module, GINConv):
                dependency_dict[name] = {'in_mask': None, 'out_mask': None, 'type': 'gcncov'}
                dependency_dict[name]['in_mask'] = module.mlp[0].mask
                if conv_count==0:
                    if 'gnn_node.atom_encoder' in dependency_dict.keys():
                        dependency_dict['gnn_node.atom_encoder']['out_mask'] = module.mlp[0].mask
                    if 'gnn_node.node_encoder' in dependency_dict.keys():
                        dependency_dict['gnn_node.node_encoder']['out_mask'] = module.mlp[0].mask
                else:        
                    dependency_dict[conv_masks_name[conv_count-1]]['out_mask'] = module.mlp[0].mask
                conv_masks_name.append(name)
                bn_masks_name.append(name)
                conv_count += 1
            
            elif name=='gnn_node.atom_encoder':
                dependency_dict[name] = {'in_mask': None, 'out_mask': None, 'type': name}
            elif name=='gnn_node.node_encoder':
                dependency_dict[name] = {'in_mask': None, 'out_mask': None, 'type': name}
            elif name=='graph_pred_linear':
                dependency_dict[name] = {'in_mask': module.mask, 'out_mask': None, 'type': name}
                dependency_dict[conv_masks_name[-1]]['out_mak'] = module.mask
            elif isinstance(module, nn.BatchNorm1d):
                if 'gnn_node.batch_norm' in name:
                    dependency_dict[name] = {'in_mask': dependency_dict[bn_masks_name[bn_count]]['out_mask'], 
                                            'out_mask': None, 'type': 'bn'}
                    bn_count += 1
            else:
                continue

        return dependency_dict
    
    def prune_model(self):
        if self.FLAG=='ogbg-ppa':
            from .conv_l0 import GCNConv, GINConv
        else:
            from .conv_mol_l0 import GCNConv, GINConv
        
        for layer in self.layers:
            if isinstance(layer, Linear):
                if layer.use_reg:
                    layer.prepare_for_inference()
        dependency_dict = self.build_dependency_graph()

        for name, module in self.named_modules():
            if name in dependency_dict.keys():
                if isinstance(module, (GCNConv, GINConv)):
                    module.prune(in_mask=dependency_dict[name]['in_mask'], out_mask=dependency_dict[name]['out_mask'])
                
                elif isinstance(module, nn.BatchNorm1d):
                    if dependency_dict[name]['in_mask']:
                        tp.prune_batchnorm_in_channels(module, idxs=dependency_dict[name]['in_mask'])
                
                elif name=='gnn_node.node_encoder':
                    out_mask = dependency_dict['gnn_node.node_encoder']['out_mask']
                    if out_mask:
                        tp.prune_embedding_out_channels(module, idxs=out_mask)
                
                elif name=='gnn_node.atom_encoder':
                    out_mask = dependency_dict['gnn_node.atom_encoder']['out_mask']
                    if out_mask:
                        for name, sub_module in module.named_modules():
                            if isinstance(sub_module, torch.nn.Embedding):
                                tp.prune_embedding_out_channels(sub_module, idxs=out_mask)

                elif name=='graph_pred_linear':
                    in_mask = dependency_dict['graph_pred_linear']['in_mask']
                    if in_mask:
                        tp.prune_linear_in_channels(module, idxs=in_mask)
                
                else:
                    print(f'{name} is not support prune!')

if __name__ == '__main__':
    GNN(num_class = 10)