import torch
from torch import nn
from torch.nn import Sequential, Linear, ReLU, ModuleList, LayerNorm, Dropout, MSELoss, SiLU, Dropout

import numpy as np

from torch_scatter import scatter, scatter_mean, scatter_sum, scatter_max, scatter_min, scatter_add

from utils import periodic_radius
from torch_geometric.nn import MessagePassing, MetaLayer, radius_graph, global_mean_pool, global_max_pool, global_add_pool, MultiAggregation

import torch.nn.functional as F


def calculate_ngp_positions(dims, BoxSize, device, Nd=3):
    inv_cell_size = dims / BoxSize
    cell_size = 1 / inv_cell_size
    
    indices = torch.arange(dims,device=device)
    
    positions = (indices) * cell_size
    mesh_coords = torch.meshgrid(*[positions,]*Nd, indexing='ij')
    
    positions = torch.stack(mesh_coords, axis=-1)
    return positions.reshape(-1,Nd)




class MLP_block(nn.Module):
    def __init__(self, n_inp, n_out, n_hidden, n_layers):
        super().__init__()
        layer = [Linear(n_inp, n_hidden),
                 LayerNorm(n_hidden),
                 SiLU()]
        
        for i in range(n_layers-1):
            layer.extend([Linear(n_hidden, n_hidden),
                          LayerNorm(n_hidden),
                          SiLU(),
                          Dropout(p=0.3)])
        layer.append(Linear(n_hidden,n_out))
        self.mlp = Sequential(*layer)
        
    def forward(self,x):
        return self.mlp(x)

class GraphEmbeddingModel(torch.nn.Module):
    def __init__(self, r_link,
                 node_input_size,
                 node_embedding_size,
                 edge_embedding_size, 
                 vel=True,
                 global_input_size=None,
                 global_embedding_size=None,
                 equivariant=False):
        super().__init__()
        self.equivariant = equivariant
        self.r_link = r_link
        self.node_embedding = MLP_block(node_input_size, node_embedding_size, node_embedding_size, 1)
        self.vel = vel
        edge_input_size = 6 if vel else 3
        self.edge_embedding = MLP_block(edge_input_size, edge_embedding_size, edge_embedding_size, 1)
        self.global_embedding = MLP_block(global_input_size, global_embedding_size, global_embedding_size, 1) if global_input_size is not None else None
    
    def forward(self, data):
        x = data.x
        pos = data.pos 
        batch = data.batch
        
        
        edge_index = radius_graph(pos, r = self.r_link, batch=batch, loop = False)
        
        senders = edge_index[0]
        receivers = edge_index[1]
        
        r_ij = pos[senders] - pos[receivers]
        to_cat = [r_ij]
        
        if self.vel:
            v_ij = data.vel[senders] - data.vel[receivers]
            to_cat.append(v_ij)
        
        
        edge_attr = self.edge_embedding(torch.cat(to_cat,axis=1))
        
        x = self.node_embedding(x)
        
        u = None
        if self.global_embedding is not None:
            u = self.global_embedding(data.u)
        
        return x, edge_index, edge_attr, u, batch
    

    

class EdgeModel(torch.nn.Module):
    def __init__(self, node_in, edge_in, hidden_size, edge_out, n_layers, global_in=0):
        super().__init__()
        self.edge_mlp = MLP_block(edge_in+2*node_in+global_in, edge_out, hidden_size, n_layers)
        self.global_in = global_in
    def forward(self, src, dst, edge_attr, u, batch):
        # src, dst: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        to_cat = [src, dst, edge_attr]
        if u is not None and self.global_in != 0:
            to_cat.append(u[batch])
        out = torch.cat(to_cat, 1)
        return self.edge_mlp(out)
    
        
class NodeModel(torch.nn.Module):
    def __init__(self, node_in, edge_in, hidden_size, node_out, n_layers, global_in=0):
        super().__init__()
        self.node_mlp = MLP_block(node_in + 3*edge_in+global_in, node_out, hidden_size, n_layers)
        self.multi_aggr = MultiAggregation(aggrs=['sum', 'max', 'mean'], mode='cat')
        self.global_in = global_in
    def forward(self, x, edge_index, edge_attr, u, batch):
        
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        
        row, col = edge_index
        
        aggr_out = self.multi_aggr(edge_attr, index=col, dim_size=x.size(0))

        to_cat = [aggr_out, x]
        
        if u is not None and self.global_in != 0:
            to_cat.append(u[batch])
        
        out = torch.cat(to_cat, dim=1)
        
        out = self.node_mlp(out)
        
        return out   
                    
    
class GraphLayer(torch.nn.Module):
    def __init__(self, node_in, edge_in, node_out, edge_out, node_hidden, edge_hidden, n_layers, global_in=0):
        super().__init__()
        self.layer = MetaLayer(edge_model=EdgeModel(node_in = node_in, edge_in = edge_in, hidden_size = edge_hidden, edge_out = edge_out, n_layers=n_layers, global_in=global_in),
                               node_model=NodeModel(node_in = node_in, edge_in = edge_out, hidden_size = node_hidden, node_out = node_out, n_layers=n_layers, global_in=global_in))
        
    def forward(self, x, edge_index, edge_attr, u, batch):
        return self.layer(x, edge_index, edge_attr, u, batch=batch)
        


def CICW_d_torch(pos, W, batch, dims):
    BoxSize = 1

    batch_size = batch.max().item() + 1
    N_c = W.shape[1]  # Number of channels
    inv_cell_size = dims / BoxSize
    
    number = torch.zeros((batch_size, N_c, dims, dims, dims), dtype=torch.float32, device=pos.device)
    
    dist = pos * inv_cell_size
    
    index_d = dist.floor().long() % dims
    index_u = (index_d + 1) % dims
    
    u = dist - index_d.float()
    d = 1.0 - u
    
    u = u.unsqueeze(1).repeat(1, N_c, 1)
    d = d.unsqueeze(1).repeat(1, N_c, 1)
    
    w_d0d1d2 = d[:, :, 0] * d[:, :, 1] * d[:, :, 2] * W
    w_d0d1u2 = d[:, :, 0] * d[:, :, 1] * u[:, :, 2] * W
    w_d0u1d2 = d[:, :, 0] * u[:, :, 1] * d[:, :, 2] * W
    w_d0u1u2 = d[:, :, 0] * u[:, :, 1] * u[:, :, 2] * W
    w_u0d1d2 = u[:, :, 0] * d[:, :, 1] * d[:, :, 2] * W
    w_u0d1u2 = u[:, :, 0] * d[:, :, 1] * u[:, :, 2] * W
    w_u0u1d2 = u[:, :, 0] * u[:, :, 1] * d[:, :, 2] * W
    w_u0u1u2 = u[:, :, 0] * u[:, :, 1] * u[:, :, 2] * W
    
    batch_expanded = batch.unsqueeze(1).expand(-1, N_c)
    
    # Accumulate contributions to the grid using index_put_
    number.index_put_((batch_expanded, torch.arange(N_c, device=pos.device).unsqueeze(0).expand(batch.size(0), -1), index_d[:, 0].unsqueeze(1).expand(-1, N_c), index_d[:, 1].unsqueeze(1).expand(-1, N_c), index_d[:, 2].unsqueeze(1).expand(-1, N_c)), w_d0d1d2, accumulate=True)
    number.index_put_((batch_expanded, torch.arange(N_c, device=pos.device).unsqueeze(0).expand(batch.size(0), -1), index_d[:, 0].unsqueeze(1).expand(-1, N_c), index_d[:, 1].unsqueeze(1).expand(-1, N_c), index_u[:, 2].unsqueeze(1).expand(-1, N_c)), w_d0d1u2, accumulate=True)
    number.index_put_((batch_expanded, torch.arange(N_c, device=pos.device).unsqueeze(0).expand(batch.size(0), -1), index_d[:, 0].unsqueeze(1).expand(-1, N_c), index_u[:, 1].unsqueeze(1).expand(-1, N_c), index_d[:, 2].unsqueeze(1).expand(-1, N_c)), w_d0u1d2, accumulate=True)
    number.index_put_((batch_expanded, torch.arange(N_c, device=pos.device).unsqueeze(0).expand(batch.size(0), -1), index_d[:, 0].unsqueeze(1).expand(-1, N_c), index_u[:, 1].unsqueeze(1).expand(-1, N_c), index_u[:, 2].unsqueeze(1).expand(-1, N_c)), w_d0u1u2, accumulate=True)
    number.index_put_((batch_expanded, torch.arange(N_c, device=pos.device).unsqueeze(0).expand(batch.size(0), -1), index_u[:, 0].unsqueeze(1).expand(-1, N_c), index_d[:, 1].unsqueeze(1).expand(-1, N_c), index_d[:, 2].unsqueeze(1).expand(-1, N_c)), w_u0d1d2, accumulate=True)
    number.index_put_((batch_expanded, torch.arange(N_c, device=pos.device).unsqueeze(0).expand(batch.size(0), -1), index_u[:, 0].unsqueeze(1).expand(-1, N_c), index_d[:, 1].unsqueeze(1).expand(-1, N_c), index_u[:, 2].unsqueeze(1).expand(-1, N_c)), w_u0d1u2, accumulate=True)
    number.index_put_((batch_expanded, torch.arange(N_c, device=pos.device).unsqueeze(0).expand(batch.size(0), -1), index_u[:, 0].unsqueeze(1).expand(-1, N_c), index_u[:, 1].unsqueeze(1).expand(-1, N_c), index_d[:, 2].unsqueeze(1).expand(-1, N_c)), w_u0u1d2, accumulate=True)
    number.index_put_((batch_expanded, torch.arange(N_c, device=pos.device).unsqueeze(0).expand(batch.size(0), -1), index_u[:, 0].unsqueeze(1).expand(-1, N_c), index_u[:, 1].unsqueeze(1).expand(-1, N_c), index_u[:, 2].unsqueeze(1).expand(-1, N_c)), w_u0u1u2, accumulate=True)

    return number


class GridAggregation(MessagePassing):
    def __init__(self, r, dims, num_channels, hidden_dim=16):
        super(GridAggregation, self).__init__(aggr='add')  # "Add" aggregation (weighted sum)
        self.r = r
        self.dims = dims
        self.num_channels = num_channels

        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),  
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_channels)
        )

    def forward(self, pos_in, h_in, pos_out, batch_in, batch_out):
        # for each idx of "out" gives the corresponding indices of "in"
        with torch.no_grad():
            row, col = periodic_radius(pos_in, pos_out, r=self.r, batch_x=batch_in, batch_y=batch_out)
        
            delta = torch.abs(pos_in[col] - pos_out[row])
            delta = torch.minimum(delta, 1. - delta)*self.dims
        
        weights = self.mlp(torch.sum(delta**2,dim=-1).view(-1,1))  
        
        edge_index = torch.stack([col, row], dim=0)
        
        out = self.propagate(edge_index=edge_index, h=(h_in, None), weights=weights, size=(pos_in.shape[0], pos_out.shape[0]))
        
        return out

    def message(self, h_j, weights):
        # h_j are the input features of the neighboring input nodes
        return weights * h_j  

    def update(self, aggr_out):
        return aggr_out 

        
class MPGNN(torch.nn.Module):
    def __init__(self, r_link, 
                 node_in, 
                 node_emb, 
                 node_out, 
                 edge_emb,
                 global_in,
                 global_emb,
                 n_mlp_layers,
                 n_graph_layers,
                 n_pix,
                 Nd,):
        super().__init__()
        self.n_pix = n_pix
        self.Nd = Nd
        self.embedding = GraphEmbeddingModel(
                 r_link,
                 node_input_size = node_in,
                 node_embedding_size = node_emb, 
                 edge_embedding_size = edge_emb,
                 global_input_size=global_in,
                 global_embedding_size=global_emb,
                 vel=False,
                 equivariant=False)
        
        self.graph_layers = []
        self.n_graph_layers = n_graph_layers
        global_emb = 0 if global_emb is None else global_emb
        for i in range(n_graph_layers):
            self.graph_layers.append(GraphLayer(node_in=node_emb, 
                                      edge_in=edge_emb,
                                      node_out=node_emb,
                                      edge_out=edge_emb,
                                      node_hidden=node_emb,
                                      edge_hidden=edge_emb,
                                      n_layers=n_mlp_layers,
                                      global_in=global_emb))
            
        self.graph_layers = nn.ModuleList(self.graph_layers)    
        
        self.final_node_mlp = MLP_block(n_inp=node_emb, n_out=node_out, n_hidden=node_emb, n_layers=n_mlp_layers)
        
    def forward(self, data):
        x, edge_index, edge_attr, u, batch = self.embedding(data)
        for layer in self.graph_layers:
            x, edge_attr, u = layer(x, edge_index, edge_attr, u, batch = batch)
        
        x = self.final_node_mlp(x)
        return x
             