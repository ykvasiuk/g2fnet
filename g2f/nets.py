import torch
from torch.optim import AdamW
from torch.nn import L1Loss 
          
from graph_layers import MPGNN, GridAggregation, calculate_ngp_positions
from conv_layers import UNet3D

import pytorch_lightning as pl

class LightG2Fnet(pl.LightningModule):
    def __init__(self, 
                 r_link: float, 
                 node_in: int, 
                 node_emb: int, 
                 edge_emb: int,
                 global_in: int | None,
                 global_emb: int | None,
                 n_mlp: int,  
                 n_graph: int, 
                 unet_base_c: int,
                 n_out: int,
                 n_pix: int = 128,
                 Nd: int = 3,
                 lr: float = 2e-3,
                 wd: float = 1e-2):
        super().__init__()
        
        self.r_link = r_link
        self.node_in = node_in
        self.node_emb = node_emb
        self.edge_emb = edge_emb
        self.global_in = global_in
        self.global_emb = global_emb
        self.n_mlp = n_mlp
        self.n_graph = n_graph
        self.unet_base_c = unet_base_c
        self.n_out = n_out
        self.n_pix = n_pix
        self.Nd = Nd
        self.lr = lr
        self.wd = wd

        self.gnn = MPGNN(self.r_link, 
                         self.node_in, 
                         self.node_emb, 
                         node_out=self.n_out, 
                         edge_emb=self.edge_emb,
                         global_in=self.global_in,
                         global_emb = self.global_emb,
                         n_mlp_layers=self.n_mlp,
                         n_graph_layers=self.n_graph,
                         n_pix=self.n_pix,
                         Nd=self.Nd)
        
        self.grid_agg = GridAggregation(r=2/self.n_pix, dims=self.n_pix, num_channels=self.n_out)
        self.cnn = torch.compile(UNet3D(self.n_out, 
                          self.n_out, 
                          base_channels=self.unet_base_c,
                          depth=2))
        
        self.loss = L1Loss()

        self.save_hyperparameters()
    
    @staticmethod
    def nb(x):
        return x.batch.max()+1
        
    def forward(self, x):
        n_pix = self.n_pix
        gnn_out = self.gnn(x)
        
        n_graphs = x.batch.max()+1

        y_pos = calculate_ngp_positions(self.n_pix, 1. ,device=x.pos.device).repeat(n_graphs,1)

        y_batch = torch.tensor([[i] * self.n_pix**self.Nd for i in range(n_graphs)], device=x.batch.device, dtype=torch.long).flatten()
        
        y = self.grid_agg(x.pos, gnn_out, y_pos, x.batch, y_batch)
        cnn_out = self.cnn(y.reshape(n_graphs,*(self.n_pix,)*self.Nd,self.n_out).permute(0,4,1,2,3))
        
        return cnn_out
    
    def _calculate_loss(self, data):
        cnn_out = self.forward(data['gal'])
        target = data['dens'].view_as(cnn_out)
        loss = self.loss(cnn_out, target)
        
        return loss
        
    
    def training_step(self, data, batch_idx):
        
        loss = self._calculate_loss(data)
        
        self.log('Losses/train_loss', loss, on_step=True, on_epoch=True, batch_size=self.nb(data['gal']))
        
        return {'loss': loss}
        
    
    def validation_step(self, data, batch_idx):
        
        loss = self._calculate_loss(data)
        
        self.log('Losses/val_loss', loss, on_step=True, on_epoch=True, batch_size=self.nb(data['gal']))
        
        return {'val_loss':loss}
    
    def predict_step(self, data, batch_idx):
        return self.forward(data['gal'])
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(),
                          lr = self.lr,
                          weight_decay = self.wd)
        return [optimizer]