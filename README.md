# g2fnet
This repository contains the code for [Reconstruction of Continuous Cosmological Fields from Discrete Tracers with Graph Neural Networks](https://arxiv.org/abs/2411.02496) - a contribution to NeurIPS-ml4ps-2024. It's a hybrid GNN-CNN architecture designed to infer the dark-matter and electron density fields directly from the galaxy catalogs.

## Dependencies
* torch 
* torch_geometric 
* torch_scatter
* torch_cluster
* pytorch_lightning 
* tqdm
* h5py

Additionally, Pylians and Camels_library for pre- and postprocessing.

The project requires a modification of torch_cluster.radius to take care of the periodicity. It's implemented only with cuda for now. After having created the environment with all the packages do the following:
1. activate the env
2. go to periodic_radius_cuda
3. run "python setup.py"

## Code Overview
* ./periodic_radius_cuda/ - a modification of torch_cluster.radius that takes into account the periodic boundary conditions. 
* ./g2f/conv_layer.py - a torch implementation of circular-padded 3D UNet with skip connections.
* ./g2f/data_utils.py - various helper utilities to work with CAMELS-IllustrisTNG simulations
* ./g2f/graph_layers.py - GNN building blocks
* ./g2f/make_snaps.py - a script to generate density fields from the simulation snapshots
* ./g2f/nets.py - the architecture
* ./g2f/train.py - a training script
* ./g2f/utils.py - periodic-radius wrapper utility
