import MAS_library as MASL
import camels_library as CL
import numpy as np
import os
import h5py
simsettype = "LH"
path = f"/scratch2/yurii/CAMELS/IllustrisTNG/{simsettype}/"
from itertools import product
from tqdm import tqdm

#dims = 64
dims=128

def get_density_field(pos,grid, Nd=3, W=None, MAS='CIC',L=25, overd=True):

    BoxSize = L
    verbose = False   
    
    delta = np.zeros((grid,)*Nd, dtype=np.float32)
    MASL.MA(pos, delta, BoxSize, MAS, W=W, verbose=verbose)
    if overd:
        delta = (delta/delta.mean()-1.)
    return delta  

def get_densities(path):
    n_e = CL.electron_density(path+'/snapshot_090.hdf5').astype(np.float32)
    data = h5py.File(path+'/snapshot_090.hdf5')
    pos_gas = data['PartType0']['Coordinates'][:].astype(np.float32)/1e3
    pos_dm = data['PartType1']['Coordinates'][:].astype(np.float32)/1e3
    delta_e = get_density_field(pos_gas, dims, W=n_e)
    delta_m = get_density_field(pos_dm, dims)
    return {'de': delta_e, 'dm': delta_m}

if __name__ == "__main__":
    #simsettype = "LH"
    #path = f"/scratch2/yurii/CAMELS/IllustrisTNG/{simsettype}/"
    #otppath = os.path.join(path,"densities_3D")
    #if not os.path.exists(otppath):
    #    os.makedirs(otppath)
    #for i in tqdm(range(1000)):
    #    inp_path = os.path.join(path,f'{simsettype}_{i}')
    #    otp_filename_dens = otppath+f'/{simsettype}_{i}_dens_{dims}.npz'
    #    np.savez(otp_filename_dens,**get_densities(inp_path))
        
    simsettype = "CV"
    path = f"/scratch2/yurii/CAMELS/IllustrisTNG/{simsettype}/"
    otppath = os.path.join(path,"densities_3D")
    if not os.path.exists(otppath):
        os.makedirs(otppath)
    for i in tqdm(range(27)):
        inp_path = os.path.join(path,f'{simsettype}_{i}')
        otp_filename_dens = otppath+f'/{simsettype}_{i}_dens_{dims}.npz'
        np.savez(otp_filename_dens,**get_densities(inp_path))    
        