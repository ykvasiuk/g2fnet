import torch
from torch_geometric.data import Data,Batch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose


from tqdm import tqdm
import os
import h5py
import numpy as np
from typing import Dict




class Scale_g:
    def __call__(self, data):
        if isinstance(data, list):
            output = [self._transform(i) for i in data]
        else:
            output = self._transform(data)
        return output
    
    def _transform(self, d):
        s_ph = norm_scale(torch.tensor(d['star_ph'], dtype=torch.float), -20., 1.04)
        s_m = norm_scale(torch.log10(torch.tensor(d['star_m'], dtype=torch.float)), -0.05, 0.47)
        s_hr = norm_scale(torch.log10(torch.tensor(d['star_hr'], dtype=torch.float)), 0.6, 0.17)
        s_vd = norm_scale(torch.log10(torch.tensor(d['star_vd'], dtype=torch.float)), 1.95, 0.14)


        gal_pos = torch.tensor(d['star_pos'], dtype=torch.float)/25.
        u = torch.tensor(d['params'], dtype=torch.float32)
        data = Data(x=torch.column_stack([s_ph, s_m, s_hr, s_vd]),
                    pos=gal_pos,
                    u = u)
        return data  
    
    
def count_occurrences_in_place(array1):
    # Get unique elements and their counts in array1
    unique, counts = np.unique(array1, return_counts=True)
    count_dict = dict(zip(unique, counts))
    
    # Create result array of the same shape as array1
    array2 = np.zeros_like(array1, dtype=int)
    
    # Fill array2 with counts corresponding to elements in array1
    for i, value in enumerate(array1):
        array2[i] = count_dict[value]
    
    return array2


def get_galaxies(path:str, npart:int = 200)->Dict:
    s_cat = h5py.File(path)

    filt1 = s_cat['Subhalo/SubhaloLenType'][:,4] > npart
    filt2 = s_cat['Subhalo/SubhaloLenType'][:,1] > npart
    
    half_mass_rads = s_cat['Subhalo/SubhaloHalfmassRadType'][:,[1,4]]
    filt3 = (half_mass_rads[:,0]>0.74)*(half_mass_rads[:,1]>0.74)
    filt = (filt1*filt2*filt3).flatten()
    
    
    stars = {}
    stars['star_ph'] = s_cat['Subhalo/SubhaloStellarPhotometrics'][filt, :][:,[4,5,7]].reshape(-1,3)
    stars['star_hr']   = s_cat['Subhalo/SubhaloHalfmassRadType'][filt, 4].reshape(-1,1)
    stars['star_m']    = s_cat['Subhalo/SubhaloMassType'][filt, 4].reshape(-1,1)
    stars['star_pos']    = s_cat['Subhalo/SubhaloPos'][filt,:].reshape(-1,3)/1e3
    stars['star_vd']    = s_cat['Subhalo/SubhaloVelDisp'][filt]
    
    return stars


class CAMELS_G2F_Dataset(Dataset):
    def __init__(self, path, 
                 grid, 
                 indexes, 
                 transform_d=None, 
                 transform_g=None, 
                 simsettype='CV',
                 npart=10):
        self.transform_d = transform_d
        self.transform_g = transform_g
        self.dens = []
        self.gals = []
        self.grid = grid
        self.indexes = indexes
        self.path = path
        self.simsettype = simsettype
        self.npart = npart
        self._load_dataset_into_memory()


    def _load_dataset_into_memory(self):
        get_params = lambda lines, i: np.array(lines[i+1].split()[1:-1]).astype(np.float32).reshape(1,-1)
        
        with open(os.path.join(self.path, f'CosmoAstroSeed_IllustrisTNG_L25n256_{self.simsettype}.txt'),'r') as f:
            param_file = f.readlines()
            
        
        for idx in tqdm(self.indexes,leave=False,desc='Loading data'):
            gals = get_galaxies(os.path.join(self.path,f'{self.simsettype}_{idx}','groups_090.hdf5'),self.npart)
            dens = np.load(os.path.join(self.path,'densities_3D', f'{self.simsettype}_{idx}_dens_{self.grid}.npz'))
            densities = np.stack([dens['de'],dens['dm']],axis=0)
            gals['params'] = get_params(param_file, idx)
            self.gals.append(gals)
            self.dens.append(densities)  
    def __len__(self):
        return len(self.dens)

    def __getitem__(self, idx):
        dens = self.dens[idx]
        gal = self.gals[idx]
        if self.transform_d:
            dens = self.transform_d(dens)
        if self.transform_g:
            gal = self.transform_g(gal)
        return {'gal':gal, 'dens':dens}  
    
    
    
class ToTensor:
    def __call__(self,x):
        if isinstance(x, list):
            return [torch.tensor(i,dtype=torch.float) for i in x]
        else: 
            return torch.tensor(x,dtype=torch.float)

def min_max_scale(x, min_, max_):
    return (x-min_)/(max_-min_)

def inv_min_max_scale(y, min_, max_):
    return y * (max_ - min_) + min_

def norm_scale(x, mean, std):
    return (x-mean) / std

def inv_norm_scale(y, mean, std):
    return y*std + mean

class NormScale:
    def __init__(self, mean, std):         
        self.mean = mean
        self.std = std
        
    def __call__(self, x, inverse=False):
        fn = self._scale if not inverse else self._unscale
        if isinstance(x, list):
            output = [fn(i) for i in x]
        else:
            output = fn(x)
        return output
    
    def _broadcast_shape(self, x):
        shape = [1] * x.ndim
        shape[0] = -1 
        return shape
    
    def _scale(self, x):
        mean = self.mean.view(*self._broadcast_shape(x)).to(x.device)
        std = self.std.view(*self._broadcast_shape(x)).to(x.device)
        return norm_scale(x,mean,std)
    
    def _unscale(self, y):
        mean = self.mean.view(*self._broadcast_shape(y)).to(y.device)
        std = self.std.view(*self._broadcast_shape(y)).to(y.device)                    
        return inv_norm_scale(y, mean, std)
    

class LogNormScale:
    def __init__(self, mean, std):         
        self.mean = mean
        self.std = std
        
    def __call__(self, x, inverse=False):
        fn = self._scale if not inverse else self._unscale
        if isinstance(x, list):
            output = [fn(i) for i in x]
        else:
            output = fn(x)
        return output
    
    def _broadcast_shape(self, x):
        shape = [1] * x.ndim
        shape[0] = -1 
        return shape
    
    def _scale(self, x):
        mean = self.mean.view(*self._broadcast_shape(x)).to(x.device)
        std = self.std.view(*self._broadcast_shape(x)).to(x.device)
        return norm_scale(torch.log10(x+1.),mean,std)
    
    def _unscale(self, y):
        mean = self.mean.view(*self._broadcast_shape(y)).to(y.device)
        std = self.std.view(*self._broadcast_shape(y)).to(y.device)                    
        return 10**inv_norm_scale(y, mean, std)-1.
    
    
    
def custom_collate_fn(batch):
    
    batch_dens = torch.stack([it['dens'] for it in batch])
    batch_gal = Batch.from_data_list([it['gal'] for it in batch])
    return {'dens':batch_dens, 'gal':batch_gal}    
    