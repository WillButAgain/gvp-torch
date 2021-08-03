import os
import random

from preprocess_protien import parse_pdb

class DataFolder(object):
    '''
    Only grabs .PDB files
    '''
    def __init__(self, data_folder, device, shuffle=False):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.shuffle = shuffle
        self.device = device

    def __len__(self):
        return len(self.data_files)
    
    def __iter__(self):
        if self.shuffle: random.shuffle(self.data_files) # shuffle data before epoch
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            if not ('.pdb' in fn.lower()): # maybe want to ignore any other files in directories, such as .README files explaining data
                continue
            yield parse_pdb(fn, device=self.device, inject_noise=False)
            
            
class NoisyDataFolder(object):
    '''
    Only grabs .PDB files
    '''
    def __init__(self, data_folder, device, shuffle=False):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.shuffle = shuffle
        self.device = device

    def __len__(self):
        return len(self.data_files)
    
    def __iter__(self):
        if self.shuffle: random.shuffle(self.data_files) # shuffle data before epoch
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            if not ('.pdb' in fn.lower()): # maybe want to ignore any other files in directories, such as .README files explaining data
                continue
            yield (parse_pdb(fn, inject_noise=False, device=self.device), parse_pdb(fn, inject_noise=True, device=self.device)) # (x, target)
