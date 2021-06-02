#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import scipy.io as sio
from Modules import M_GCN
import os
import numpy as np

class Load_Dataset(torch.utils.data.Dataset):

    def __init__(self, directory, mode, filename):
        
        """
        Args:
            directory (string): Path to the dataset.
            mode: train or validation
        """
        self.directory = directory
        self.mode = mode
        self.filename = filename
        
        strtrain =  str(filename) +".mat"
        
        if self.mode=="train":
            
            X_train = sio.loadmat(os.path.join(directory, strtrain))['corr_train'][:][:]
            L_DTI_train = sio.loadmat(os.path.join(directory, strtrain ))['L_train'][:][:]
            Y_train =  sio.loadmat(os.path.join(directory, strtrain ))['Y_train']
            
            x = X_train
            l = L_DTI_train
            y = Y_train
            
        elif self.mode=="validation":
           
            X_val = sio.loadmat(os.path.join(directory, strtrain ))['corr_val'][:][:]
            L_DTI_val = sio.loadmat(os.path.join(directory, strtrain ))['L_val'][:][:]
            Y_val =  sio.loadmat(os.path.join(directory, strtrain ))['Y_val']
            
            x = X_val
            l = L_DTI_val
            y = Y_val
            
        else:
            
            X_test = sio.loadmat(os.path.join(directory, strtrain ))['corr_test'][:][:]
            L_DTI_test = sio.loadmat(os.path.join(directory, strtrain ))['L_test'][:][:]
            Y_test =  sio.loadmat(os.path.join(directory, strtrain ))['Y_test']
            
            x = X_test
            l = L_DTI_test
            y = Y_test
                                   
        self.X = torch.FloatTensor(np.expand_dims(x,1).astype(np.float32))
        self.L = torch.FloatTensor(np.expand_dims(l,1).astype(np.float32))        
        self.Y = torch.FloatTensor(y.astype(np.float32))
                    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
           
        sample = [self.X[idx], self.L[idx], self.Y[idx]] 
        return sample

