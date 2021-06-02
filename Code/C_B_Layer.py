#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch

import torch.nn.functional as F
import torch.nn


class Connectome_Filter_Block(torch.nn.Module):
    
    '''Connectome Filter Block'''

    def __init__(self, n_filt, planes, input_ex, bias=False):
       
        super(Connectome_Filter_Block, self).__init__() #initialize
       
        self.d = input_ex.size(3) 
        self.in_planes = input_ex.size(1)
        
        self.cnn1 = torch.nn.Conv2d(n_filt,planes,(1,self.d),bias=bias) #row 
        self.cnn2 = torch.nn.Conv2d(n_filt,planes,(self.d,1),bias=bias) #column

        
    def forward(self, x, l, g_flag):
        
        '''
        Input : 
            x -> rs-fMRI connectome
            l -> DTI Laplacian
            g_flag -> graph filtering on
        '''
        
        if g_flag: #graph pre-filtering if True
            x = torch.matmul(l,x) 
        
        r = self.cnn1(x) #row filtering
        c = self.cnn2(x) #column filterning
        
        return torch.cat([r]*self.d,3)+torch.cat([c]*self.d,2)
        
