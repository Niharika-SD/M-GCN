#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

import torch.nn.functional as F
import torch.nn

from C_B_Layer import Connectome_Filter_Block

class M_GCN(torch.nn.Module):
    
    def __init__(self, input_ex, num_classes):
        super(M_GCN, self).__init__()
        
        self.in_planes = input_ex.size(1)
        self.d = input_ex.size(3)
        
        self.cf_1 = Connectome_Filter_Block(1,32,input_ex,bias=True)      
        self.ef_1 = torch.nn.Conv2d(32,1,(1,self.d))
        self.nf_1 = torch.nn.Conv2d(1,256,(self.d,1))
        
        #ANN for regression
        self.dense1 = torch.nn.Linear(256,128)
        self.dense2 = torch.nn.Linear(128,30)
        self.dense3 = torch.nn.Linear(30,num_classes)
        
    def forward(self, x, l, g_f):

        out = F.leaky_relu(self.cf_1(x, l, g_f),negative_slope=0.1)

        if g_f:  # graph filtering     
            out = torch.matmul(l,out) # graph filtering
            
        out = F.leaky_relu(self.ef_1(out),negative_slope=0.1)
       
        if g_f:  # graph filtering     
            out = torch.matmul(l,out)
       
        out = F.leaky_relu(self.nf_1(out),negative_slope=0.1)
     
        #regression
        out = out.view(out.size(0), -1)     
        out = F.leaky_relu(self.dense1(out),negative_slope=0.1)
        out = F.leaky_relu(self.dense2(out),negative_slope=0.1)
        out = F.leaky_relu(self.dense3(out),negative_slope=0.1)
        
        return out
    
def init_weights(m):
    
    if type(m) == torch.nn.Linear:
            
            torch.nn.init.xavier_normal(m.weight, gain=torch.nn.init.calculate_gain('relu'))
            m.bias.data.fill_(1e-02)