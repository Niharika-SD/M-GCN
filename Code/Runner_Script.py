#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import numpy as np
import os,sys

import scipy.io as sio
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import pickle

from sklearn.metrics import mean_absolute_error as mae
from scipy.stats import pearsonr

#my libs
from Loader import Load_Dataset
from Modules import M_GCN,init_weights

def train(net, epoch):
    
    net.train() #train mode

    #loss criteria
    criterion2 = torch.nn.L1Loss()
    criterion = torch.nn.MSELoss()
    
    optimizer = torch.optim.SGD(net.parameters(),lr=lr,momentum=momentum,nesterov=True,weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.9, last_epoch=-1)
    
    #pre-allocate    
    total = 0
    running_loss = 0.0
    
    for batch_idx, (inputs, inputs_2, targets) in enumerate(trainloader):
        
        optimizer.zero_grad() #zero gradients
        
        inputs, inputs_2, targets = Variable(inputs),Variable(inputs_2), Variable(targets)
        
        mask = (targets>0).type(torch.FloatTensor)  #exclude unknowns
        outputs = net(inputs,inputs_2,graph_flag) #forward pass
        
        #backprop
        
        loss = targets.size(1)*(torch.sqrt(criterion(outputs.mul(mask), targets.mul(mask))) 
                                + criterion2(outputs.mul(mask), targets.mul(mask))) 
        
        loss.backward(retain_graph=True)

        optimizer.step() 
        
        # store loss
        running_loss += loss.data.numpy()
        total += targets.size(0)
    
    #scheduler step    
    scheduler.step()

    return running_loss/batch_idx

def validation(net):
       
    net.eval() #eval mode
    
    total = 0
    running_loss = 0.0
    
    #loss criteria
    criterion2 = torch.nn.L1Loss()
    criterion = torch.nn.MSELoss()
    
    for batch_idx, (inputs, inputs_2, targets) in enumerate(valloader):
        
        with torch.no_grad(): #no gradient calculation
        
            inputs,inputs_2, targets = Variable(inputs),Variable(inputs_2), Variable(targets)

            mask = (targets>0).type(torch.FloatTensor) #exclude unknowns
            outputs = net(inputs,inputs_2,graph_flag) #forward pass
            
            #loss compute          
            loss = targets.size(1)*( torch.sqrt(criterion(outputs.mul(mask), targets.mul(mask))) 
                                    + criterion2(outputs.mul(mask), targets.mul(mask)))

            running_loss += loss.data.numpy()
            total += targets.size(0)
  
    
    return running_loss/batch_idx



if __name__ == '__main__':   
   
    #presets
    momentum = 0.9
    lr = 0.001 #learning rate
    wd = 0.001 ## Decay for L2 regularization 
    nbepochs = 40
    
    val = False
    graph_flag = True
    
    num_classes = 1
    
    behavdir = "//home/niharika-shimona/Documents/Projects/Autism_Network/M_GCN/Data/HCP/"
    filename = "Dataset_HCP"
    
    if graph_flag:
        folder_name = behavdir + '/Outputs/G_F/'
    else:
        folder_name = behavdir + '/Outputs/no_G_F/'
    
            
    if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
    #log outputs  
    log_filename = folder_name + 'logfile1.txt'
    log = open(log_filename, 'w')
    sys.stdout = log
    
    # Loaders
    trainset =  Load_Dataset(directory=behavdir, mode= "train",filename= filename)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=0)
    
    testset =  Load_Dataset(directory=behavdir, mode= "test",filename= filename)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

    if val:
        valset =  Load_Dataset(directory=behavdir, mode= "validation",filename= filename)
        valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=0)

    #initialize
    net = M_GCN(trainset.X,num_classes)
    net.apply(init_weights)
    

    # Run Epochs of training 
    # and validation   

    allloss_train = []
    allloss_val= []

    for epoch in range(nbepochs):
       
        loss_train = train(net,epoch)
        allloss_train.append(loss_train)     
         
        Y_train_pred = net.forward(trainset.X,trainset.L,graph_flag).detach().numpy()
        Y_train_meas = trainset.Y.detach().numpy()
       
        Y_train_pred[Y_train_meas==0] = 0 # for unknowns
        
        if val:
            
            loss_val = validation(net)
            allloss_val.append(loss_val)   
            
            Y_val_pred = net.forward(valset.X,valset.L,graph_flag).detach().numpy()
            Y_val_meas = valset.Y.detach().numpy()
        
            Y_val_pred[Y_val_meas==0] = 0
            
            Y_test_pred = net.forward(testset.X,testset.L,graph_flag).detach().numpy()
            Y_test_meas = testset.Y.detach().numpy()
       
            Y_test_pred[Y_test_meas==0] = 0 # for unknowns                   

            
            print("Epoch %d  || Loss Train %f || Loss Val %f " % (epoch,loss_train,loss_val))
            dict_save = {'Y_train_meas':Y_train_meas,'Y_train_pred':Y_train_pred,
                     'Y_val_meas':Y_val_meas,'Y_val_pred':Y_val_pred,
                     'Y_test_meas':Y_test_meas,'Y_test_pred':Y_test_pred}
        
        else:
            
            print("Epoch %d  || Loss Train %f " % (epoch,loss_train))
            
            Y_test_pred = net.forward(testset.X,testset.L,graph_flag).detach().numpy()
            Y_test_meas = testset.Y.detach().numpy()
       
            Y_test_pred[Y_test_meas==0] = 0 # for unknowns                       
    
            dict_save = {'Y_train_meas':Y_train_meas,'Y_train_pred':Y_train_pred,
                         'Y_test_meas':Y_test_meas,'Y_test_pred':Y_test_pred}
    
    
    #save performance
    filename_mat = '/Perf.mat'
    sio.savemat(folder_name+filename_mat,dict_save) 
    
    #save model
    dict_cvf = {'model': net}
    filename_models =  folder_name + '/model.p'
    pickle.dump(dict_cvf, open(filename_models, "wb"))

    #loss curves   
    fig,ax = plt.subplots()

    ax.plot(list(range(nbepochs)),allloss_train,'r',label='train')

    if val:
        ax.plot(list(range(nbepochs)),allloss_val,'b',label='val')
    
    ax.legend(loc='upper left')
       
    plt.title('Loss',fontsize=16)
    plt.ylabel('Error' ,fontsize=12)
    plt.xlabel('num of iterations',fontsize=12)
    
    plt.show()
    #save fig
    
    figname = folder_name + '/Loss.png'
    fig.savefig(figname)   # save the figure to file
    plt.close(fig)