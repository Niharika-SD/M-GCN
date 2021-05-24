# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 18:05:02 2020

@author: niharika-shimona
"""



# coding: utf-8

# In[1]:


import torch
import numpy as np

import torch.nn.functional as F
import torch.nn
from torch.autograd import Variable

import torch.backends.cudnn as cudnn

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pickle
from numpy.testing import rundocs
#use_cuda = torch.cuda.is_available()
use_cuda = 0

# %reset

# In[2]:


class E2EBlock(torch.nn.Module):
    '''E2Eblock.'''

    def __init__(self, in_planes, planes,example,bias=False):
        super(E2EBlock, self).__init__()
        self.d = example.size(3)
        self.in_planes = example.size(1)
        self.cnn1 = torch.nn.Conv2d(in_planes,planes,(1,self.d),bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes,planes,(self.d,1),bias=bias)

        
    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a]*self.d,3)+torch.cat([b]*self.d,2)
        #return torch.cat([a]*self.d,3)


def init_weights(m):
    
    np.random.seed(0) 
    
    if type(m) == torch.nn.Linear:
            
            torch.nn.init.xavier_normal(m.weight, gain=torch.nn.init.calculate_gain('relu'))
#        torch.nn.init.uniform_(m.weight, a=0, b=1e-04)
            m.bias.data.fill_(1e-03)
# BrainNetCNN Network for fitting Gold-MSI on LSD dataset

# In[3]:


class BrainNetCNN(torch.nn.Module):
    def __init__(self, example, num_classes=10):
        super(BrainNetCNN, self).__init__()
        self.in_planes = example.size(1)
        self.d = example.size(3)
        
        self.e2econv1 = E2EBlock(1,32,example,bias=True)
        self.e2econv2 = E2EBlock(8,64,example,bias=True)
        self.E2N = torch.nn.Conv2d(32,1,(1,self.d))
        self.N2G = torch.nn.Conv2d(1,256,(self.d,1))
        
        self.dense1 = torch.nn.Linear(256,128)
        self.dense2 = torch.nn.Linear(128,30)
        self.dense3 = torch.nn.Linear(30,3)
        
    def forward(self, x, y):
#        print x.size()
#        
#        n,r,c = x.size()
        # x = torch.matmul(y,x) # graph filtering
#        print x.size()
        out_fMRI = F.leaky_relu(self.e2econv1(x),negative_slope=0.1)
        E2E_out = out_fMRI
#        print self.e2econv1.cnn1.weight.size()
        
        # out_fMRI = torch.matmul(y,out_fMRI) # graph filtering
#        out_fMRI = F.leaky_relu(self.e2econv2(out_fMRI),negative_slope=0.1)
     
        # out_fMRI = torch.matmul(y,out_fMRI) # graph filtering        
        out_fMRI = F.leaky_relu(self.E2N(out_fMRI),negative_slope=0.1)
       
        E2N_out = out_fMRI
        # out_fMRI = torch.matmul(y,out_fMRI) # graph filtering     
       
        out_fMRI = F.leaky_relu(self.N2G(out_fMRI),negative_slope=0.1)
     
        N2G_out = out_fMRI
        out_fMRI = out_fMRI.view(out_fMRI.size(0), -1)     
        out = F.leaky_relu(self.dense1(out_fMRI),negative_slope=0.1)
        out = F.leaky_relu(self.dense2(out),negative_slope=0.1)
        out = F.leaky_relu(self.dense3(out),negative_slope=0.1)
        
        return out,E2E_out,E2N_out,N2G_out


# Loader for GoldMSI-LSD77 dataset

# In[20]:
    
import sys

data_size = 5

train_dat_size =  'data_all_sub_'+ str(data_size) 
# = sys.argv[1]
behavdir = "/home/niharika-shimona/Documents/Projects/Autism_Network/Sparse-Connectivity-Patterns-fMRI/Weighted_Frob_Norm/DTI_data/Multi_Aut_DTIfMRI_CV/Final/"

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import os
import scipy.io as sio

import torch.utils.data.dataset

class GoldMSI_LSD_Dataset(torch.utils.data.Dataset):

    def __init__(self, directory=behavdir,mode="train",transform=False,class_balancing=False):
        """
        Args:
            directory (string): Path to the dataset.
            mode (str): train = 90% Train, validation=10% Train, train+validation=100% train else test.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.directory = directory
        self.mode = mode
        self.transform = transform
 
#        x = sio.loadmat(os.path.join(directory,"data_complete.mat"))['corr_test'][0][cvf][:][:]
#        y_all = sio.loadmat(os.path.join(directory,"data_complete.mat"))['Y']            
#        y_2=y_all[:,[3,4]]
        
  #      y = normalize(y_all,axis=1)
        strtrain =  str(train_dat_size) +".mat"
        X_train = sio.loadmat(os.path.join(directory, strtrain))['corr_train'][0][cvf][:][:]
        X_train_DTI = sio.loadmat(os.path.join(directory, strtrain ))['L_train'][0][cvf][:][:]
        Y_train_temp =  sio.loadmat(os.path.join(directory, strtrain ))['Y_train'][0][cvf]
        X_test = sio.loadmat(os.path.join(directory, strtrain ))['corr_test'][0][cvf][:][:]
        X_test_DTI = sio.loadmat(os.path.join(directory, strtrain ))['L_test'][0][cvf][:][:]
        Y_test_temp =  sio.loadmat(os.path.join(directory, strtrain ))['Y_test'][0][cvf]

        
        f_0 = 1.0/3.0
        f_1 = 1.0/15.0
        f_2 = 1.0/10.0
        
        Y_train = torch.zeros((Y_train_temp.shape[0],Y_train_temp.shape[1]))
        Y_test = torch.zeros((Y_test_temp.shape[0],Y_test_temp.shape[1]))
        
#        Y_train = torch.zeros((Y_train_temp.shape[0],1))
#        Y_test = torch.zeros((Y_test_temp.shape[0],1))
        
        for i in range(Y_train.size()[1]):
            
            if (i == 1):
               
               Y_train[:,i] = (f_1)*(torch.from_numpy((Y_train_temp[:,i]).ravel()).float())
               Y_test[:,i] = (f_1)*(torch.from_numpy((Y_test_temp[:,i]).ravel()).float())
       
            elif(i == 2):
               
               Y_train[:,i] = (f_2)*(torch.from_numpy((Y_train_temp[:,i]).ravel()).float())
               Y_test[:,i] = (f_2)*(torch.from_numpy((Y_test_temp[:,i]).ravel()).float())
       
            else:
               
               Y_train[:,i] = (f_0)*(torch.from_numpy((Y_train_temp[:,i]).ravel()).float())
               Y_test[:,i] = (f_0)*(torch.from_numpy((Y_test_temp[:,i]).ravel()).float())
#        Y_train = f_0*torch.from_numpy(Y_train_temp).float()
#        Y_test = f_0*torch.from_numpy(Y_test_temp).float()
#        

        
        if self.mode=="train":
            
            x = X_train
            l = X_train_DTI
            y = Y_train
            
        elif self.mode=="validation":
            x = X_test
            l = X_test_DTI
            y = Y_test
            
        elif mode=="train+validation":
            x=x
            y=y
            l=l
        else:
            x=x
            y=y
            l=l
            
            
        self.X = torch.FloatTensor(np.expand_dims(x,1).astype(np.float32))

        self.L = torch.FloatTensor(np.expand_dims(l,1).astype(np.float32))
        #self.X = torch.FloatTensor(x.astype(np.float32))
        self.Y = y
         
        print(self.mode,self.X.shape,(self.Y.shape))
            
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
       
        
        sample = [self.X[idx], self.L[idx], self.Y[idx]]
        if self.transform:
            sample[0] = self.transform(sample[0])
            
            sample[1] = self.transform(sample[1])
        return sample

def init_weights_he(m):
    #https://keras.io/initializers/#he_uniform
    print(m)
    if type(m) == torch.nn.Linear:
        fan_in = net.dense1.in_features
        he_lim = np.sqrt(6) / fan_in
        m.weight.data.uniform_(-he_lim,he_lim)
        print(m.weight)

def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    running_loss = 0.0
    
    for batch_idx, (inputs, inputs2, targets) in enumerate(trainloader):
        
        if use_cuda:
            inputs, inputs2, targets = inputs.cuda(),inputs2.cuda(), targets.cuda()
        
        optimizer.zero_grad()
        inputs, inputs2, targets = Variable(inputs),Variable(inputs2), Variable(targets)
        
        mask = (targets>0).type(torch.FloatTensor)
        outputs = net(inputs,inputs2)
        
        loss = criterion(outputs[0].mul(mask), targets.mul(mask)) + criterion2(outputs[0].mul(mask), targets.mul(mask))
        # loss = torch.norm(outputs[0].mul(mask) - targets.mul(mask),2) + torch.norm(outputs[0].mul(mask) - targets.mul(mask),1)
        
        # loss = loss/(targets.size()[0])
        
        loss.backward(retain_graph=True)
        optimizer.step() 
        
        
        # print statistics
        running_loss += loss.data

       
        # if batch_idx % 10 == 9:    # print every 10 mini-batches
        #     print('Training loss: %.6f' % ( running_loss /10.0))
            
#        _, predicted = torch.max(outputs.data, 1)
        
        total += targets.size(0)
        
        #correct += predicted.eq(targets.data).cpu().sum()
    scheduler.step()

    return running_loss/total

def test(net):
    
    
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    running_loss = 0.0
    
    preds = []
    ytrue = []
    
    for batch_idx, (inputs,inputs2, targets) in enumerate(testloader):
        
        if use_cuda:
            inputs,inputs2 ,targets = inputs.cuda(),inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs,inputs2, targets = Variable(inputs),Variable(inputs2), Variable(targets)

            outputs = net(inputs,inputs2)
            
            mask = (targets>0).type(torch.FloatTensor)
            
            loss = criterion(outputs[0].mul(mask), targets.mul(mask)) + criterion2(outputs[0].mul(mask), targets.mul(mask))
            # loss = torch.norm(outputs[0].mul(mask) - targets.mul(mask),2) + torch.norm(outputs[0].mul(mask) - targets.mul(mask),1)
            # loss = loss/targets.size()[0]
            
            test_loss += loss.data
            
            preds.append(outputs[0].numpy())
            ytrue.append(targets.numpy())
        
    
    
        
        # print statistics
        running_loss += loss.data.numpy()
        if batch_idx % 5 == 4:    # print every 5 mini-batches
            print('Test loss: %.6f' % ( running_loss/batch_idx))
            
        
        #_, predicted = torch.max(outputs.data, 1)
        #total += targets.size(0)
        #correct += predicted.eq(targets.data).cpu().sum()

    
    return np.vstack(preds),np.vstack(ytrue),running_loss/(batch_idx)
    # Save checkpoint.
    #acc = 100.*correct/total
    

for k in range(1,10):

  for cvf in range(5):
   
    momentum = 0.9
    lr = 0.0001
    wd = 0.001 ## Decay for L2 regularization 
    import sys
    nbepochs = 40
    
    folder_name = behavdir + 'GCN_Multi/no_DTI/' + str(k) + '_iter/'
    # + str(lr) + '_lr/' + str(wd) + '_wd/' 
    
            
    if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        
    log_filename = folder_name + 'logfile_'+ str(cvf) +'.txt'

    log = open(log_filename, 'w')
    sys.stdout = log
    

    trainset = GoldMSI_LSD_Dataset(mode="train")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=0)

    testset = GoldMSI_LSD_Dataset(mode="validation")
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=0)


# Training


    


# In[22]:


    net = BrainNetCNN(trainset.X,trainset.L)
    
    foldername_preload = '//home/niharika-shimona/Documents/Projects/Autism_Network/Sparse-Connectivity-Patterns-fMRI/Weighted_Frob_Norm/DTI_data/Multi_Aut_DTIfMRI_CV/GCN_Multi/Pre-train/E2E/Not_pretrained/23_iter/32_HL/'
    filename_preload = foldername_preload +'/model_cvf' + str(0) + '.p'
    
    with open(filename_preload, 'rb') as f:
            data_pl = pickle.load(f)

    model = data_pl['model']
    
    # net.e2econv1.cnn1.weight =  model.e2econv1.cnn1.weight 
    # net.e2econv1.cnn2.weight =  model.e2econv1.cnn2.weight 
    # net.E2N.weight =  model.E2N.weight
    # net.N2G.weight =  model.N2G.weight
    # net.dense1.weight = model.dense1.weight
    # net.dense2.weight = model.dense2.weight
    # net.dense3.weight = model.dense3.weight
            
        
   
#wd = 0


### Weights initialization for the dense layers using He Uniform initialization
### He et al., http://arxiv.org/abs/1502.01852


    net.apply(init_weights)


    criterion2 = torch.nn.L1Loss()
    criterion = torch.nn.MSELoss()
    
    optimizer = torch.optim.SGD(net.parameters(),lr=lr,momentum=momentum,nesterov=True,weight_decay=wd)
#    optimizer = torch.optim.Adam(net.parameters(), lr=lr,weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.9, last_epoch=-1)

# In[23]:




# Run Epochs of training and testing 

# In[ ]:


    from sklearn.metrics import mean_absolute_error as mae
    from scipy.stats import pearsonr

    allloss_train = []
    allloss_test= []

    allmae_test1 = []
    allpears_test1 = []

    allmae_test2 = []
    allpears_test2 = []


    for epoch in range(nbepochs):
        loss_train = train(epoch)
    
        allloss_train.append(loss_train)
    
        net.eval() 
        preds,y_true,loss_test = test(net)
    
        # allloss_test.append(loss_test/16)
         
        Y_train_pred = net.forward(trainset.X,trainset.L)[0].detach().numpy()
        Y_train_meas = trainset.Y.detach().numpy()
        
        Y_test_pred = net.forward(testset.X,testset.L)[0].detach().numpy()
        Y_test_meas = testset.Y.detach().numpy()
        
        
        preds = Y_test_pred
        y_true = Y_test_meas    
        

        Y_train_pred[Y_train_meas==0] = 0
        Y_test_pred[Y_test_meas==0] = 0
        
        
        mae_1 = mae(preds[:,0],y_true[:,0])
        pears_1 = pearsonr(preds[:,0],y_true[:,0])
        
        # mae_2 = mae(preds[:,1],y_true[:,1])
        # pears_2 = pearsonr(preds[:,1],y_true[:,1])
    
        allmae_test1.append(mae_1)
        allpears_test1.append(pears_1)
        
        print("Epoch %d" % epoch)
        # allmae_test2.append(mae_2)
        # allpears_test2.append(pears_2)
        
#        print("Test Set : MAE for ADOS: %0.2f %%" % (mae_1))
#        print("Test Set : pearson R for ADOS : %0.2f, p = %0.2f" % (pears_1[0],pears_1[1]))
#
#        mae_2 = mae(preds[:,1],y_true[:,1])
#        pears_2 = pearsonr(preds[:,1],y_true[:,1])
#    
#        allmae_test2.append(mae_2)
#        allpears_test2.append(pears_2)
#    
#        print("Test Set : MAE for SRS : %0.2f %%" % (mae_2))
#        print("Test Set : pearson R for SRS : %0.2f, p = %0.2f" % (pears_2[0],pears_2[1]))
#
#        mae_3 = mae(preds[:,2],y_true[:,2])
#        pears_3 = pearsonr(preds[:,2],y_true[:,2])
#    
#        allmae_test2.append(mae_3)
#        allpears_test2.append(pears_3)
#    
#        print("Test Set : MAE for Praxis : %0.2f %%" % (mae_3))
#        print("Test Set : pearson R for Praxis : %0.2f, p = %0.2f" % (pears_3[0],pears_3[1]))

                

    dict_save = {'Y_train_meas':Y_train_meas,'Y_train_pred':Y_train_pred,
                     'Y_test_meas':Y_test_meas,'Y_test_pred':Y_test_pred}

        
        
    filename_mat = '/Perf_' + str(cvf) + '.mat'
 
    sio.savemat(folder_name+filename_mat,dict_save) 
    dict_cvf = {'model': net}
    filename_models =  folder_name + '/model_cvf' + str(cvf) + '.p'
    pickle.dump(dict_cvf, open(filename_models, "wb"))
    
    dict_save2 = {'E2Er': net.e2econv1.cnn1.weight.detach().numpy(), 'E2Ec': net.e2econv1.cnn2.weight.detach().numpy()}   
    output_filename = folder_name + str(cvf) +'_out.mat'
    sio.savemat(output_filename,dict_save2)
        
    fig3,ax3 = plt.subplots()
    ax3.plot(list(range(nbepochs)),allloss_train,'r',label='train')
    # ax3.plot(list(range(nbepochs)),allmae_test2,'g',label='test score2')
    ax3.plot(list(range(nbepochs)),allmae_test1,'b',label='test score1')
    # ax3.plot(list(range(nbepochs)),allpears_test1,'g',label='test score1')
    ax3.legend(loc='upper left')
       
    plt.title('Loss',fontsize=16)
    plt.ylabel('Error' ,fontsize=12)
    plt.xlabel('num of iterations',fontsize=12)
    plt.show()
    figname3 = folder_name + '/Loss_'+ str(cvf) +'.png'
    fig3.savefig(figname3)   # save the figure to fil
    plt.close(fig3)

