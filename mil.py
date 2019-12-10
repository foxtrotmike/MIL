# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 10:41:35 2019
A simple example of multiple instance learning using an implementation of
"From Group to Individual Labels using Deep Features" by Kotzias et al.
@author: fayyaz
"""

import numpy as np
import matplotlib.pyplot as plt
from plotit import plotit
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
from torch.utils.data import Dataset, DataLoader
#from sklearn.metrics import roc_auc_score as auc_roc
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from copy import deepcopy
from scipy.spatial import distance_matrix
USE_CUDA = torch.cuda.is_available()

def cuda(v):
    if USE_CUDA:
        return v.cuda()
    return v
def tensor(v):
    return  cuda(Variable(torch.tensor(v)).float())
from numpy.random import randn #importing randn

def getExamples(n=800,d=2):
    """
    Generates n d-dimensional normally distributed examples of each class        
    The mean of the positive class is [1] and for the negative class it is [-1]    
    """
    Xp1 = randn(int(n/2),d)+3.0#+1   #generate examples of the positie class    
    Xp2 = randn(int(n/2),d)-9.0#+1   #generate examples of the positie class
    Xp = np.vstack((Xp1,Xp2))
    
    Xn1 = randn(int(n/2),d)-3.0#-1   #generate n examples of the negative class
    Xn2 = randn(int(n/2),d)+7.0
    Xn = np.vstack((Xn1,Xn2))
    
    X = np.vstack((Xp,Xn))  #Stack the examples together to a single matrix
    y = np.array([+1]*Xp.shape[0]+[-1]*Xn.shape[0]) #Associate Labels
    
    Noise = randn(n,d)+[-3,5] #generate noise
    Noise = np.vstack((Noise, randn(n,d)+[-3,-11]))
    X = np.vstack((X,Noise)) #add noise
    y = np.append(y,[0]*len(Noise))
    
    X+=2
#    y = -y
    ridx = np.random.permutation(range(len(y)))
    X,y = X[ridx,:],y[ridx]
    return X,y

def genBags(y):
    """
    Add examples to bags
        Positive bag: has at least one positive example 
        mexpb: maximum number of examples per bag
        mprop: proportion of majority class in a bag
        nprop: proportion of noise class in a bag
        
    """
    pid,nid,noise = list(np.where(y==1)[0]),list(np.where(y==-1)[0]),list(np.where(y==0)[0])   
    
    Nbags = 20 #number of bags
    mepb = 30 # max number of example per bag
    mprop = 0.1 #majority proportion
    nprop = 0.00 #noise proportion per bag
    Bsize = np.random.binomial(n=mepb, p=0.5, size=Nbags)
    print("Avg. Examples/Bag:",np.mean(Bsize))
    Bidx = []
    Y = np.array([-1]*int(Nbags/2)+[1]*int(Nbags/2))#np.random.choice([-1, 1], size=(Nbags,), p=[0.5, 0.5])
    for i in range(len(Y)):
        M = int(np.ceil(Bsize[i]*mprop))
        n = int(Bsize[i]*nprop)
        m = Bsize[i]-M-n 
        if Y[i]==1:
            B = pid[:M]; pid = pid[M:] #add M examples from the positive class
#            print("Pos",len(B))
            B+= nid[:m]; nid = nid[m:] #add m examples from the negative class            
        else:
            B = nid[:M]; nid = nid[M:] #add M+m examples from negative class
            B+= nid[:m]; nid = nid[m:]

        B+= noise[:n]; noise = noise[n:] #add n examples of noise
        
        Bidx.append(np.array(B))
        
    return Bidx,Y

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.classifier = nn.Sequential(nn.Linear(2, 20),
                                        #nn.BatchNorm1d(8),
                                        nn.Tanh(),
                                        nn.Linear(20,1))
    def forward(self,x):
        
        x=self.classifier(x)        
        return x

def hinge(y_true, y_pred):
    zero = torch.Tensor([0]) 
    return torch.mean(torch.max(zero, 1 - y_true * y_pred))
    
#%%

def Experiment():
    X,y = getExamples()
    B,Y = genBags(y)
    model = cuda(Model())
    optmod=optim.Adam(model.parameters(), lr=0.01,weight_decay=0.000010, betas=(0.9, 0.999))
    
    episodes = 1000
    aggop =   torch.mean #  torch.max #
    L = []
    Pid,Nid = np.where(Y==1)[0],np.where(Y==-1)[0]
    for e in range(episodes):
        
        pidx,nidx = np.random.choice(Pid),np.random.choice(Nid)
        Bp,Bn = X[B[pidx]],X[B[nidx]] 
        
        Zp,Zn = model(tensor(Bp)),model(tensor(Bn))
        
        zp,zn = aggop(Zp), aggop(Zn)
        zz = cuda(Variable(torch.from_numpy(np.array([0.0]))).type(torch.FloatTensor))
        loss=torch.max(zz, (zn-zp+1)) #hinge(1.0, zp)+hinge(-1.0, zn)#
        
        Zpn = torch.cat((Zp,Zn))
        Zpn = 1/(1+torch.exp(-Zpn))
        Bpn = np.vstack((Bp,Bn))
        K = tensor(np.exp(-0.01*distance_matrix(Bpn,Bpn)))
        s = torch.mean(K*((Zpn-Zpn.T)**2))
        
        tloss = loss+1*s
    
        
        optmod.zero_grad()
        tloss.backward(retain_graph=True)
        optmod.step()
      
        L.append([float(tloss),float(loss),float(s)])
        
    #%%    
    
    plt.close("all")
    plt.figure()    
    plt.plot(np.log(L))        
    
        
    for param in model.parameters():
        param.requires_grad =False
            
    def clf(inputs): 
      return model(tensor(inputs))
    
    Z = np.array([float(aggop(clf(X[b]))) for b in B])
    plt.figure()
    plotit(X,y,clf=clf, conts=[np.min(Z),0.5*(np.mean(Z[Y==1])+np.mean(Z[Y==-1])),np.max(Z)],colors='scaled')
    for b in Pid:
        plt.plot(X[B[b],0],X[B[b],1],'^')
    for b in Nid:
        plt.plot(X[B[b],0],X[B[b],1],'v')
        
    from roc import roc
    fpr,tpr,auci = roc(list(y[y!=0]),list(clf(X[y!=0])))
    plt.figure()
    plt.plot(fpr,tpr);plt.title("Instance level: "+str(auci)); plt.axis([0,1,0,1])
    fpr,tpr,aucg = roc(list(Y),list(Z))
    plt.figure()
    plt.plot(fpr,tpr)  
    fpr,tpr,auc = roc(list(Y),[aggop(tensor(y[b])) for b in B])
    plt.plot(fpr,tpr);plt.axis([0,1,0,1])
    plt.legend(['group predictions: '+str(aucg),'group labels: '+str(auc)])
    return auci,aucg,auc

if __name__=='__main__':
    for i in range(5):
        z = Experiment()
        print(i,z)
        
