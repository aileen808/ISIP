# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 14:32:30 2018

@author: AZhang6
"""

import os
import numpy as np
os.chdir("./../../iFindTimeSeriesData")

import matplotlib
from sklearn import cluster, datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler,RobustScaler
data_V = np.zeros((5000,300))
count = 0 
for j in range(50):
    data = np.load("batch"+str(j)+".npy")
    label = np.load("batch"+str(j)+"_ISIPlabel.npy")
    tmpdata = data[:,1:,:]
    for i in range(len(data)):
        r = MaxAbsScaler().fit_transform(X=tmpdata[i,0,:].reshape(-1, 1)).flatten()
        p = MinMaxScaler().fit_transform(X=tmpdata[i,1,:].reshape(-1, 1)).flatten()
        if(min(r)<-0.5 or min(p)<-0.5 or label[i]==0):
            continue
        data_V[count,0:150] = r
        data_V[count,150:] = p
        count +=1
data_V = data_V[0:count]    

l = len(data_V)
#cc1 = np.zeros((l,l))
#cc2 = np.zeros((l,l))
#for i in range(l):
#    if(i%100 ==0):
#        print(i)
#    for j in range(i,l,1):
#        sig1 = data_V[i,0:150]
#        sig2 = data_V[j,0:150]
#        sigp1 = data_V[i,150:]
#        sigp2 = data_V[j,150:]
#        cc1[i,j] = max(np.correlate(sig1,sig2)/(sum(sig1**2)*sum(sig2**2))**0.5)
#        cc1[j,i] = cc1[i,j]
#        cc2[i,j] = max(np.correlate(sigp1,sigp2)/(sum(sigp1**2)*sum(sigp2**2))**0.5)
#        cc2[j,i] = cc2[i,j]
#        
        

#dis = np.zeros((l,l))
#dis = 1-cc2
#np.save('disM',dis)


from sklearn import cluster
db = cluster.SpectralClustering(n_clusters= 10,  affinity = 'precomputed').fit(dis)
labels = db.labels_
print(max(labels))

for j in range(max(labels)+1):
    ind = np.where(labels==j)[0]
    print(len(ind))
    stack_signal = np.median(data_V[ind,:],axis=0)
    plt.figure()
    for ii in range(len(ind)):
         plt.plot(data_V[ind[ii],0:150],c = [0,0,0.6,0.1],linewidth = 0.5)
         plt.plot(data_V[ind[ii],150:],c = [0.6,0,0,0.1],linewidth = 0.5)
    
    
    plt.plot(stack_signal[0:150],'b')
    plt.plot(data_V[ind[0],0:150],'b--')
    plt.plot(stack_signal[150:],'r')
    plt.plot(data_V[ind[0],150:],'r--')
    plt.ylim([-0.2 , 1.2])
    plt.title(str(len(ind))+" Traces for Cluster "+str(j+1))