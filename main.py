# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 13:26:04 2018

@author: robin
"""
import os
import pandas as pd
os.chdir('C:/Users/robin/Documents/GitHub/SMC-ABC') # Change path to run it on your computer

from functions_oop import *
 
import numpy as np
import pickle
#from functions import *

N = 1000
alpha = 0.9
e = 0.0025
pop_size= 1000 #pop_size= 473
# while 20 fois

actual_data = np.array([30,23,15,10,8,5,4,3,2,1]), np.array([1,1,1,1,1,2,4,13,20,282]) # Data described in the paper end of p8
smc_abc = SMC_ABC(actual_data, N, e, pop_size, alpha)
smc_abc.sampler(True)

output = smc_abc.output

with open('smc_abc_'+str(N)+'_'+str(e)+'_'+ str(pop_size), 'wb') as fichier:
    mon_pickler = pickle.Pickler(fichier)
    mon_pickler.dump(output)
  
    
    
#=========================================================================================================
# Analysis
#=========================================================================================================
with open('smc_abc_'+str(N)+'_'+str(e)+'_'+ str(pop_size), 'rb') as fichier:
        mon_depickler = pickle.Unpickler(fichier)
        output_depickled = mon_depickler.load()

smc_abc.pop_size
thetas = np.concatenate(np.array(output_depickled[0]))
phi = thetas[:,0]
tau =  thetas[:,1]
xi =  thetas[:,2]

pd.Series(phi).plot(kind='density')
pd.Series(tau).plot(kind='density')
pd.Series(xi).plot(kind='density')

acc_rate = phi - tau
pd.Series(acc_rate).plot(kind='density')

log2_over_acc_rate = np.log(2)/acc_rate
pd.Series(log2_over_acc_rate).plot(kind='density')

phi.mean()
tau.mean()
xi.mean()

ess = output[1]
pd.Series(ess).plot()

epsilon = output[2]
pd.Series(epsilon[1:]).plot() 

#=========================================================================================================
# Posterior mean
#=========================================================================================================

N = 50 
alpha = 0.9
e = 0.0025
pop_size= 473

posterior_means = []
for i in range(20):
    print('--------------------------------------------------------------')
    print('iteration =', i)
    try:
        smc_abc = SMC_ABC(actual_data, N, e, pop_size, alpha)
        smc_abc.sampler(True)
        thetas = np.concatenate(np.array(smc_abc.output[0]))
        phi = thetas[:,0]
        tau =  thetas[:,1]
        xi =  thetas[:,2]
        posterior_means.append(np.array([phi.mean(), tau.mean(),xi.mean()]))
    except RuntimeError:
        print(RuntimeError)
        pass



post_means = np.stack(np.array(posterior_means))
phi_mean = post_means[:,0]
tau_mean =  post_means[:,1]
xi_mean =  post_means[:,2]

pd.Series(phi_mean).plot(kind='density')
pd.Series(tau_mean).plot(kind='density')
pd.Series(xi_mean).plot(kind='density')

