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
#from functions import *

N = 3
alpha = 0.9
e = 0.0045
pop_size=1000
T=10

actual_data = np.array([30,23,15,10,8,5,4,3,2,1]), np.array([1,1,1,1,1,2,4,13,20,282]) # Data described in the paper end of p8
smc_abc = SMC_ABC(actual_data, N, e, pop_size, T, alpha)
smc_abc.sampler(True)

output = smc_abc.output

phi = np.concatenate(output[0][:,:,0].reshape(-1,1))
tau =  np.concatenate(output[0][:,:,1].reshape(-1,1))
xi =  np.concatenate(output[0][:,:,2].reshape(-1,1))

pd.Series(phi).plot(kind='density')
pd.Series(tau).plot(kind='density')
pd.Series(xi).plot(kind='density')


phi_acceptance = output[1][:,:,0].mean(axis=1)
tau_acceptance = output[1][:,:,1].mean(axis=1)
xi_acceptance = output[1][:,:,2].mean(axis=1)

pd.Series(phi_acceptance).plot()
pd.Series(tau_acceptance).plot()
pd.Series(xi_acceptance).plot()

ess = output[2]
pd.Series(ess).plot()

epsilon = output[3]
pd.Series(epsilon).plot() 