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

N = 50
alpha = 0.9
e = 0.01
pop_size=10000

actual_data = np.array([30,23,15,10,8,5,4,3,2,1]), np.array([1,1,1,1,1,2,4,13,20,282]) # Data described in the paper end of p8
smc_abc = SMC_ABC(actual_data, N, e, pop_size, alpha)
smc_abc.sampler(True)

output = smc_abc.output

thetas = np.concatenate(np.array(output[0]))
phi = thetas[:,0]
tau =  thetas[:,1]
xi =  thetas[:,2]

pd.Series(phi).plot(kind='density')
pd.Series(tau).plot(kind='density')
pd.Series(xi).plot(kind='density')


ess = output[1]
pd.Series(ess).plot()

epsilon = output[2]
pd.Series(epsilon).plot() 

w = np.ones(4)
a = np.full(10,0)
np.concatenate(w,a, axis=1)

