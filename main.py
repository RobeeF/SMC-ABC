# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 13:26:04 2018

@author: robin
"""
import os
os.chdir('C:/Users/robin/Documents/GitHub/SMC-ABC') # Change path to run it on your computer
 
import numpy as np
from functions import *


N = 10**3 # Typically 10**4 in the paper but relatively long to run, change it for final computation
population = simulate_population(N) # Simulate the population 
n = 473 # To match the size of the sample described in the paper

# Compute eta on a sample from the simulated population
sample = np.random.choice(population, n, replace=False) # Draw a sample without replacement
eta_pop_sample = compute_eta(sample)
print(eta_pop_sample)

# Compute eta from the actual data
actual_data = np.array([30,23,15,10,8,5,4,3,2,1]), np.array([1,1,1,1,1,2,4,13,20,282]) # Data described in the paper end of p8
actual_sample = generate_sample_from_clusters(actual_data[0],actual_data[1])
eta_actual_data = compute_eta(actual_sample)
print(eta_actual_data)

# Compute the difference between the 2
compute_rho(eta_pop_sample,eta_actual_data,n)
