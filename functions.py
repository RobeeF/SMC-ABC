# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 11:13:57 2018

@author: robin
"""

import numpy as np
from numpy.random import multinomial, gamma
from scipy.stats import truncnorm
from numpy.random import uniform

def generate_sample_from_clusters(clusters_sizes,nb_of_clusters_of_that_size): # need to shuffle ?
    '''Generates a sample from the types in the population and the number of people that are of that type 
    clusters_sizes: (array-like)  How many people in each cluster (a cluster = people with the same type)
    nb_of_clusters_of_that_size: (array-like) How many clusters have [clusters_sizes] people in it
    '''    
    # Initialisation
    current_group_index = 0 # Assign numbers to each cluster 
    sample = [] # Store the samples generated
    
    for cluster_size, nb_clusters in zip(clusters_sizes,nb_of_clusters_of_that_size):
        for i in range(current_group_index, current_group_index+nb_clusters):
            sample+= np.full(shape=(cluster_size,),fill_value=i).tolist()
        current_group_index+=nb_clusters
    return np.array(sample)


def compute_eta(sample):
    ''' Compute eta p9 of the paper'''
    gi, ni = np.unique(sample, return_counts=True) 
    g = len(ni)
    n = np.sum(ni)
    return (g,1-np.square(ni).sum()/n**2)

def compute_rho(eta,eta_bar,n):
    ''' Compute rho as in the paper 
    eta (tuple of size 2): a mesure performed on a sample
    eta (tuple of size 2): a mesure performed on another sample
    n: size of the samples used to compute the etas
    returns: (float) rho
    '''
    return (abs(eta[0] - eta_bar[0]) + abs(eta[1] - eta_bar[1]))/n
    
  
def simulate_population(N, verbose=True):
    ''' Simulate the population as presented in the paper 
    N: (int) number of people in the population to simulate
    verbose: if True then the algorithm prints intermediate results
    
    returns: (array-like) sample of size N, the values in the array are the type of each individual belonging in the population '''    
    phi, xi = gamma(1, 0.1, 1)[0],truncnorm.rvs(a=0, b=10*15, loc=0.198, scale=0.067352)
    tau = uniform(0,phi)
    
    event_proba = np.array([phi, tau, xi])
    X,G = np.ones(1),1 
    
    fails = 0
    t=1 
    
    while X.sum()<N: 
        if X.sum()>0:  
            t+=1
            event_picked = multinomial(n=1, pvals=event_proba/event_proba.sum())
            type_picked = multinomial(n=1, pvals=X/X.sum()) # Pas bon besoin d'une conversion
            type_picked_index = np.where(type_picked==1)[0][0]
            
            if (event_picked == np.array([1,0,0])).all():
                X[type_picked_index] += 1 
                        
            elif (event_picked == np.array([0,1,0])).all(): # Death
                X[type_picked_index] -= 1 
        
            else: # Mutation
                X[type_picked_index] -= 1 
                X =np.append(X, 1)
                G+=1

            if verbose:
                print(X.sum())
        else: # If all types have died we relaunch the algorithm (as prescribed in Tanaka (2006) p4)
            t= 1
            X,G = np.ones(1),1
            fails+=1
            if verbose:
                print('The generation of the population has failed ', fails,' times')
    
    # Turn X into a sample of pop = X$
    Gi, Ni = np.unique(X, return_counts=True) # Count size of each group and how many groups have this size 
    if Gi[0]==0: # We don't care about the types that have 0 people
        Gi,Ni = Gi[1:], Ni[1:]
    
    return generate_sample_from_clusters(Ni.astype(int),Gi.astype(int))
