# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 11:13:57 2018

@author: robin
"""

import numpy as np
from numpy.random import multinomial
from scipy.stats import truncnorm, gamma, uniform
from particles import resampling as rs
import copy

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


def compute_prior():
    # Maybe the definition of gamma is wrong the paper seems to inverse a and b in the definition of the mean of the gamma
    # Put mean of the gamma to 0.5 instead of 0.1 because too long to run otherwise
    phi, xi = gamma.rvs(a=0.5, size=1)[0],truncnorm.rvs(a=0, b=10**10, loc=0.198, scale=0.067352)
    tau = uniform.rvs(loc=0,scale=phi)
    return np.array([phi, tau, xi])


test = compute_prior()
gamma.pdf(x=test[0],a=0.1)*truncnorm.rvs(a=0, b=10**10, loc=0.198, scale=0.067352)*uniform.rvs(loc=0,scale=test[0])

  
def simulate_population(N, prior ,verbose=True):
    ''' Simulate the population as presented in the paper 
    N: (int) size of the particle to simulate
    prior: (array-like) the prior drawn from compute prior
    verbose: if True then the algorithm prints intermediate results
    
    returns: (array-like) particle of size n, the values in the array are the type of each individual belonging in the population '''    

    X,G = np.ones(1),1 
    
    fails = 0
    t=1 
    
    while X.sum()<N: 
        if X.sum()>0:  
            t+=1
            event_picked = multinomial(n=1, pvals=prior/prior.sum())
            type_picked = multinomial(n=1, pvals=X/X.sum()) # Pas propre besoin d'une conversion
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


#==========================================================================================================
# SMC Sampler
#==========================================================================================================
def compute_ess(Wn): 
    if Wn.sum()==0:
        return 1
    else:
        return (np.square(Wn).sum())**(-1)

def compute_Wn(prev_weights, prev_epsilon, epsilon, actual_sample_eta, sample_size, gen_sample_etas):
    new_weights =  prev_weights*np.array([int(compute_rho(eta,actual_sample_eta,sample_size)<epsilon) for eta in gen_sample_etas])/np.array([int(compute_rho(eta,actual_sample_eta,sample_size)<prev_epsilon) for eta in gen_sample_etas])
    new_weights = np.nan_to_num(new_weights) # For the weights divided by 0
    return new_weights

def find_epsilon_n(e,alpha, prev_epsilon, prev_weights, sample_size, gen_sample_etas, actual_sample_eta, prev_ess):
    ''' Returns the next epsilon such that ESS(new_epsilon) is the closest to ESS(prev_epsilon)
    '''
    new_epsilons = np.linspace(e,prev_epsilon,10)
    new_epsilon_ess_dict = {}

    for epsilon in new_epsilons:
        new_weights = compute_Wn(prev_weights, prev_epsilon, epsilon, actual_sample_eta, sample_size, gen_sample_etas)
        new_epsilon_ess_dict[epsilon] = compute_ess(new_weights)
        
    dist_to_alpha_ess = {k: v - alpha*prev_ess for k, v in new_epsilon_ess_dict.items()}
    best_epsilon = max(dist_to_alpha_ess, key=dist_to_alpha_ess.get)
    return best_epsilon

def MHRW(theta, cov, epsilon, pop_size, sample_size, prev_gen_sample ,prev_gen_eta, actual_sample_eta):
    ''' Return the Metropolis Hastings Random Walk Kernel updates of the thetas=(phi, eta, tau) '''
    # Need to reparametrize theta over the real line ? 
    new_theta = theta + np.random.multivariate_normal([0,0,0], 2*cov) # Pas sûr que ce soit le bon proposal
    new_theta = new_theta/new_theta.sum() # reparametrization on the real line. Is it the right way to do it ? 
    
    new_pop = simulate_population(pop_size, theta, False)
    new_sample = np.random.choice(new_pop, sample_size, replace=True)
    new_gen_eta = compute_eta(new_sample)
    
    indicatrices_ratio = int(compute_rho(new_gen_eta,actual_sample_eta,sample_size)<epsilon)/int(compute_rho(prev_gen_eta,actual_sample_eta,sample_size)<epsilon)
    proposal_ratio = 1 # random walk proposal is symetric then the ratio of q(theta*,theta)/q(theta,theta*)=1
    new_prior = gamma.pdf(x=new_theta[0],a=0.1)*truncnorm.pdf(x=new_theta[2],a=0, b=10**10, loc=0.198, scale=0.067352)*uniform.pdf(x=new_theta[1],loc=0,scale=theta[0])
    old_prior = gamma.pdf(x=theta[0],a=0.1)*truncnorm.pdf(x=theta[2],a=0, b=10**10, loc=0.198, scale=0.067352)*uniform.pdf(x=theta[1],loc=0,scale=theta[0])
    
    acceptance_proba = min(1,indicatrices_ratio*proposal_ratio*new_prior/old_prior)
    unif_draw = uniform.rvs(size=1)[0]

    if unif_draw<=acceptance_proba:
        return new_theta, new_sample
    else:
        return theta, prev_gen_sample


def SMC_sampler():
    # Genereate the actual sample
    actual_data = np.array([30,23,15,10,8,5,4,3,2,1]), np.array([1,1,1,1,1,2,4,13,20,282]) # Data described in the paper end of p8
    actual_sample = generate_sample_from_clusters(actual_data[0],actual_data[1])
    eta_actual_data = compute_eta(actual_sample)
    
    N = 5 # To match the size of the sample described in the paper should =1000
    N_T = N/2
    alpha=0.9
    #pop = simulate_population(N, verbose=False) # Simulate the population 
    e = 0.00045
    pop_size = 300 # Should equal 10000
    sample_size = 100 #Should equal 473
        
    epsilon = []
    ess = [] 
    weights = {'n-1':0,'n':0}
    thetas = []
    X = [] # Each element of this array is the N samples at time n
  
    # Step 0 
    n=0
    T = 10 # Should equal 100 to match the paper
    
    epsilon.append(10**6) # Set epsilonO to a big number
    ess.append(N) # Append ESSo
    weights['n-1'] = np.full(N,1/N) # Append Wo
    thetas.append(np.array([compute_prior() for i in range(N)]))
    pops = np.array([simulate_population(pop_size, theta, True) for theta in thetas[n-1]])
    
    X.append(np.array([np.random.choice(pop, sample_size, replace=False) for pop in pops])) # Draw a sample without replacement
    etas = np.array([compute_eta(pop) for pop in X[0]])
    
    # Step 1
    while (n<T) and (epsilon[n-1]>= e):
        n+=1
        print(n)
        epsilon.append(find_epsilon_n(e,alpha, epsilon[n-1], weights['n-1'], sample_size, etas, eta_actual_data, ess[n-1])) # Solve ESS = alpha*ESS
        weights['n'] = compute_Wn(weights['n-1'],epsilon[n-1],epsilon[n], eta_actual_data, sample_size, etas)
        ess.append(compute_ess(weights['n']))
            
        # Step 2
        if ess[n]<N_T:
            print('resampling')
            indices = rs.systematic(weights['n'])
            thetas[n-1] = copy.deepcopy(thetas[n-1][indices])
            X[n-1] = copy.deepcopy(X[n-1][indices])
            weights['n-1'] = np.full(N,1/N)
                
        # Step 3
        var_theta = np.cov(np.array(thetas[n-1]), rowvar=False)
        new_Z = np.array([MHRW(thetas[n-1][i], var_theta, epsilon[n-1], pop_size, sample_size, X[n-1][i], etas[i], eta_actual_data) for i in range(N) if weights['n'][i]>0])
        thetas.append(copy.deepcopy(np.stack(new_Z[:,0], axis=0))) 
        X.append(copy.deepcopy(np.stack(new_Z[:,1], axis=0)))
        # Prepare the weights to store the next iteration (better in head of the loop ?) 
        weights['n-1'] = weights['n']
        
   
    len(thetas)
    len(X)
    len(ess)
    len(epsilon)
    

# Non-vectorized functions  
N = 100
priors = np.array([compute_prior() for i in range(10)])
pops = np.array([simulate_population(1000, prior, False) for prior in priors])


#simulate_population(N, priors[0])
veq_simul_pops = np.vectorize(simulate_population, excluded=['N','verbose'])
a = veq_simul_pops(prior=priors, N=100, verbose=False)

