# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 13:31:14 2019

@author: robin
"""

import numpy as np
from numpy.random import multinomial
from scipy.stats import truncnorm, gamma, uniform
from particles import resampling as rs
import copy
from numba import jit, prange


class SMC_ABC(object):
    def __init__(self, data, N, e, pop_size, T, alpha):
        if N<2:
            raise ValueError('SMC_ABC sampler works wth at least 2 particules') # Right error type ?
        self.data = self.generate_sample_from_clusters(data[0], data[1])
        self.actual_eta = self.compute_eta(self.data)
        self.N = N
        self.priors = self.compute_prior()
        self.e = e
        self.pop_size = pop_size
        self.sample_size = len(self.data)
        self.N_T = N/2
        self.T = T
        self.alpha = alpha

    @jit(parallel=True)
    def generate_sample_from_clusters(self, clusters_sizes, nb_of_clusters_of_that_size): # need to shuffle ?
        '''Generates a sample from the types in the population and the number of people that are of that type 
        clusters_sizes: (array-like)  How many people in each cluster (a cluster = people with the same type)
        nb_of_clusters_of_that_size: (array-like) How many clusters have [clusters_sizes] people in it'''    
        # Initialisation
        current_group_index = 0 # Assign numbers to each cluster 
        sample = [] # Store the samples generated
        
        for cluster_size, nb_clusters in zip(clusters_sizes,nb_of_clusters_of_that_size):
            for i in prange(current_group_index, current_group_index+nb_clusters):
                sample+= np.full(shape=(cluster_size,),fill_value=i).tolist()
            current_group_index+=nb_clusters
        return np.array(sample)
    
    
    def compute_eta(self, pop):
        ''' Compute eta p9 of the paper'''
        gi, ni = np.unique(pop, return_counts=True) 
        g = len(ni)
        n = np.sum(ni)
        return (g,1-np.square(ni).sum()/n**2)
    
    def compute_rho(self, eta):
        ''' Compute rho as in the paper 
        eta (tuple of size 2): a mesure performed on a sample
        eta (tuple of size 2): a mesure performed on another sample
        n: size of the samples used to compute the etas
        returns: (float) rho
        '''
        return (abs(eta[0] - self.actual_eta[0]) + abs(eta[1] - self.actual_eta[1]))/self.sample_size
    
    
    def compute_prior(self):
        # Maybe the definition of gamma is wrong the paper seems to inverse a and b in the definition of the mean of the gamma
        # Put mean of the gamma to 0.5 instead of 0.1 because too long to run otherwise
        phi, xi = gamma.rvs(a=0.5, size=self.N),truncnorm.rvs(a=0, b=10**10, loc=0.198, scale=0.067352, size=self.N)
        tau = uniform.rvs(loc=0,scale=phi, size=self.N)
        theta = np.dstack((phi,tau,xi))[0]
        return theta
    
    @jit(parallel=True)
    def samples_and_etas_from_pop(self, priors):
        samples = []
        etas = []
        for theta in priors:
            pop = self.simulate_population(theta, True)
            samples.append(np.random.choice(pop, self.sample_size, replace=False))
            etas.append(self.compute_eta(pop))
            
        return samples, etas
    
    @jit
    def simulate_population(self, theta, verbose=True):
        ''' Simulate the population as presented in the paper 
        N: (int) size of the particle to simulate
        theta: (array-like) the theta parameter (improve def) 
        verbose: if True then the algorithm prints intermediate results
        
        returns: (array-like) particle of size n, the values in the array are the type of each individual belonging in the population '''    

        X = np.ones(1, 'int')
        G = 1 

        fails = 0
        t=1 
        
        while X.sum()<self.pop_size: 
            if X.sum()>0:  
                t+=1
                event_picked = multinomial(n=1, pvals=theta/theta.sum())
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
        
        return self.generate_sample_from_clusters(Ni.astype(int),Gi.astype(int))
    
    
    #==========================================================================================================
    # SMC Sampler
    #==========================================================================================================
    def compute_ess(self, Wn): 
        return (np.square(Wn).sum())**(-1)
    
    def compute_Wn(self, prev_weights, prev_epsilon, current_epsilon, gen_sample_etas):
        new_weights =  prev_weights*np.array([int(self.compute_rho(eta)<current_epsilon) for eta in gen_sample_etas])/np.array([int(self.compute_rho(eta)<prev_epsilon) for eta in gen_sample_etas])
        new_weights = np.nan_to_num(new_weights) # For the weights divided by 0
        new_weights = new_weights/new_weights.sum() if new_weights.sum()!=0 else new_weights# Renomalise the weights 
        return new_weights
    
    #@jit(parallel=True)
    def find_epsilon_n(self, prev_epsilon, prev_weights, gen_sample_etas, prev_ess):
        ''' Returns the next epsilon such that ESS(new_epsilon) is the closest to ESS(prev_epsilon)
        '''
        SIZE_GRID_EPSILON = 10
        new_epsilons = np.linspace(self.e,prev_epsilon,5 + np.ceil(SIZE_GRID_EPSILON/prev_epsilon))
        new_epsilon_ess_dict = {}
        for epsilon in new_epsilons:
            new_weights = self.compute_Wn(prev_weights, prev_epsilon, epsilon, gen_sample_etas)
            if new_weights.sum()!=0:
                new_epsilon_ess_dict[epsilon] = self.compute_ess(new_weights)
         
        if len(new_epsilon_ess_dict.keys()) == 0:
            raise RuntimeError('Could not find the next epsilon value')
            
        dist_to_alpha_ess = {k: abs(v - self.alpha*prev_ess) for k, v in new_epsilon_ess_dict.items()}
        min_value = min(dist_to_alpha_ess.values())
        best_epsilon = min([key for key,val in dist_to_alpha_ess.items() if val == min_value])
        return best_epsilon
        
    def indiv_MHRW(self, theta, cov, epsilon, prev_gen_sample ,prev_gen_eta):
        ''' Return the Metropolis Hastings Random Walk Kernel updates of the thetas=(phi, eta, tau) '''
        # Need to reparametrize theta over the real line ? 
        new_theta = theta + np.random.multivariate_normal([0,0,0], 2*cov) # Pas sûr que ce soit le bon proposal
        new_theta = new_theta/new_theta.sum() # reparametrization on the real line. Is it the right way to do it ? 
        
        new_pop = self.simulate_population(theta, False)
        new_sample = np.random.choice(new_pop, self.sample_size, replace=True)
        new_gen_eta = self.compute_eta(new_sample)
        
        indicatrices_ratio = int(self.compute_rho(new_gen_eta)<epsilon)/int(self.compute_rho(prev_gen_eta)<epsilon)
        proposal_ratio = 1 # random walk proposal is symetric then the ratio of q(theta*,theta)/q(theta,theta*)=1
        new_priors = np.array([gamma.pdf(x=new_theta[0],a=0.1),uniform.pdf(x=new_theta[1],loc=0,scale=theta[0]),truncnorm.pdf(x=new_theta[2],a=0, b=10**10, loc=0.198, scale=0.067352)])
        old_priors = np.array([gamma.pdf(x=theta[0],a=0.1),uniform.pdf(x=theta[1],loc=0,scale=theta[0]),truncnorm.pdf(x=theta[2],a=0, b=10**10, loc=0.198, scale=0.067352)])
    
        acceptance_probas = np.minimum(np.ones(len(theta)),indicatrices_ratio*proposal_ratio*new_priors/old_priors)
        unif_draws = uniform.rvs(size=len(theta))
        
        new_theta_accepted = unif_draws<=acceptance_probas
        final_theta = np.where(new_theta_accepted,new_theta, theta)

        return final_theta, acceptance_probas # Check if final_theta<1 for all theta
    
    @jit
    def sampler(self, full_output=False):
        
        epsilon = []
        ess = [] 
        weights = []
        thetas = []
        X = [] # Each element of this array is the N samples at time n
        acceptance_probas = []
        samples = []
        etas = []
        
        # Step 0 
        n=0        
        epsilon.append(10**6) # Set epsilonO to a big number
        ess.append(self.N) # Append ESSo
        weights.append(np.full(self.N,1/self.N)) # Append Wo
        thetas.append(self.priors)
        
        samples, etas = self.samples_and_etas_from_pop(thetas[0])

        X.append(samples)

        
        # Step 1
        while (n<self.T) and (epsilon[n-1]>= self.e):
            n+=1
            print(n)

            epsilon.append(self.find_epsilon_n(epsilon[n-1], weights[n-1], etas, ess[n-1])) # Solve ESS = alpha*ESS
            weights.append(self.compute_Wn(weights[n-1],epsilon[n-1],epsilon[n], etas))
            ess.append(self.compute_ess(weights[n]))
            
            # Step 2
            if ess[n]<self.N_T:
                print('resampling')
                indices = rs.systematic(weights[n])
                thetas[n-1] = copy.deepcopy(thetas[n-1][indices])
                X[n-1] = copy.deepcopy(X[n-1][indices])
                weights[n] = np.full(self.N,1/self.N)
                    
            # Step 3
            print('Kernel update')
            cov_theta = np.cov(np.array(thetas[n-1]), rowvar=False)            
            thetas_from_MHRW = []
            ac_proba_from_MHRW = []
            for i in prange(self.N):
                if weights[n][i]>0:
                    MHRW_results = self.indiv_MHRW(thetas[n-1][i], cov_theta, epsilon[n-1], X[n-1][i], etas[i])
                    thetas_from_MHRW.append(np.stack(MHRW_results[0]))
                    ac_proba_from_MHRW.append(MHRW_results[1])
            
            samples, etas = self.samples_and_etas_from_pop(thetas_from_MHRW)
                        
            thetas.append(copy.deepcopy(np.stack(np.array(thetas_from_MHRW))))
            X.append(copy.deepcopy(np.stack(np.array(samples))))
            acceptance_probas.append(copy.deepcopy(np.stack(np.array(ac_proba_from_MHRW))))
            etas = np.array(etas)

        if full_output: 
            self.output = np.stack(thetas)[1:], np.stack(acceptance_probas), ess, epsilon
            return np.stack(thetas)[1:], np.stack(acceptance_probas), ess, epsilon
        
        else:
            self.output = np.stack(thetas)[1:]
            return np.stack(thetas)[1:] # [1:] to not return the prior draw of n=0

