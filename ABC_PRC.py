import numpy as np
import scipy.stats as sc
import seaborn as sns
import matplotlib.pyplot as plt



def ABC_PRC (D, x0, f, prior, nu, epsilon_list, E, K_list, L_list, eta, rho, N, verbose = False) :
    '''
    Sisson's Approximate Bayesian Computation with Partial Rejection control
    The algorithm generate N particules that approximate the posterior distribution
    
    -D :                dimension of the parameter theta                    (int)
    -x0 :               the observed data                                   (list of floats)
    -f :                the likelihood function                             (function)
    -prior :            prior distribution of theta                         (function)
    -nu :               initial sampling distribution                       (function)
    -espilon_list :     list of tolerances                                  (list of floats)
    -E :                resampling threshold, commonly N/2                  (float)
    -K_list :           list of Markov transition kernel                    (list of functions)
    -L_list :           list of backward transition kernel                  (list of functions)
    -eta :              function of a mesure of a sample                    (function)
    -rho :              distance function between two mesures of sample     (function)
    -N :                number of particules to generate                    (int)
    
    Note : espilon_list, K_list, L_list should have the same length
    
    '''
    
    #PRC1
    T = len(epsilon_list)
    B = len(x0)
    
    theta = np.zeros((N,T,D))
    theta1 = np.zeros(D)
    theta2 = np.zeros(D)
    
    x = np.zeros((B,D))
    
    weight = np.zeros((N,T))
    weight[:,0] = np.array([1/N]*N)
    
    t = 0
    i = 0
    
    index = [n for n in range(N)]
    
    if verbose :
        Number_gen = np.zeros(T)
    
    while True :
        while True :
            #PRC2
            while True :

                #PRC2.1
                if t == 0 :
                    theta2 = nu()
                    
                else :
                    choice = int(np.random.choice(index, 1, p = weight[:,t-1]))
                    theta1 = theta[choice,t-1,:]                    
                    theta2 = K_list[t](theta1)
                
                
                x = f(B, theta2)
                
                if verbose :
                    print("======")
                    print(str(t) + " " + str(i))
                    print(theta2)
                    Number_gen[t] = Number_gen[t] + 1
                    print(Number_gen[t])
                    
                
                if rho(eta(x), eta(x0)) < epsilon_list[t] :
                    break
            
            #PRC2.2
            theta[i,t] = theta2
            
            if t == 1 :
                if nu(theta[i,t,:]) == 0:
                    weight[i,t] = 0
                else :
                    weight[i,t] = prior(theta[i,t,:])/nu(theta[i,t,:])
                
            else :
                Lt = L_list[t-1](theta1, theta[i,t,:])
                Kt = K_list[t](theta[i,t,:], theta1)
                
                weight[i,t] = (prior(theta[i,t,:])*Lt)/(prior(theta1)*Kt)
                
            i = i + 1
            
            if i >= N :
                i = 0
                break
        
        #PRC3
        weight[:,t] = weight[:,t]/np.sum(weight[:,t])
        
        ess = np.sum(np.array(list(map(lambda x : x**2, weight))))
        ess = 1/ess
        
        if ess < E :
            choice = np.random.choice(index, N, p = weight[:,t])
            theta[:,t,:] = theta[choice,t,:]
            weight[:,t] = np.array([1/N]*N)
        
        #PRC4
        t = t + 1
        
        if t >= T :
            break
    
    if verbose :
        return(theta, Number_gen)
    
    return(theta)




### Toy example ###
    
N = 1000
B = 100

def f (N, theta) :
    return (np.random.normal(float(theta), 1, N))

x0 = f(B, 0)

def prior (theta = None) :
    if theta is None :
        return (float(np.random.uniform(-10,10, 1)))
    
    return (sc.uniform(-10,20).pdf(float(theta)))

nu = prior

epsilon_list = [2, 0.5, 0.025]

E = N/2

def K (theta1, theta2 = None) :
    
    if theta2 is None :
        return (np.random.normal(float(theta1), 1, 1))    
    
    return (sc.norm(float(theta1), 1).pdf(float(theta2)))

K_list = [K]*3
L_list = K_list

def eta(x) :
    return (x)

def rho(eta1, eta2) : 
    coin = np.random.randint(0,2)
    if coin == 0 :
        return (abs(eta1[0]))
    
    return (abs(np.mean(eta1)))

theta, Number_gen = ABC_PRC(1, x0, f, prior, nu, epsilon_list, E, K_list, L_list, eta, rho, N, verbose = True)



line = np.arange(-4, 4, 0.01)

#Population 1
sns.distplot(theta[:,0], kde=False, norm_hist = True)
plt.plot(line, sc.norm(0, 0.1).pdf(line)/2 + sc.norm(0, 1).pdf(line)/2)


#Population 2
sns.distplot(theta[:,1], kde=False, norm_hist = True)
plt.plot(line, sc.norm(0, 0.1).pdf(line)/2 + sc.norm(0, 1).pdf(line)/2)


#Population 3
sns.distplot(theta[:,2], kde=False, norm_hist = True)
plt.plot(line, sc.norm(0, 0.1).pdf(line)/2 + sc.norm(0, 1).pdf(line)/2)



#Number of generation at each step
print(Number_gen[0])
print(Number_gen[1])
print(Number_gen[2])




### Tuberculosis Example ###


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

  
def simulate_population(N, phi, tau, xi, verbose=True):
    ''' Simulate the population as presented in the paper 
    N: (int) number of people in the population to simulate
    verbose: if True then the algorithm prints intermediate results
    
    returns: (array-like) sample of size N, the values in the array are the type of each individual belonging in the population '''    
    event_proba = np.array([phi, tau, xi])
    X,G = np.ones(1),1 
    
    if (event_proba < 0).any() :
        return generate_sample_from_clusters(np.array([1]),np.array([N]))

    fails = 0
    t=1 
    
    while X.sum()<N: 
        if X.sum()>0:  
            t+=1
            event_picked = np.random.multinomial(n=1, pvals=event_proba/event_proba.sum())
            type_picked = np.random.multinomial(n=1, pvals=X/X.sum()) # Pas bon besoin d'une conversion
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
            
            #If it fails too much, stop the generation
            if fails > 5 :
                return generate_sample_from_clusters(np.array([1]),np.array([N]))
            
    # Turn X into a sample of pop = X$
    Gi, Ni = np.unique(X, return_counts=True) # Count size of each group and how many groups have this size 
    if Gi[0]==0: # We don't care about the types that have 0 people
        Gi,Ni = Gi[1:], Ni[1:]
    
    return generate_sample_from_clusters(Ni.astype(int),Gi.astype(int))


N = 1000
B = 473

def f (N, parameter) :
    alpha, delta, theta = parameter[0], parameter[1], parameter[2]
    return (simulate_population(N, alpha, delta, theta, verbose = False))

actual_data = np.array([30,23,15,10,8,5,4,3,2,1]), np.array([1,1,1,1,1,2,4,13,20,282]) # Data described in the paper end of p8
x0 = generate_sample_from_clusters(actual_data[0],actual_data[1])

def prior (parameter = None) :
    
    if parameter is None  :
    
        alpha = np.random.gamma(1, 0.1)
        delta = np.random.uniform(0, alpha)
        theta = sc.truncnorm.rvs(a=0, b=10*15, loc=0.198, scale=0.067352)
        
        return(np.array([alpha, delta, theta]))
        
    alpha, delta, theta = parameter[0], parameter[1], parameter[2]
        
    if 0 < delta and delta < alpha  :
        return (sc.norm(0.198, 0.06735**2).pdf(theta))
    
    return (0)
    


nu = prior

def compute_epsilon(T, ini, last) :
    espilon = [ini]
    
    if(T > 2) :
        for t in range(1,T-1) :
            espilon.append((espilon[t-1] + last)/2)
            #And not (3*espilon[t-1] - espilon[last])/2
        
    espilon.append(last)
    
    return (espilon)

epsilon_list = compute_epsilon(10, 1, 0.0025)

E = N/2

def K (theta1, theta2 = None) :
    
    sigma = np.array([[0.5**2,0.225,0],[0.225,0.5**2,0],[0,0,0.015**2]])
    
    if theta2 is None :
                
        return ((np.random.multivariate_normal(theta1, sigma, 1)).flatten())
    
    return (sc.multivariate_normal(theta1, sigma).pdf(theta2))
        


K_list = [K]*10
L_list = K_list

def compute_eta(sample):
    gi, ni = np.unique(sample, return_counts=True) 
    g = len(ni)
    n = np.sum(ni)
    return (g,1-np.square(ni).sum()/n**2)

def compute_rho(eta,eta_bar,n):
    return (abs(eta[0] - eta_bar[0]) + abs(eta[1] - eta_bar[1]))/n

eta = compute_eta

def rho(eta1, eta2) :
    return(compute_rho(eta1, eta2, n = B))


theta, Number_gen = ABC_PRC(3, x0, f, prior, nu, epsilon_list, E, K_list, L_list, eta, rho, N, verbose = True)
