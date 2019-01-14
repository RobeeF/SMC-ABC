import numpy as np
import scipy.stats as sc
import seaborn as sns
import matplotlib.pyplot as plt
import math


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
        ess_list = []
        count = 0
    
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
                    
                    if count == 10 :
                        ess = np.sum(np.array(list(map(lambda x : x**2, weight[:,t]/np.sum(weight[:,t])))))
                        ess = 1/ess
                        ess_list.append(ess)
                        count = 0
                        
                    else :
                        count = count + 1
                    
                
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
        
        ess = np.sum(np.array(list(map(lambda x : x**2, weight[:,t]))))
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
        return(theta, Number_gen, ess_list)
    
    return(theta)




### Toy example ###
    
##Define the arguments
    
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


## Launch the simlation

theta, Number_gen, ess_list = ABC_PRC(1, x0, f, prior, nu, epsilon_list, E, K_list, L_list, eta, rho, N, verbose = True)


## Get some graphes

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

plt.plot(ess_list)


### Tuberculosis Example ###

##Define the arguments
   
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

  
def simulate_population(pop_size, theta, verbose=True):
    ''' Simulate the population as presented in the paper 
    N: (int) number of people in the population to simulate
    verbose: if True then the algorithm prints intermediate results
    
    returns: (array-like) sample of size N, the values in the array are the type of each individual belonging in the population '''
    if (theta < 0).any() :
        return generate_sample_from_clusters(np.array([1]),np.array([N]))

    X = np.ones(1, 'int')
    G = 1 

    t=1
    fail_to_gen_pop = False
    
    while X.sum() < pop_size: 
        if verbose:
            print(X.sum())
        t+=1
        event_picked = np.random.multinomial(n=1, pvals=theta/theta.sum())
        type_picked = np.random.multinomial(n=1, pvals=X/X.sum()) # Pas propre besoin d'une conversion
        type_picked_index = np.where(type_picked==1)[0][0]
        
        if (event_picked == np.array([1,0,0])).all():
            X[type_picked_index] += 1 
                    
        elif (event_picked == np.array([0,1,0])).all(): # Death
            X[type_picked_index] -= 1 
    
        else: # Mutation
            X[type_picked_index] -= 1 
            X =np.append(X, 1)
            G+=1
            
        if X.sum()==0 or t>=7*pop_size:
            fail_to_gen_pop=True
            break
        
    if fail_to_gen_pop:
        return generate_sample_from_clusters(np.array([1]),np.array([N]))
    else:
        # Turn X into a sample of pop = X$
        Gi, Ni = np.unique(X, return_counts=True) # Count size of each group and how many groups have this size 
        if Gi[0]==0: # We don't care about the types that have 0 people
            Gi,Ni = Gi[1:], Ni[1:]
        return generate_sample_from_clusters(Ni.astype(int),Gi.astype(int))


N = 1000
B = 473

def f (N, theta, pop_size = 473) :
    return simulate_population(pop_size, theta, verbose = False)

actual_data = np.array([30,23,15,10,8,5,4,3,2,1]), np.array([1,1,1,1,1,2,4,13,20,282]) # Data described in the paper end of p8
x0 = generate_sample_from_clusters(actual_data[0],actual_data[1])

def prior (parameter = None) :
    
    if parameter is None  :
    
        phi = np.random.gamma(0.1, 1)
        tau = np.random.uniform(0, phi)
        xi = sc.truncnorm.rvs(a=0, b=10*10, loc=0.198, scale=0.067352)
        
        return(np.array([phi, tau, xi]))
        
    phi, tau, xi = parameter[0], parameter[1], parameter[2]
        
    if 0 < tau and tau < phi  :
        return (sc.norm(0.198, 0.06735**2).pdf(xi))
    
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


## Launch the simlation

theta, Number_gen, ess_list = ABC_PRC(3, x0, f, prior, nu, epsilon_list, E, K_list, L_list, eta, rho, N, verbose = True)



## Get some graphes


sns.kdeplot(theta[:,9,0] - theta[:,9,1])

sns.kdeplot(math.log(2)/(theta[:,9,0] - theta[:,9,1]))

sns.jointplot(theta[:,9,0] - theta[:,9,1], theta[:,9,2], kind="kde")

sns.jointplot(theta[:,9,0], theta[:,9,1], kind="kde")


for i in range(10) :
    print(Number_gen[i])

print(sum(Number_gen))


plt.plot(ess_list)


theta0 = theta

theta = np.concatenate((theta0, theta000), axis=0)


N = 50

theta_list = []

for i in range(20) :
    theta, Number_gen, ess_list = ABC_PRC(3, x0, f, prior, nu, epsilon_list, E, K_list, L_list, eta, rho, N, verbose = True)
    theta_list.append(theta)


sns.kdeplot(mean(theta_list[:,9,0]))

sns.kdeplot(mean(theta_list[:,9,1]))

sns.kdeplot(mean(theta_list[:,9,2]))
