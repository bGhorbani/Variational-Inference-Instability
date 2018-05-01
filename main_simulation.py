# This file has the code for running multiple instances of the
# AMP or Variational estimators across different parameter ranges. 
# This code generates and saves the data to the disk. Later on, the
# figures can be generated from the saved data. 
#
# This script takes 3 inputs:
# 1- The estimator type: Either 'amp' or 'variational'
# 2- The index of the relevant d, for this file, from 0 to 7
# 3- The index of the relevant n, for this file, from 0 to 7

# Author: Behrooz Ghorbani
# Email: ghorbani@stanford.edu

import simulation_code as Dev
import sys
import numpy as np
import cPickle as pickle
import os
import time

delta_grid_meta = np.linspace(1, 3, num = 8)
gsize = len(delta_grid_meta)
d_grid_meta = [1000] * gsize
n_grid_meta = []
for i in range(gsize):
    temp = np.round(delta_grid_meta[i] * d_grid_meta[i])
    n_grid_meta.append(temp)
beta_grid_meta = np.round(np.linspace(1, 15, num = 40), 2)

# The estimator type: Either 'amp' or 'variational'
estimator_type = sys.argv[1]
assert (estimator_type == 'amp' or estimator_type == 'variational')
print sys.argv
i_grid = int(sys.argv[2])
j_grid = int(sys.argv[3])

beta = round(beta_grid_meta[j_grid], 2)
n = n_grid_meta[i_grid]
d = d_grid_meta[i_grid] + 0.0
delta = round(n / d, 2)
directory = "new_k_3_" + estimator_type
filename = directory + "/" + estimator_type + '_' + str(delta) + '_' + str(beta)
damping = 0.8
nu = 1
k = 3
if not os.path.isfile(filename):
        print(estimator_type)
        print("beta")
        print(beta)
        print("n")
        print(n)
        print("d")
        print(d)

        alpha = [nu] * k
        iteration_tol = [0.005, 300, 70]
        integration_tol = 0.01        
        num_instances = 400
        seed = (250 + i_grid + j_grid)        

        result_grid = {}                
	beta_tag = round(beta, 2)               	
       	delta_tag = round(n / d, 2)
        result_grid[(beta_tag, delta_tag)] = []     
       	def simulate_instance(t):               
       	        sys.stdout.flush()
                beta = round(beta_grid_meta[j_grid], 2) + 0.0
                n = int(n_grid_meta[i_grid])
                d = int(d_grid_meta[i_grid])                            
	        np.random.seed(t + seed * 4000)
                if estimator_type == 'variational':
       	                instance = Dev.Variational_Simulation(n, d, \
                       	                      beta, alpha, iteration_tol,\
                               	                              integration_tol, 0.05)
                else:
               	        if damping < 1:
                                instance = Dev.Damped_AMP_Simulation(n, d, \
                       	                              beta, alpha, iteration_tol,\
                               	                                      integration_tol, damping, 0.05)
                        else:   
                                print "Pure AMP"
                                instance = Dev.AMP_Simulation(n, d, \
                                                      beta, alpha, iteration_tol,\
                                                                      integration_tol, 0.05)   
                        
                instance.estimate(False, memory_efficient = True)
       	        instance.compute_accuracy(False)
                return(instance)               	
        print "starting iterations"     
        for t in range(num_instances):                    
    	    t0 = time.time()                    
            result_grid[(beta_tag, delta_tag)].append(simulate_instance(t))  
            t1 = time.time()
            print "iteration : " + str(t) + " took " + str(t1 - t0)       
            sys.stdout.flush()                                                    	
	   

        if not os.path.exists(directory):
       	    os.makedirs(directory)
                
       	with open(filename, 'wb') as output:    
            pickle.dump(result_grid, output, -1)
