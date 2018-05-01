# This file includes the code for generating an instance of
# the topic modeling problem (Simulation Class), variational 
# inference algorithm (Variational_Simulation Class), the AMP 
# (AMP_Simulation Class), and the damped AMP algorithm 
# (Damped_AMP_Simulation). 

# Last edited by Behrooz Ghorbani on April 1st 2018.

import numpy as np
import math 
import itertools
import sys
import scipy.linalg as SCL

class Simulation(object):
    """This class simulates an instance of the problem from
        the model X = \sqrt(beta) / d * W * H^T + Z 
        where Z ~ N(0, 1 / d), rows of W ~ Dir(alpha)
        and H ~ N(0, 1). W is n-by-k and H is d-by-k.

    Main Attributes:
        _beta : signal strength
        _n : number of variables
        _d : number of observations
        _alpha : the Dirichlet parameter (k-dimensional)
        _tol : convergence tolerence
        _max_iter : maximum number of allowed estimation iterations
        _min_iter : minimum number of allowed estimation iterations
        _W : population class weights
        _H : population class profile
        _W_hat : estimated class weights
        _H_hat = estimated class profile
    """

    def __init__(self, n, d, beta, alpha, iteration_tol, integration_tol):
        """Instantiates the simulation by setting the fundamental parameters of
        the problem."""                
        self._beta = beta        
        self._n = int(n)
        self._d = int(d)
        self._alpha = np.array(alpha)
        self._delta = n / (d + 0.0)
        # The default parameters
        if (iteration_tol is None):
            self._tol = 0.005
            self._max_iters = 250
            self._min_iters = 50
        else:
            self._tol = iteration_tol[0]
            self._max_iters = iteration_tol[1]
            self._min_iters = iteration_tol[2]
            
        self._integ_tol = integration_tol
        k = len(alpha)
        self._k = int(k)
        
        # integration variables
        self._grid = None
        self._grid_inner_prods = None
        self._prior_density = None
        self._construct_grid()        
        M = self._grid.shape[0]
        self._tilt_cache = np.ones((M, 1))
        
        # Generating the Problems
        self._W = np.random.dirichlet(alpha, size = self._n)
        self._H = np.random.normal(0, 1, size = (self._d, k))
        self._Z = np.random.normal(0, (1.0 / math.sqrt(d)), size = (self._n, self._d))
        self._X = (math.sqrt(beta + 0.0) / d) * np.dot(self._W, self._H.T) + self._Z
        self._flag = False         
        
        # estimates        
        self._W_hat = np.ones((self._n, self._k)) / (k + 0.0)
        self._H_hat = np.random.normal(size = (self._d, self._k))
        
        # initial start points
        self._W_0 = np.copy(self._W_hat)                
        self._H_0 = np.copy(self._H_hat)
                
        # The attributes below record the statistics of the final estimate:      
        self._deltaW_norms = np.zeros((k, 2))
        self._deltaH_norms = np.zeros((k, 2))        
        self._iterations_taken = None
        
        # Convergence Errors
        self._err = np.zeros((self._max_iters, 1))
        
        # Distance from uninformative fixed-point
        self._dist = np.zeros((self._max_iters, 2))
        
        # Average Norm:
        self._H_ave_norm = 0
        self._W_ave_norm = 0
                
        # inner product of the column difference
        self._Wdiff_in = 0
        self._Wdiff_corr = 0    
        
        self._Hdiff_corr = 0                  
        self._Hdiff_in = 0        

        # norm of the difference
        self._Wdiff_norm = 0
        self._Hdiff_norm = 0
        
        # Calculation for Binder Cumulant
        self._delta_H_inner = 0
        self._delta_W_inner = 0                        
        self._delta_H_norm = 0
        self._delta_W_norm = 0
        self._binder_epsilon = 0.0001
        # Indicator of the convergence of the instance
        self._converged = False        

    def _construct_grid(self):
        """This function constructs a uniform grid over the simplex
        for computing numerical integrals."""
        
        tol = self._integ_tol
        N = int(1.0 / tol) - 1
        k = int(self._k)
        
        temp = np.zeros((N**(k - 1), k))
        indices = [0] * k
        one_dim_grid = np.linspace(tol, 1.0, num = int(N), endpoint = False)          
        index = 0
        for i in range(N ** (k - 1)):
            tau = []
            zeta = int(i)
            for j in range(k - 1):
                indices[j] = zeta % N
                zeta = zeta / N
                tau.append(one_dim_grid[indices[j]])
            if np.sum(tau) < 1:
                tau.append(1 - np.sum(tau))                
                temp[index, :] = np.array(tau)
                index += 1
        self._grid = temp[:index, :]
        
        # To speed up our computation, we precompute some
        # recurring quantities here
        nu = self._alpha[0] - 1        
        M = self._grid.shape[0]        
        self._prior_density = np.prod(self._grid ** nu, axis = 1, keepdims=True)
        self._grid_inner_prods = {}
        for i in range(k):
            for j in range(k):
                t1 = self._grid[:, i].reshape(M, 1)
                t2 = self._grid[:, j].reshape(M, 1)
                self._grid_inner_prods[(i, j)] = t1 * t2
   
    def _set_cache(self, A):
        """ This function precomputes and caches some recurring computation in the integration process."""
        Zeta = np.dot(self._grid, A.T)
        Zeta2 = self._grid * Zeta
        t1b = np.sum(Zeta2, axis = 1) / 2.0
        M = self._grid.shape[0]
        self._tilt_cache = t1b.reshape(M, 1)
        
    def _fs(self, A, b, cache = False):
        ''' This function computes the first and second moments of
        the posterior distribution. In addition,
        the log(normalizing factor) of the distribution is also
        computed. '''        
        
        b = b.reshape(len(b), 1)
        k = int(self._k)
        tol = self._integ_tol
        N = int(1.0 / tol)
        M = self._grid.shape[0]
        
        # Computing the tilt to the density:                                
        t1a = np.dot(self._grid, b)        
        # if the part of the tilt  due to A is cached
        if cache:
            t1b = self._tilt_cache
        else:
            Zeta = np.dot(self._grid, A.T)
            Zeta2 = self._grid * Zeta
            t1b = np.sum(Zeta2, axis = 1) / 2.0
            t1b = t1b.reshape(M, 1)        
        t1 = t1a - t1b             
                
        # computing the prior density
        norm_const = np.max(t1) - 1.0               
        t2 = self._prior_density
                
        density = np.exp(t1 - norm_const) * t2
        
        # normalization factor
        q = np.sum(density) / (N ** (k - 1) + 0.0)
        if q <= 0:        
            raise Exception('Probability density has no mass')
        
        # first moment        
        s = np.dot(density.T, self._grid).T / (N ** (k - 1) + 0.0)      
        s = s / np.sum(s) * q
        
        # second moment
        omega = np.zeros((k, k))        
        for i in range(k): 
            for j in range(i, k):
                omega[i, j] = np.dot(density.T, self._grid_inner_prods[(i, j)]) / (N ** (k - 1) + 0.0)  
                omega[j, i] = omega[i, j]
        
        # Scaling factor, first moment, second moment                
        return (np.log(q) + norm_const, s / q, omega / q)    

    def iteration(self):
        """ This function does one iteration of estimation according to (Variational / AMP) method. 
        This is an abstract method and should be implemented in the lower level classes."""
        pass

    @staticmethod
    def _min_permutation_error(X, Y):
        """ For two n-by-m matrices with m<<n, this function computes the minimum
        ell_infinity norm of the difference of the two matrices over all permutations
        of the columns."""
        n = X.shape[0]
        m = X.shape[1]
        if not ((Y.shape[0] == n) and (Y.shape[1] == m)):
            raise Exception("The Dimensions are inconsistent")
        permutations = list(itertools.permutations(range(m)))
        N = len(permutations)
        value = float("inf")
        for perm in permutations:
            X_temp = X[:, perm]            
            Z = X_temp - Y
            temp = np.max(np.abs(Z))
            value = np.min([value, temp])
        return(value)
        
    def estimate(self, verbose = False, memory_efficient = True):      
        """ This function runs an iterative estimation procedure for estimating W_hat and
        H_hat. The function stops when the ell_infinity norm between two consequitive estimates
        is small or the number of iterations has surpassed self._max_iters."""
        self._iterations_taken = 0
        convergence_criteria = float("inf")
        while ((self._iterations_taken < self._min_iters) or ((convergence_criteria > self._tol) and (self._iterations_taken < self._max_iters))):
            W_hat = np.copy(self._W_hat)
            H_hat = np.copy(self._H_hat)
            # Do one iteration of the estimator            
            self.iteration()            
            self._err[self._iterations_taken] = Simulation._min_permutation_error(W_hat, self._W_hat);
            self._dist[self._iterations_taken, 0] = np.linalg.norm(self._W_hat - self._W_0)
            self._dist[self._iterations_taken, 1] = np.linalg.norm(self._H_hat - self._H_0)
            convergence_criteria = self._err[self._iterations_taken]  
            self._iterations_taken += 1            
            if verbose:
                print(convergence_criteria)
        if verbose:
            print(self._iterations_taken)

        self._converged = (self._iterations_taken < self._max_iters)
        if np.isnan(convergence_criteria):
            raise Exception('Convergence Criteria is NAN')        

    def release_memory(self):
        """This function deletes all of the high-dimensional matrices so that the object can be saved
        in a memory efficient manner."""
        del self._H
        del self._W
        del self._W_hat
        del self._H_hat
        del self._X
        del self._Z        
        del self._W_0
        del self._H_0
        
        del self._grid 
        del self._prior_density 
        del self._tilt_cache
        del self._grid_inner_prods
        
    def compute_accuracy(self, verbose = False, Eigenvector_analysis = False):
        """ This function computes accuracy statistics of the instance. Afterwards, memory consuming
        attributes of the object are deleted. """
        if not self._flag:
            k = len(self._alpha)            
            
            # delta_W inner product and norms 
            zeta = np.mean(self._W, axis = 1).reshape((self._n, 1))
            zeta = np.dot(zeta, np.ones((1, k)))
            temp_W = self._W - zeta
            
            zeta = np.mean(self._W_hat, axis = 1).reshape((self._n, 1))
            self._W_ave_norm = np.linalg.norm(zeta)
            zeta = np.dot(zeta, np.ones((1, k)))
            temp_W2 = self._W_hat - zeta + self._binder_epsilon * np.random.normal(size = (self._n, k))        
            
            self._Wdiff_corr =  np.corrcoef(temp_W.T, temp_W2.T)
            self._Wdiff_in =  np.dot(temp_W.T, temp_W2) 
            for i in range(k):
                self._deltaW_norms[i, 0] = np.linalg.norm(temp_W[:, i])                 
                self._deltaW_norms[i, 1] = np.linalg.norm(temp_W2[:, i])                                 
            self._Wdiff_norm = np.linalg.norm(temp_W2)  / np.sqrt(self._n + 0.0)                              
            
            # delta_H inner product and norms 
            zeta = np.mean(self._H, axis = 1).reshape((self._d, 1))            
            zeta = np.dot(zeta, np.ones((1, k)))
            temp_H = self._H - zeta
            
            zeta = np.mean(self._H_hat, axis = 1).reshape((self._d, 1))
            self._H_ave_norm = np.linalg.norm(zeta)
            zeta = np.dot(zeta, np.ones((1, k)))
            temp_H2 = self._H_hat - zeta  + self._binder_epsilon * np.random.normal(size = (self._d, k))        
            
            self._Hdiff_corr =  np.corrcoef(temp_H.T, temp_H2.T)
            self._Hdiff_in = np.dot(temp_H.T, temp_H2) / (self._d + 0.0)
            for i in range(k):
                self._deltaH_norms[i, 0] = np.linalg.norm(temp_H[:, i])                 
                self._deltaH_norms[i, 1] = np.linalg.norm(temp_H2[:, i])                                             
            self._Hdiff_norm = np.linalg.norm(temp_H2) / np.sqrt(self._d + 0.0)

            # Calculations Necessary for Binder Cumulant:
            if k == 2:                      
                    temp_W2 = self._W_hat[:, 1] - self._W_hat[:, 0]                
                    self._delta_W_norm = np.linalg.norm(temp_W2)
                    temp_W2 = temp_W2.reshape((self._n, 1)) + self._binder_epsilon * np.random.normal(size = (self._n, 1))        
                    temp_W = self._W[:,1] - self._W[:, 0]                                                 
                    Wdiff_in =  np.dot(temp_W.T, temp_W2)
                    self._delta_W_inner = Wdiff_in / (np.linalg.norm(temp_W) * np.linalg.norm(temp_W2))                                                

                    temp_H2 = self._H_hat[:,1] - self._H_hat[:, 0]                        
                    self._delta_H_norm = np.linalg.norm(temp_H2)
                    temp_H2 = temp_H2.reshape((self._d, 1)) + self._binder_epsilon * np.random.normal(size = (self._d, 1))        
                    temp_H = self._H[:,1] - self._H[:, 0]                         
                    Hdiff_in = np.dot(temp_H.T, temp_H2)
                    self._delta_H_inner = Hdiff_in / (np.linalg.norm(temp_H) * np.linalg.norm(temp_H2))                                                

            self._flag = True
            self.release_memory()

class Variational_Simulation(Simulation):
    """This class is a sub-class of Simulation and
    specializes it for variational inference method.     
    """

    def __init__(self, n, d, beta, alpha, iteration_tol, integration_tol, eps):
        """Instantiates the simulation by setting the fundamental parameters of
        the problem."""               
        super(Variational_Simulation, self).__init__(n, d, beta, alpha, iteration_tol, integration_tol)                
        self._tq2 = 0
        self._tq1 = 0
        self._q1 = 0
        self._q2 = 0        
        self._compute_initial_conds(eps)                
        k = len(alpha)
        
        # Extra parameters associated with the Variational Estimator
        self._Q = (self._q1 * np.identity(k) + self._q2 * np.ones((k, k))) 
        self._tQ = np.identity(k)                
        self._tG = np.zeros((k, k))        
        self._m = np.dot(self._X.T, self._W_hat) * np.sqrt(beta)        
        self._tm = np.dot(self._X, self._H_hat) * np.sqrt(beta)                
        self._kl = np.zeros((self._max_iters, 1))
        self._energy1 = 0
   
    def _compute_initial_conds(self, eps):  
        """ This function computes the uninformative fixed point of the model when there is no signal. 
        This fixed-point is perturbed and then used for initialization of the iterative algorithm. """
        k = self._k + 0.0
        k0 = int(self._k)
        n = self._n + 0.0
        d = self._d + 0.0
        self._W_hat = np.ones((self._n, self._k)) / (k + 0.0)
        self._W_0 = np.copy(self._W_hat)
        
        # computing the H_0
        tol_h_0 = 0.005
        difference = 1000
        q1 = 1
        tq1 = 1
        while (difference > tol_h_0):
            tq1_hat = self._beta / (1.0 + q1)
            lgq, mu, omega = self._fs(tq1_hat * np.eye(k0), np.zeros(k0))
            q1_hat = self._beta * self._delta / (k - 1.0) * (k * omega[0, 0] - 1.0 / k)
            difference = np.abs(q1 - q1_hat) + np.abs(tq1 - tq1_hat)
            q1 = q1_hat
            tq1 = tq1_hat 
        q2 = (self._delta * self._beta - k * q1) / (k ** 2 + 0.0)        
        m_star = np.dot(self._X.T, self._W_0)
        
        self._tq2 = self._beta * ((np.linalg.norm(m_star) ** 2) / (k * d * (1 + q1 + k * q2) ** 2) \
                            - (q2) / ((1 + q1) * (1 + q1 + k * q2)))
        self._q2 = q2
        self._q1 = q1
        self._tq1 = tq1
        
        h_star = m_star * np.sqrt(self._beta) / (1.0 + q1 + k * q2)
        self._H_hat = h_star
        self._H_0 = np.copy(self._H_hat)
        
        # Adding Perturbations:
        temp = np.random.dirichlet(self._alpha, size = self._n)
        temp = temp * np.linalg.norm(self._W_hat) / np.linalg.norm(temp)
        self._W_hat = (1 - eps) * self._W_hat + eps * temp

        temp = np.random.normal(0, 1, size = (self._d, self._k))
        temp = temp * np.linalg.norm(self._H_hat) / np.linalg.norm(temp)        
        self._H_hat = (1 - eps) * self._H_hat + eps * temp
                
    def release_memory(self):
        super(Variational_Simulation, self).release_memory()                        
        del self._m
        del self._tm        
        del self._tG
        
    def _compute_kl(self):
        """ This function computes the KL-divergence of the variational posterior and the true posterior. 
        To save computation, constants are left out of the calculation. """
        beta = self._beta + 0.0
        n = self._n + 0.0
        d = self._d + 0.0
        k = len(self._alpha)  
        
        covariance = np.identity(k) + self._Q
        covariance = np.linalg.inv(covariance) 
        F = np.sqrt(beta) * self._H_hat
        tF = np.sqrt(beta) * self._W_hat
        sG =  np.dot(F.T, F) + d * beta * covariance
        stG = self._tG                        
        
        fit = np.dot(tF, F.T)
        term1 = - 1.0 / np.sqrt(beta) * np.trace(np.dot(self._X.T, fit))         
        term1 += 1.0 / (2 * d * beta) * np.trace(np.dot(sG, stG)) 
        
        term2 = 0
        term2 = 1.0 / np.sqrt(beta) * np.trace(np.dot(self._tm, tF.T))
        term2 += -1.0 / (2 * beta) * np.trace(np.dot(self._tQ, stG))
        
        term2 +=  1.0 / np.sqrt(beta) * np.trace(np.dot(self._m, F.T))
        term2 += -1.0 / (2 * beta) * np.trace(np.dot(self._Q, sG))        
        term3 = 0
        term3 += - self._energy1
        determinant = np.linalg.det(covariance)
        m_term = - np.sum(self._m * np.dot(self._m, covariance / 2.0), axis = 1)
        term3 += - d * np.log(np.sqrt((2 * math.pi) ** k * determinant)) + np.sum(m_term)
        self._kl[self._iterations_taken] =  term1 + term2 + term3
    
    def iteration(self):
        """ This function does one iteration of Variational method. 
        In other words, it goes from W_t to W_{t+1}"""
        
        beta = self._beta + 0.0
        n = self._n + 0.0
        d = self._d + 0.0
        k = len(self._alpha)        
       
        # updating H_t                
        covariance = np.identity(k) + self._Q
        covariance = np.linalg.inv(covariance) 
        F = np.sqrt(beta) * np.dot(self._m, covariance)
        self._H_hat = F / np.sqrt(beta)
        
        self._tm = np.dot(self._X, F)
        self._tQ = np.dot(F.T, F) / d + beta * covariance         

        # deriving W_{t + 1}        
        self._energy1 = 0                     
        tF = np.zeros((self._n, k))
        self._tG = np.zeros((k, k))
        self._set_cache(self._tQ)
        for i in range(self._n):     
            q, mu, omega = self._fs(A = self._tQ, b = self._tm[i, :], cache = True)                        
            tF[i, :] = np.sqrt(beta) * mu[:, 0]
            self._tG += omega * beta                
            self._energy1 += q                        
        self._W_hat = tF / np.sqrt(beta)
        # Here Compute the KL
        self._compute_kl()
        self._m = np.dot(self._X.T, tF)
        self._Q = self._tG / d


class AMP_Simulation(Simulation):
    """This class is a sub-class of Simulation and
    specializes it for AMP estimator. 
    
    Extra Attributes:
    _JW = Jacobian with respect to W_t
    _JH = Jacobian with respect to H_t
    """

    def __init__(self, n, d, beta, alpha, iteration_tol, integration_tol, eps):
        """Instantiates the simulation by setting the fundamental parameters of
        the problem."""             
        super(AMP_Simulation, self).__init__(n, d, beta, alpha, iteration_tol, integration_tol)
        k = len(alpha)                    
        # Extra estimated  parameters
        self._Q = np.identity(k)
        self._tQ = np.identity(k)                
        self._tG = np.zeros((k, k))        
        self._m = np.zeros((self._d, k))
        self._tm = np.zeros((self._n, k))      
        
        self._kl = None
        self._energy1 = 0                
        self.compute_initial_conds(eps)
    
    def release_memory(self):
        super(AMP_Simulation, self).release_memory()
        del self._m
        del self._tm
        del self._tG

    @staticmethod    
    def jacobian_W(omega, mu, q): 
        temp = np.outer(mu, mu)
        return (omega - temp)
    
    def iteration(self):
        """ This function does one iteration of Variational method. 
        In other words, it goes from W_t to W_{t+1}"""
        
        beta = self._beta + 0.0
        n = self._n + 0.0
        d = self._d + 0.0
        k = len(self._alpha)        
       
        # updating H_t                
        covariance = np.identity(k) + self._Q
        covariance = np.linalg.inv(covariance) 
        F = np.sqrt(beta) * np.dot(self._m, covariance)
        self._H_hat = F / np.sqrt(beta)
        
        B = np.sqrt(beta) * covariance
        self._tm = np.dot(self._X, F) - np.dot(self._W_hat, B) * np.sqrt(beta) 
        self._tQ = np.dot(F.T, F) / d 
        
        # deriving W_{t + 1}        
        self._energy1 = 0                     
        tF = np.zeros((self._n, k))
        self._tG = np.zeros((k, k))
        JW = np.zeros((k, k))
        self._set_cache(self._tQ)
        for i in range(self._n):     
            q, mu, omega = self._fs(A = self._tQ, b = self._tm[i, :], cache = True)                        
            tF[i, :] = np.sqrt(beta) * mu[:, 0]
            self._tG += omega * beta    
            JW += AMP_Simulation.jacobian_W(omega, mu, q)                 
            self._energy1 += q                        
        self._W_hat = tF / np.sqrt(beta)                     
        
        C = JW / d * np.sqrt(beta)        
        self._m = np.dot(self._X.T, tF) - np.dot(F, C)
        self._Q = np.dot(tF.T, tF) / d  
    
    def compute_initial_conds(self, eps):
        """ This function computes the uninformative fixed point of the model when there is no signal. 
        This fixed-point is perturbed and then used for initialization of the iterative algorithm. """
        k = self._k + 0.0
        k0 = self._k
        n = self._n + 0.0
        d = self._d + 0.0
        beta = self._beta
        
        #compute Q
        self._Q = np.ones((k0 , k0)) / (k ** 2) * (n / d) * beta
        covariance = np.identity(k0) + self._Q
        covariance = np.linalg.inv(covariance) 
        
        #compute tOmega
        q, mu, tomg = self._fs(A = np.zeros((k0, k0)), b = np.zeros((k0, 1)))
        tomega = (tomg - np.dot(mu, mu.T)) * (n / d)
        
        temp = np.dot(covariance, tomega)[:, 0]
        LHS = beta * np.dot(np.ones((1, k0)), temp)
        RHS = np.sqrt(beta) * np.sum(self._X.T, axis = 1, keepdims = True) / k
        self._m = RHS / (LHS + 1) * np.ones((self._d, k0))
        
        F = np.sqrt(beta) * np.dot(self._m, covariance)
        tF = np.ones((self._n, k0)) / k * np.sqrt(beta)
        
        self._tQ = np.dot(F.T, F) / d
        self._tm = np.dot(self._X, F) - np.sqrt(beta) * np.dot(tF, covariance) 
        self._W_hat = tF / np.sqrt(beta)
        self._H_hat = F / np.sqrt(beta)
        
        perturbation = np.random.normal(size = (int(d), k0))
        NP = np.linalg.norm(perturbation)
        NM = np.linalg.norm(self._m)
        self._m += perturbation / NP * eps * NM
        
        self._W_0 = np.copy(self._W_hat)
        self._H_0 = np.copy(self._H_hat)
        
    def _compute_energy(self):
        pass


class Damped_AMP_Simulation(Simulation):
    """This class is a sub-class of Simulation and
    specializes it for (Damped) AMP estimator. 
    
    Extra Attributes:
    _JW = Jacobian with respect to W_t
    _JH = Jacobian with respect to H_t
    _gamma = The damping parameter
    """

    def __init__(self, n, d, beta, alpha, iteration_tol, integration_tol, gamma, eps):
        """Instantiates the simulation by setting the fundamental parameters of
        the problem."""             
        super(Damped_AMP_Simulation, self).__init__(n, d, beta, alpha, iteration_tol, integration_tol)
        k = len(alpha)
        
        # Extra estimated  parameters        
        self._KW = np.zeros((k, k))
        self._KH = np.zeros((k, k))
        self._m = np.zeros((d, k))
        self._tm = np.zeros((n, k))
        # The damping parameter. Chosen from (0, 1]
        self._gamma = gamma
        self._tQ = np.identity(k)
        self._Q = np.identity(k)
        self.compute_initial_conds(eps)
        
    def compute_initial_conds(self, eps):
        k = self._k + 0.0
        k0 = self._k
        n = self._n + 0.0
        d = self._d + 0.0
        beta = self._beta
        
        #compute Q
        self._Q = np.ones((k0 , k0)) / (k ** 2) * (n / d) * beta
        covariance = np.identity(k0) + self._Q
        covariance = np.linalg.inv(covariance) 
        
        #compute tOmega
        q, mu, tomg = self._fs(A = np.zeros((k0, k0)), b = np.zeros((k0, 1)))
        tomega = (tomg - np.dot(mu, mu.T)) * (n / d)
        
        temp = np.dot(covariance, tomega)[:, 0]
        LHS = beta * np.dot(np.ones((1, k0)), temp)
        RHS = np.sqrt(beta) * np.sum(self._X.T, axis = 1, keepdims = True) / k
        self._m = RHS / (LHS + 1) * np.ones((self._d, k0))
        
        F = np.sqrt(beta) * np.dot(self._m, covariance)
        tF = np.ones((self._n, k0)) / k * np.sqrt(beta)
        
        self._tQ = np.dot(F.T, F) / d
        self._tm = np.dot(self._X, F) - np.sqrt(beta) * np.dot(tF, covariance) 
        self._W_hat = tF / np.sqrt(beta)
        self._H_hat = F / np.sqrt(beta)
        
        perturbation = np.random.normal(size = (int(d), k0))
        NP = np.linalg.norm(perturbation)
        NM = np.linalg.norm(self._m)
        self._m += perturbation / NP * eps * NM
        
        self._W_0 = np.copy(self._W_hat)
        self._H_0 = np.copy(self._H_hat)
        
    def iteration(self):
        """ This function does one iteration of AMP method. 
        In other words, it goes from W_t to W_{t+1}"""   
        
        beta = self._beta + 0.0
        d = self._d + 0.0        
        n = self._n + 0.0
        k = len(self._alpha)    
        # Damping Parameter
        gamma = self._gamma + 0.0
        
        # updating H                           
        covariance = np.identity(k) + self._Q
        covariance = np.linalg.inv(covariance)                
        self._H_hat = np.dot(self._m, covariance)                        
        F = self._H_hat * np.sqrt(beta)
        
        JH = d * covariance
        self._KH = (1 - gamma) * self._KH + JH        
        # Updating W_hat (weights)
        self._tm = gamma * np.dot(self._X, F) + (1 - gamma) * self._tm \
                - (gamma ** 2) * (beta / d) * np.dot(self._W_hat, self._KH)
        self._tQ = np.dot(F.T, F) / d                
        
        JW = np.zeros((k, k))                
        self._set_cache(self._tQ)
        for i in range(self._n):
            q, mu, omega = self._fs(self._tQ, self._tm[i, :], cache = True)
            self._W_hat[i, :] = mu[:, 0]
            JW += AMP_Simulation.jacobian_W(omega, mu, q)  
        
        self._KW = (1 - gamma) * self._KW + JW
        self._m = gamma * np.sqrt(beta) * np.dot(self._X.T, self._W_hat)  \
                + (1 - gamma) * self._m - (beta / d) * (gamma ** 2) * np.dot(self._H_hat, self._KW.T)                
        self._Q = (beta / d) * (np.dot(self._W_hat.T, self._W_hat)) 
        
    def release_memory(self):
        super(Damped_AMP_Simulation, self).release_memory()
        del self._m
        del self._tm
