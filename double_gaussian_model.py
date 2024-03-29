from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import rv_continuous, norm, beta, gamma
from scipy.ndimage import gaussian_filter
from method_base import MethodBase


class DoubleGaussianModel(MethodBase):
    param_names: list = ['amplitude','mean','sigma', 'amplitude2', 'mean2', 'sigma2','offset']
    param_guesses: np.ndarray = np.array([.75, .5, .1,.375,.25,.01,.2]) #amp, mean, sigma, amp2, mean2, sigma2, offset
    param_bounds: np.ndarray = np.array([[0.01,1.],[.01,1.],[0.01,5.],[0.01,1.],[.01,1.],[0.01,5.],[0.01,1.]]) 
    def __init__(self,distribution_data:np.ndarray = None):
        if distribution_data is not None: 
            self.distribution_data =  distribution_data
            self.find_priors(self.distribution_data)

    def find_init_values(self,data:np.ndarray)->list:
        offset = float(np.min(data))
        amplitude = np.max(gaussian_filter(data,sigma=5)) - offset
        mean = np.argmax(gaussian_filter(data,sigma=5))/(len(data))
        sigma = .1

        amplitude2 = amplitude/3
        # want a double peaked mean2_prior distribution that is very weak where mean_prior has a high likelyhood
        mean2 = np.argmax(gaussian_filter(data,sigma=5))/(2*len(data))
        sigma2 = .01

        self.init_values = [amplitude,mean,sigma,amplitude2, mean2, sigma2, offset]
        return self.init_values
    
    def find_priors(self,data:np.ndarray)->dict:
        '''do initial guesses based on data and make distribution from that guess'''

        init_values = self.find_init_values(data)
        
        amplitude_mean = init_values[0]
        amplitude_var = 0.05
        amplitude_alpha = (amplitude_mean**2)/amplitude_var
        amplitude_beta = amplitude_mean/amplitude_var
        amplitude_prior = gamma(amplitude_alpha, loc = 0, scale = 1/amplitude_beta)
    
        mean_prior = norm(init_values[1],0.1)

        sigma_alpha = 2.5
        sigma_beta = 5.0
        sigma_prior = gamma(sigma_alpha,loc=0,scale = 1/sigma_beta)

        amplitude2_mean = init_values[3]
        amplitude2_var = 0.05
        amplitude2_alpha = (amplitude2_mean**2)/amplitude2_var
        amplitude2_beta = amplitude2_mean/amplitude2_var
        amplitude2_prior = gamma(amplitude2_alpha, loc = 0, scale = 1/amplitude2_beta)

        # want bimodal distribution with lowest likelyhood inbetween peaks centered at mean1
        mean2_prior = norm(init_values[4],0.1)

        sigma2_alpha = 2.5
        sigma2_beta = 5.0
        sigma2_prior = gamma(sigma2_alpha,loc=0,scale = 1/sigma2_beta)

        offset_prior = norm(init_values[6], .5)

        self.priors= {self.param_names[0]: amplitude_prior, 
                      self.param_names[1]: mean_prior, 
                      self.param_names[2]: sigma_prior, 
                      self.param_names[3]: amplitude2_prior,
                      self.param_names[4]: mean2_prior,
                      self.param_names[5]: sigma2_prior,
                      self.param_names[6]: offset_prior
                      }
        return self.priors

    @staticmethod
    def forward(x:float, params:list)->float:
        amplitude = params[0]
        mean = params[1]
        sigma = params[2]
        amplitude2 = params[3]
        mean2 = params[4]
        sigma2 = params[5]        
        offset = params[6]
        return amplitude * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2)) + amplitude2 * np.exp(-(x - mean2) ** 2 / (2 * sigma2 ** 2)) + offset

    def log_prior(self,params): 
        return np.sum([prior.logpdf[params[i]] for i, (key, prior) in enumerate(self.priors.items())])
