from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import rv_continuous, norm, beta, gamma
from scipy.ndimage import gaussian_filter
from method_base import MethodBase


class GaussianModel(MethodBase):
    param_guesses: np.ndarray = np.array([.75, .5, .1,.2]) #amp, mean, sigma,offset
    param_bounds: np.ndarray = np.array([[0.01,1.],[.01,1.],[0.01,5.],[0.01,1.]]) 
    param_names: list = ['ampl','mean',
                        'sigma','offset']
    
    def __init__(self,distribution_data:np.ndarray = None):
        if distribution_data is not None: 
            self.distribution_data =  distribution_data
            self.find_priors(self.distribution_data)
        
    @property
    def distribution_data(self):
        """Image, typically numpy array or 2darray"""
        print('getting distribution')
        return self._distribution_data

    @distribution_data.setter
    def distribution_data(self,distribution_data):
        print('setting distribution')
        if not isinstance(distribution_data, np.ndarray):
            raise TypeError("Input must be ndarray")
        self._distribution_data = distribution_data
        self.find_priors(distribution_data)


    def find_priors(self,data):
        '''do initial guesses based on data and make distribution from that guess, very rough first pass'''
        # clean this
        # addedv priors to dict 
        offset = float(np.min(data))
        self.offset_prior = norm(offset, .5)
        
        ampl = np.max(gaussian_filter(data,sigma=5)) -offset
        ampl_mean = ampl
        ampl_var = 0.05
        ampl_alpha = (ampl_mean**2)/ampl_var
        ampl_beta = ampl_mean/ampl_var
        self.ampl_prior = gamma(ampl_alpha, loc = 0, scale = 1/ampl_beta)
        
        mean = np.argmax(gaussian_filter(data,sigma=5))/(len(data))
        self.mean_prior = norm(mean,0.1)

        sigma = .1
        sigma_alpha = 2.5
        sigma_beta = 5.0
        self.sigma_prior = gamma(sigma_alpha,loc=0,scale = 1/sigma_beta)
    
        self.init_priors = {self.param_names[0]:ampl,self.param_names[1]:mean,self.param_names[2]:sigma,self.param_names[3]:offset}
        print(self.init_priors)

        #### change type hints in base class after returning array
    
    @staticmethod
    def forward(x, params):
        # unpack params
        amplitude = params[0]
        mean = params[1]
        sigma = params[2]
        offset = params[3]
        return amplitude * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2)) + offset
    
    def log_prior(self, params):
        return self.ampl_prior.logpdf(params[0]) + self.mean_prior.logpdf(params[1]) + self.sigma_prior.logpdf(params[2]) + self.offset_prior.logpdf(params[3])
    
    def plot_priors(self):
        pass

    

