from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import rv_continuous, norm, beta, gamma
from scipy.ndimage import gaussian_filter
from method_base import MethodBase


class GaussianModel(MethodBase):
    param_names: list = ['ampl','mean','sigma','offset']
    param_guesses: np.ndarray = np.array([.75, .5, .1,.2]) #amp, mean, sigma,offset
    param_bounds: np.ndarray = np.array([[0.01,1.],[.01,1.],[0.01,5.],[0.01,1.]]) 
    
    def __init__(self,distribution_data:np.ndarray = None):
        if distribution_data is not None: 
            self.distribution_data =  distribution_data
            self.find_priors(self.distribution_data)
        
    @property
    def distribution_data(self):
        """1d array typically projection data"""
        print('getting distribution')
        return self._distribution_data

    @distribution_data.setter
    def distribution_data(self,distribution_data):
        print('setting distribution')
        if not isinstance(distribution_data, np.ndarray):
            raise TypeError("Input must be ndarray")
        self._distribution_data = distribution_data
        self.find_priors(self._distribution_data)

    #def find_init_values()
        #self.init_values = []
        #return self.init_values
    def find_priors(self,data:np.array)->None: # maybe want dictionary or list returned then make that that return value and store it as an instance attribute
        '''do initial guesses based on data and make distribution from that guess'''
        #init_values = self.find_init_guess()
        offset = float(np.min(data))
        self.offset_prior = norm(offset, .5)
        
        ampl = np.max(gaussian_filter(data,sigma=5)) - offset
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
    

        ##discuss this 
   
        # init guesses maybe a better name dictionary form
        self.init_priors = {self.param_names[0]:ampl,self.param_names[1]:mean,self.param_names[2]:sigma,self.param_names[3]:offset}
        # could maybe pass fit_model and change forward function and log prior to accept dictionaries not lists

        #dictionary with param names and the prior distribution function as its value maybe can drop the self on self.{xx}_prior
        self.priors= {self.param_names[0]:self.ampl_prior,self.param_names[1]:self.mean_prior,self.param_names[2]:self.sigma_prior,self.param_names[3]:self.offset_prior}
        
        #list form for init guesses need this for forward, log_priors, and and projection fit fit model
        self.init_values = [ampl,mean,sigma,offset]
        
        print(self.init_priors)
    
    @staticmethod
    def forward(x:float, params:list)->float:
        amplitude = params[0]
        mean = params[1]
        sigma = params[2]
        offset = params[3]
        return amplitude * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2)) + offset
    
    def log_prior(self, params:list)->float:
        # can change too :
        # return self.priors['ampl'].logpdf(params[0]) + self.priors['mean'].logpdf(params[1]) + self.priors['sigma'].logpdf(params[2]) + self.priors['offset'].logpdf(params[3])
        return self.ampl_prior.logpdf(params[0]) + self.mean_prior.logpdf(params[1]) + self.sigma_prior.logpdf(params[2]) + self.offset_prior.logpdf(params[3])
    
    def plot_priors(self)-> None:
        num_plots = len(self.priors) + 1
        fig, axs = plt.subplots(num_plots,1,figsize = (10,10))
        for i, (param, prior) in enumerate(self.priors.items()):
            x = np.linspace(0,self.param_bounds[i][-1],len(self.distribution_data))
            axs[i].plot(x,prior.pdf(x)) 
            axs[i].axvline(self.param_bounds[i,0], ls='--', c='k',)
            axs[i].axvline(self.param_bounds[i,1], ls='--', c='k', label='bounds')
            axs[i].set_title(param + ' prior')
            axs[i].set_ylabel('Density')
            axs[i].set_xlabel(param)

        x = np.linspace(0,1,len(self.distribution_data))
        y_fit = self.forward(x,self.init_values)
        axs[-1].plot(x,self.distribution_data, label = 'Projection Data')
        axs[-1].plot(x,y_fit, label = 'Initial Guess Fit Data')
        axs[-1].set_xlabel('x')
        axs[-1].set_ylabel('Forward(x)')
        axs[-1].set_title('Initial Fit Guess')

        fig.tight_layout()
