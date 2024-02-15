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
        
    def find_init_values(self,data:np.array)->list:
        offset = float(np.min(data))
        amplitude = np.max(gaussian_filter(data,sigma=5)) - offset
        mean = np.argmax(gaussian_filter(data,sigma=5))/(len(data))
        sigma = .1
        self.init_values = [amplitude,mean,sigma,offset]
        return self.init_values
            
    def find_priors(self,data:np.array)->None:
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
    
        offset_prior = norm(init_values[3], .5)

        self.priors= {self.param_names[0]: amplitude_prior, 
                      self.param_names[1]: mean_prior, 
                      self.param_names[2]: sigma_prior, 
                      self.param_names[3]: offset_prior
                      }
        
        return self.priors
        
    @staticmethod
    def forward(x:float, params:list)->float:
        amplitude = params[0]
        mean = params[1]
        sigma = params[2]
        offset = params[3]
        return amplitude * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2)) + offset
    
    def log_prior(self, params:list)->float:
        return np.sum([prior.logpdf[params[i]] for i, (key, prior) in enumerate(self.priors.items())])
    
    def plot_priors(self)-> None:
        num_plots = len(self.priors)
        fig, axs = plt.subplots(num_plots,1,figsize = (10,10))
        for i, (param, prior) in enumerate(self.priors.items()):
            x = np.linspace(0,self.param_bounds[i][-1],len(self.distribution_data))
            axs[i].plot(x,prior.pdf(x)) 
            axs[i].axvline(self.param_bounds[i,0], ls='--', c='k',)
            axs[i].axvline(self.param_bounds[i,1], ls='--', c='k', label='bounds')
            axs[i].set_title(param + ' prior')
            axs[i].set_ylabel('Density')
            axs[i].set_xlabel(param)
        fig.tight_layout()
        return fig,axs
    
    def plot_init_values(self):
        fig, axs = plt.subplots(1,1,figsize = (10,5))
        x = np.linspace(0,1,len(self.distribution_data))
        y_fit = self.forward(x,self.init_values)
        axs.plot(x,self.distribution_data, label = 'Projection Data')
        axs.plot(x,y_fit, label = 'Initial Guess Fit Data')
        axs.set_xlabel('x')
        axs.set_ylabel('Forward(x)')
        axs.set_title('Initial Fit Guess')
        return fig,axs