from matplotlib import pyplot as plt
import numpy as np
from abc import ABC, abstractmethod
import scipy.optimize
from scipy.stats import rv_continuous, norm, beta, gamma
from scipy.ndimage import gaussian_filter

# work in progress


class Base(ABC):
    @staticmethod
    @abstractmethod
    def forward(x:np.array, params:np.array)->np.array:
        pass
    @abstractmethod
    def find_priors(self,data:np.array)->None:
        pass
    
    def log_likelihood(self, x, y, params):
        return -np.sum((y - self.forward(x, params)) ** 2)

    @abstractmethod
    def log_prior(self, params):
        pass
    
    def loss(self, params, x, y, use_priors=False):
        l = -self.log_likelihood(x, y, params)
        if use_priors:
            l = l - self.log_prior(params)
        return l
    

class GaussianModel(Base):
    param_guesses: np.ndarray = np.array([.75, .5, .1,.2]) #amp, mean, sigma,offset
    param_bounds: np.ndarray = np.array([[0.01,1.],[.01,1.],[0.01,5.],[0.01,1.]]) 

    def __init__(self,distribution_data):
        self.distribution_data =  distribution_data
        self.find_priors(self.distribution_data)
        
    def find_priors(self,data):
        '''do initial guesses based on data and make distribution from that guess, very rough first pass'''
        offset = float(np.min(data))
        ampl = 1 - offset
        mean = np.argmax(gaussian_filter(data,sigma=5))/(len(data))
        sigma = .1
        print( ampl, mean, sigma, offset)
        
        self.init_priors = [ampl,mean,sigma,offset]
        
        self.offset_prior = norm(offset, .5)
        #self.ampl_prior = norm(ampl,1)
        mean_ampl = 0.8
        var_ampl = 0.05
        alpha = (mean_ampl**2)/var_ampl
        beta = mean_ampl/var_ampl
        self.ampl_prior = gamma(alpha, loc = 0, scale = 1/beta)
        self.mean_prior = norm(mean,0.1)
        

        sigma_alpha = 2.5
        sigma_beta = 5.0#sigma_alpha/sigma
        self.sigma_prior = gamma(sigma_alpha,loc=0,scale = 1/sigma_beta)
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
    
    #takes some x and returns the amplitude of the distributions at x s
    
    def plot_priors(self):
        fig, axs = plt.subplots(4,1,figsize = (10,10))
        
        ax = axs[0]
        x = np.linspace(0,1,len(self.distribution_data))
        ax.plot(x, self.ampl_prior.pdf(x))
        #ax.axvline(self.param_bounds[0,0], ls='--', c='k',)
        #ax.axvline(self.param_bounds[0,1], ls='--', c='k', label='bounds')
        ax.set_title('Ampl Prior')
        ax.set_ylabel('Density')
        ax.set_xlabel(r'$\mathrm{A}$')
        
        ax = axs[1]

        ax.plot(x, self.mean_prior.pdf(x))
        #ax.axvline(self.param_bounds[1,0], ls='--', c='k',)
        #ax.axvline(self.param_bounds[1,1], ls='--', c='k', label='bounds')
        ax.set_title('Mean Prior')
        ax.set_ylabel('Density')
        ax.set_xlabel(r'$\mu$')

        ax = axs[2]
        sig_linspace = np.linspace(0,5,len(self.distribution_data))
        ax.plot(sig_linspace, self.sigma_prior.pdf(sig_linspace))
        #ax.axvline(self.param_bounds[2,0], ls='--', c='k',)
        #ax.axvline(self.param_bounds[2,1], ls='--', c='k', label='bounds')
        ax.set_xlabel('x')
        ax.set_title('Sigma Prior')
        ax.set_ylabel('Density')
        ax.set_xlabel(r'$\sigma$')
        
        ax = axs[3]
        ax.plot(x, self.sigma_prior.pdf(x))
        #ax.axvline(self.param_bounds[3,0], ls='--', c='k',)
        #ax.axvline(self.param_bounds[3,1], ls='--', c='k', label='bounds')
        ax.set_xlabel('x')
        ax.set_title('Offset Prior')
        ax.set_ylabel('Density')
        ax.set_xlabel(r'$\mathrm{Offset}$')


        fig.tight_layout()
        
    def plot_initial_guess(self):
        x_fit = np.linspace(0,1,len(self.distribution_data))
        y_fit = self.forward(x_fit,self.init_priors)
        plt.plot(x_fit,self.distribution_data, label = 'Projection Data')
        plt.plot(x_fit,y_fit, label = 'Guess Fit')
        plt.show()
