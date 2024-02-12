from matplotlib import pyplot as plt
import numpy as np
from abc import ABC, abstractmethod
import scipy.optimize
from scipy.stats import rv_continuous, norm, beta, gamma
from scipy.ndimage import gaussian_filter

# work in progress


class MethodBase(ABC):
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

    @property
    def init_prior(self):
        """Initial Priors store in a dictionary where the keys are the complete set of parameters of the Model"""
        return self._init_prior 
    
    @init_prior.setter
    def init_prior(self,init_prior):
        if not isinstance(init_prior,dict):
            raise TypeError("Input must be a dictionary")
        self._init_prior = init_prior   

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
        ampl = np.max(gaussian_filter(data,sigma=5)) -offset
        mean = np.argmax(gaussian_filter(data,sigma=5))/(len(data))
        sigma = .1

    
        self.init_priors = {self.param_names[0]:ampl,self.param_names[1]:mean,self.param_names[2]:sigma,self.param_names[3]:offset}
        #print(self.init_prior)

        self.offset_prior = norm(offset, .5)
        #need to fix values for this
        mean_ampl = ampl
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
        pass
        '''
        fig, axs = plt.subplots(5,1,figsize = (10,10))
        # takes in arbitrary set of priors, iterate through dictionary
        ax = axs[0]
        x = np.linspace(0,1,len(self.distribution_data))
        ax.plot(x, self.ampl_prior.pdf(x))
        ax.axvline(self.param_bounds[0,0], ls='--', c='k',)
        ax.axvline(self.param_bounds[0,1], ls='--', c='k', label='bounds')
        ax.set_title('Ampl Prior')
        ax.set_ylabel('Density')
        ax.set_xlabel(r'$\mathrm{A}$')
        
        ax = axs[1]

        ax.plot(x, self.mean_prior.pdf(x))
        ax.axvline(self.param_bounds[1,0], ls='--', c='k',)
        ax.axvline(self.param_bounds[1,1], ls='--', c='k', label='bounds')
        ax.set_title('Mean Prior')
        ax.set_ylabel('Density')
        ax.set_xlabel(r'$\mu$')

        ax = axs[2]
        sig_linspace = np.linspace(0,5,len(self.distribution_data))
        ax.plot(sig_linspace, self.sigma_prior.pdf(sig_linspace))
        ax.axvline(self.param_bounds[2,0], ls='--', c='k',)
        ax.axvline(self.param_bounds[2,1], ls='--', c='k', label='bounds')
        ax.set_xlabel('x')
        ax.set_title('Sigma Prior')
        ax.set_ylabel('Density')
        ax.set_xlabel(r'$\sigma$')
        
        ax = axs[3]
        ax.plot(x, self.sigma_prior.pdf(x))
        ax.axvline(self.param_bounds[3,0], ls='--', c='k',)
        ax.axvline(self.param_bounds[3,1], ls='--', c='k', label='bounds')
        ax.set_xlabel('x')
        ax.set_title('Offset Prior')
        ax.set_ylabel('Density')
        ax.set_xlabel(r'$\mathrm{Offset}$')

        ax = axs[4]
        y_fit = self.forward(x,self.init_priors)
        ax.plot(x,self.distribution_data, label = 'Projection Data')
        ax.plot(x,y_fit, label = 'Initial Guess Fit Data')
        ax.set_xlabel('x')
        ax.set_ylabel('Forward(x)')
        ax.set_title('Initial Fit Guess')

        fig.tight_layout()
        '''

    

