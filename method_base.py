from abc import ABC, abstractmethod
import numpy as np

class MethodBase(ABC):
    def __init__(self):
        init_values: list = None
        
    @abstractmethod
    def find_init_values(self):
        pass

    @abstractmethod
    def find_priors(self):
        pass

    @abstractmethod
    def plot_init_values(self):
        pass

    @abstractmethod
    def plot_priors(self):
        pass

    @staticmethod
    @abstractmethod
    def forward(x:np.array, params:np.array)->np.array:
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
    # priors not init priors and move more stuff into base
    def priors(self):
        """Initial Priors store in a dictionary where the keys are the complete set of parameters of the Model"""
        return self._priors 
    
    @priors.setter
    def priors(self,priors):
        if not isinstance(priors,dict):
            raise TypeError("Input must be a dictionary")
        self._priors = priors   

    @property
    def distribution_data(self):
        """1d array typically projection data"""
        return self._distribution_data

    @distribution_data.setter
    def distribution_data(self,distribution_data):
        if not isinstance(distribution_data, np.ndarray):
            raise TypeError("Input must be ndarray")
        self._distribution_data = distribution_data
        self.find_priors(self._distribution_data)