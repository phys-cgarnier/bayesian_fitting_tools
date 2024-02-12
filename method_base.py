from abc import ABC, abstractmethod
import numpy as np

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
    def init_priors(self):
        """Initial Priors store in a dictionary where the keys are the complete set of parameters of the Model"""
        return self._init_priors 
    
    @init_priors.setter
    def init_priors(self,init_priors):
        if not isinstance(init_priors,dict):
            raise TypeError("Input must be a dictionary")
        self._init_priors = init_priors   