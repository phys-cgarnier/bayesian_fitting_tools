
import numpy as np
from matplotlib import patches, pyplot as plt 
from typing import Union,List,Type,TypeVar
from pydantic import BaseModel, PositiveFloat, ConfigDict
from gaussian_model import GaussianModel
from method_base import MethodBase
import scipy.optimize

# work in progress


class ProjectionFit(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model : MethodBase 
    visualize: bool = True #False
    use_priors: bool = True #False


    def normalize(self,old_data:np.ndarray)->np.ndarray:
        data = old_data.copy()
        normalized_data = data/(np.max(data))
        return normalized_data

    def unnormalize_model_params(self,params_dict: dict[str,float],projection_data:np.ndarray)->np.ndarray:
        max_value = np.max(projection_data)
        length = len(projection_data)

        for key, val in params_dict.items():
            if 'sigma' in key or 'mean' in key:
                true_fitted_val = val*length
            else: 
                true_fitted_val = val * max_value # hmm.... maybe this should not be callable externally the model needs normalized data only
        self.model.distribution_data = projection_data
        if self.visualize:
            self.model.plot_priors()

    def fit_model(self)->scipy.optimize._optimize.OptimizeResult:
        x = np.linspace(0,1,len(self.model.distribution_data))
        y = self.model.distribution_data
        
        res =  scipy.optimize.minimize(self.model.loss, self.model.param_guesses,
                                    args=(x, y, self.use_priors),
                                    bounds=self.model.param_bounds)
        
        if self.visualize:
            fig, ax = plt.subplots(figsize = (10,5))
            y_fit = self.model.forward(x,res.x)
            ax.plot(x,y, label='data')
            ax.plot(x, y_fit, label='fit')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.legend(loc= 'upper right')
            assert len(self.model.param_names) == len(res.x)
            textstr = '\n'.join([r'$\mathrm{%s}=%.2f$'%(self.model.param_names[i],res.x[i]) for i in range(len(res.x))])
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)
        return res
    
    def fit_projection(self,projection_data:np.ndarray)->dict[str,float]:
        assert len(projection_data.shape) == 1
        fitted_params_dict = {}
        normalized_data =  self.normalize(projection_data)
        self.model_setup(projection_data=normalized_data)
        res = self.fit_model()
        for i, param in enumerate(self.model.param_names):
            fitted_params_dict[param] = (res.x)[i]
        print(fitted_params_dict)
        params_dict = self.unnormalize_model_params(fitted_params_dict,projection_data)
        return params_dict
            temp = {key:true_fitted_val}
            params_dict.update(temp)
        return params_dict 
    
    def model_setup(self,projection_data=np.array)->None:
        # hmm.... maybe this should not be callable externally the model needs normalized data only
        self.model.distribution_data = projection_data
        if self.visualize:
            self.model.plot_priors()

    def fit_model(self)->scipy.optimize._optimize.OptimizeResult:
        x = np.linspace(0,1,len(self.model.distribution_data))
        y = self.model.distribution_data
        
        res =  scipy.optimize.minimize(self.model.loss, self.model.param_guesses,
                                    args=(x, y, self.use_priors),
                                    bounds=self.model.param_bounds)
        
        if self.visualize:
            fig, ax = plt.subplots(figsize = (10,5))
            y_fit = self.model.forward(x,res.x)
            ax.plot(x,y, label='data')
            ax.plot(x, y_fit, label='fit')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.legend(loc= 'upper right')
            assert len(self.model.param_names) == len(res.x)
            textstr = '\n'.join([r'$\mathrm{%s}=%.2f$'%(self.model.param_names[i],res.x[i]) for i in range(len(res.x))])
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)
        return res
    
    def fit_projection(self,projection_data:np.ndarray)->dict[str,float]:
        assert len(projection_data.shape) == 1
        fitted_params_dict = {}
        normalized_data =  self.normalize(projection_data)
        self.model_setup(projection_data=normalized_data)
        res = self.fit_model()
        for i, param in enumerate(self.model.param_names):
            fitted_params_dict[param] = (res.x)[i]
        print(fitted_params_dict)
        params_dict = self.unnormalize_model_params(fitted_params_dict,projection_data)
        return params_dict
