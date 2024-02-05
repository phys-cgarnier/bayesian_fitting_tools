
import numpy as np
from matplotlib import patches, pyplot as plt 
from typing import Union,List,Type,TypeVar
from pydantic import BaseModel, PositiveFloat, ConfigDict
from gaussian_model import GaussianModel
import scipy.optimize

# work in progress


class ProjectionFit(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    # should add getter/setter methods otherwise there is no reason to call model_setup externally
    distribution_data : np.ndarray
    model : GaussianModel 
    visualize: bool = True
    use_priors: bool = True

    def get_projections(self):
        # should have projection being passed
        # no need for this
        pass

    def normalize(self,old_data:np.ndarray)->np.ndarray:
        # normalize 
        data = old_data.copy()
        dmax = np.max(data)
        assert len(data.shape) == 1
        normalized_data = data/dmax
        return normalized_data

    def unnormalize(self,xfmd_data:np.ndarray,dmax:float)->np.ndarray:
        old_data = xfmd_data*dmax
        return old_data
    
    def model_setup(self)->None:
        normalized_data = self.normalize(self.distribution_data)
        self.model.distribution_data = normalized_data
        if self.visualize:
            self.model.plot_priors()

    def fit_model(self)->scipy.optimize._optimize.OptimizeResult:
        x = np.linspace(0,1,len(self.model.distribution_data))
        y = self.model.distribution_data
        res =  scipy.optimize.minimize(self.model.loss, self.model.param_guesses,
                                    args=(x, y, self.use_priors),
                                    bounds=self.model.param_bounds)

        # need to make this function model dependent
        if self.visualize:
            fig, ax = plt.subplots(figsize = (10,5))
            y_fit = self.model.forward(x,res.x)
            ax.plot(x,y, label='data')
            ax.plot(x, y_fit, label='fit')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.legend()
            
            textstr = '\n'.join((
                r'$\mathrm{A}=%.2f$' % (res.x[0], ),
                r'$\mu=%.2f$' % (res.x[1], ),
                r'$\sigma=%.2f$' % (res.x[2], ),r'$\mathrm{offset}=%.2f$' % (res.x[3], )))

            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

            # place a text box in upper left in axes coords
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)
        return res