
import numpy as np
from matplotlib import patches, pyplot as plt 
from typing import Union,List,Type,TypeVar
from pydantic import BaseModel,PositiveFloat
from gaussian_model import GaussianModel

# work in progress


class ProjectionFit(BaseModel):
    model: GaussianModel
    visualize: bool = True

    def get_projections(self):
        pass

    def normalize_1d(self):
        pass

    def unnormalize_1d(self):
        pass

    def fit_model(self):
        pass