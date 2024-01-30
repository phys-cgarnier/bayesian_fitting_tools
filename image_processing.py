from abc import ABC, abstractmethod
from typing import Union,List,Type,TypeVar
import numpy as np
from matplotlib import patches, pyplot as plt 
from pydantic import BaseModel,PositiveFloat
from roi import CircularROI, RectangularROI, ROI
import os

roi_var = TypeVar('roi_var', bound='ROI')

class ImageProcessor(BaseModel):
    roi_type: Type[roi_var]
    background_file: str = None
    threshold: PositiveFloat = 0.0
    subtract_background_flag: bool = True
    visualize: bool = True

    def subtract_background(self,raw_image):
        if self.background_file is not None:
            image = raw_image - self.background_image
        else:
            image = np.clip(raw_image-self.threshold,0,1e7)
        return image
       
    def process(self,raw_image):
        processed_image = self.subtract_background(raw_image)
        # processed_image = roi.crop_image(processed_image)
        if self.visualize:
            fig, ax = plt.subplots()
            c = ax.imshow(processed_image, origin="lower")
            fig.colorbar(c)
        return processed_image
    
    # this method is useless without reading from an h5pyfile
    @property
    def background_image(self) -> Union[np.ndarray, float]:
        if self.background_file is not None:
            return np.load(self.background_file)
        else:
            return 0.0

    @background_image.setter
    def background_image(self):
        pass
