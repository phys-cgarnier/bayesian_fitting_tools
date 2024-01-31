from abc import ABC, abstractmethod
from typing import Union,List,Type,TypeVar
import numpy as np
from matplotlib import patches, pyplot as plt 
from pydantic import BaseModel,PositiveFloat
from roi import CircularROI, RectangularROI, ROI
import os


class ImageProcessor(BaseModel):
    roi_obj: ROI 
    background_file: str = None
    threshold: PositiveFloat = 0.0
    subtract_background_flag: bool = True
    visualize: bool = True

    # needs docstrings
    def subtract_background(self,raw_image):
        if self.background_file is not None:
            image = raw_image - self.background_image
        else:
            image = np.clip(raw_image-self.threshold,0,1e7)
        return image
       
    def process(self,raw_image):
        processed_image = self.subtract_background(raw_image)
        if self.roi_obj is not None:
            processed_image =self.roi_obj.crop_image(processed_image)
        if self.visualize:
            fig, ax = plt.subplots()
            c = ax.imshow(raw_image>0, origin="lower")
            rect = self.roi_obj.get_patch()
            ax.add_patch(rect)
            fig.colorbar(c)
        return processed_image
    
    # needs read h5 file or update property to not load from file
    @property
    def background_image(self) -> Union[np.ndarray, float]:
        if self.background_file is not None:
            return np.load(self.background_file)
        else:
            return 0.0

    @background_image.setter
    def background_image(self):
        pass
