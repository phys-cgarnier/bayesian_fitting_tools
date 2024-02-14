from abc import ABC, abstractmethod
from typing import Union,List,Type,TypeVar
import numpy as np
from matplotlib import patches, pyplot as plt 
from pydantic import BaseModel,PositiveFloat,ConfigDict
from roi import CircularROI, RectangularROI, ROI
import os


class ImageProcessor(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    roi: ROI 
    background_image: np.ndarray = None
    threshold: PositiveFloat = 0.0
    subtract_background_flag: bool = True
    visualize: bool = True

    # needs docstrings
    def subtract_background(self,raw_image:np.ndarray)->np.ndarray:
        #if self.background_image is not None:
            #print(self.background_image)
            #image = raw_image - np.load(self.background_image)
            #pass
        #else:
        image = np.clip(raw_image-self.threshold,0,1e7)
        return image
       
    def process(self,raw_image:np.ndarray)->np.ndarray:
        processed_image = self.subtract_background(raw_image)
        if self.roi is not None:
            processed_image =self.roi.crop_image(processed_image)
        if self.visualize:
            fig, ax = plt.subplots()
            c = ax.imshow(raw_image>0, origin="lower")
            rect = self.roi.get_patch()
            ax.add_patch(rect)
            fig.colorbar(c)
        return processed_image
    
    # needs read h5 file or update property to not load from file
    @property
    def background_image(self) -> Union[np.ndarray, float]:
        return self._background_image

    @background_image.setter
    def background_image(self,background_image):
        self._background_image = background_image
