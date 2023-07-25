from abc import *
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
import numpy as np


class BaseAugmentation():
    def __init__(self, p=1):
        super(BaseAugmentation, self).__init__()
        
        self.p = p
        
        self.transform = A.Compose(self.get_transform_list(), additional_targets={'image1': 'image', 'image2': 'image', 'image3': 'image', 'image4': 'image'})
        
    @abstractmethod
    def get_transform_list(self) -> list:
        pass
    
    def __call__(self, image, image1, image2, image3, image4):
        return self.transform(image=image, image1=image1, image2=image2, image3=image3, image4=image4)
        

class RandBrightness(BaseAugmentation):
    def __init__(self, p=1):
        super(RandBrightness, self).__init__(p)
    
    def get_transform_list(self):
        return [
            A.RandomBrightnessContrast(brightness_limit=[0.9, 0.9], contrast_limit=[-0.1, 0.3], p=self.p)
        ]
    
class DummyAugmentation(BaseAugmentation):
    def __init__(self, p=1):
        super(DummyAugmentation, self).__init__(p)
    
    def get_transform_list(self):
        return [
            A.RandomBrightnessContrast(brightness_limit=[1.0, 1.0], contrast_limit=[1.0, 1.0], p=self.p)
        ]

class ImgCutTop(ImageOnlyTransform):    
    def __init__(self, always_apply=False, p=1):
        super(ImgCutTop, self).__init__(always_apply, p)
    
    def apply(self, img, **params):
        transformed_img = img.copy()
        transformed_img[:50, :] = np.zeros_like(transformed_img[:50, :])

        return transformed_img