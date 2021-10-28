## Basic Python imports
import math
import random 

## PyTorch imports
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF 

## Image / Array imports
from PIL import Image
from PIL import ImageFilter
import numpy as np 

class RotationTransform(object):
    """
    Rotates a PIL image by 0, 90, 180, or 270 degrees. Randomly chosen using a uniform distribution.
    """
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, image):
        angle = random.choice(self.angles)
        return TF.rotate(image, angle)

class GammaJitter(object):
    """
    Jitters the gamma of a PIL image between a uniform distribution of two values (low & high).
    Larger gammas make the shadows darker, smaller gammas make the shadows lighter.
    """
    def __init__(self, low=0.9, high=1.1):
        self.low = low
        self.high = high
    
    def __call__(self, image):
        gamma = np.random.uniform(self.low, self.high)
        return TF.adjust_gamma(image, gamma)

class BrightnessJitter(object):
    """
    Jitters the gamma of a PIL image between a uniform distribution of two values (low & high).
    Larger gammas make the shadows darker, smaller gammas make the shadows lighter.
    """
    def __init__(self, low=0.9, high=1.1):
        self.low = low
        self.high = high
    
    def __call__(self, image):
        factor = np.random.uniform(self.low, self.high)
        return TF.adjust_brightness(image, factor)

class RandomScale(object):
    """
    Scales a PIL image based on a value chosen from a uniform distribution of two values (low & high).
    """
    def __init__(self, low=1.0, high=1.1):
        self.low = low
        self.high = high

    def __call__(self, image):
        height = image.height
        width = image.width
        scale = np.random.uniform(self.low, self.high)
        image = TF.resize(image, (math.floor(height * scale), math.floor(width * scale)))
        return TF.center_crop(image, (height, width))

class Resize(object):
    """
    Scales a PIL image based on a value chosen from a uniform distribution of two values (low & high).
    """
    def __init__(self, resolution=1, size=0):
        self.resolution = resolution
        self.size = size

    def __call__(self, image):
        if self.size == 0:
            height = image.height
            width = image.width
            return image.resize((width // self.resolution, height // self.resolution))
        else:
            return image.resize((self.size, self.size), Image.BILINEAR)

class AdaptiveCenterCrop(object):
    """
    Center crops the image to be a square of the smallest edge squared.
    """
    def __init__(self):
        pass

    def __call__(self, image):
        length = min(image.width, image.height)
        return TF.center_crop(image, (length, length))

class MedianFilter(object):
    """
    Randomly applies a median filter to an image.
    """
    def __init__(self, filter_size=3, p=0.1):
        self.filter_size = filter_size
        self.p = p

    def __call__(self, image):
        roll = random.random()
        if roll < self.p:
            return image.filter(ImageFilter.MedianFilter(self.filter_size))
        else:
            return image
