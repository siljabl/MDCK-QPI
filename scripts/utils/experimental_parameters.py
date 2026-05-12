import numpy as np
import scipy as sc
from skimage.morphology import disk


def get_voxel_size_35mm():
    ''' 
    Returns the spacings from what corresponds to 35mm dish (based on Thomas' assumption) 
    '''

    return np.array([0.8035, 0.155433, 0.155433])


def get_pixel_size():
    ''' 
    From Nigar's thesis 
    '''

    return np.array([567 / 1024, 567 / 1024])
