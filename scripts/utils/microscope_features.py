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


class Holomonitor:
    def __init__(self):
        # Name
        self.name = "holomonitor"

        # Experimental parameters
        self.frame_to_h = 1 / 12
        self.pix_to_um  = 567 / 1024
        self.h_to_um    = 1 / 100

        # Tracking parameters
        self.A_scale = 50
        self.memory = 5
        self.threshold    = 6 
        self.search_range = 20  # 20 in segment_cells, 10 in track_cells



class Tomocube:
    def __init__(self):
        # Name
        self.name = "tomocube"

        # Experimental parameters
        self.frame_to_h    = 1 / 4
        self.pix_to_um     = 0.155433
        self.h_to_um       = 0.8035 / 100
        self.ri_conversion = 1 / 10_000

        # Tracking parameters
        self.A_scale = 150
        self.memory = 3     # 3 in segment_cells, 2 in track_cells
        self.threshold    = 2
        self.search_range = 80    