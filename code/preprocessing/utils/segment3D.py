'''
Functions for transforming Tomocube data to 'Holomonitor data', i.e. 2D tiffs of heights and mean refractive indices
'''

import numpy as np
from skimage.morphology import disk


def get_voxel_size_35mm():
    ''' 
    Returns the spacings from what corresponds to 35mm dish (based on Thomas' assumption) 
    '''
    
    return np.array([0.946946, 0.155433, 0.155433])


def split_tiles(stack, tiles, xsize=912, Nx=4):
    '''
    Split stack into tiles
    '''
    for ix in range(Nx):
        for iy in range(Nx):
            tiles[ix, iy] = stack[:, xsize*iy:xsize*(1+iy), xsize*ix:xsize*(1+ix)]

    return tiles


def scale_refractive(n_z):
    n_cell = 1.38
    scaling_factor = n_cell / (np.mean(n_z[n_z > 0]))

    return n_z * scaling_factor



def estimate_cell_bottom(dn_dz):
    '''
    Estimates first z-slice with cells.
    Assumes it is where derivative of refractive index is max.
    dn_dz = np.diff(np.mean(n, axis=(1,2))), i.e. the derivative along z of the mean refractive index of each stack
    '''

    #dn_dz_mean = np.mean(dn_dz, axis=2)
    #dn_ydz_mean = np.mean(dn_ydz, axis=0)

    z0 = np.argmax(dn_dz, axis=-1)
    #z0_y = np.argmax(dn_ydz_mean, axis=0)
    #z0_X, z0_Y = np.meshgrid(z0_x, z0_y)
    #z0 = (z0_X + z0_Y) / 2

    return z0



def determine_threshold(thresholds, sum_mask):
    '''
    Determine threshold to be used to distinguish cell from media.
    Uses threshold that minimizes magnitude of derivative of cell mask.
    '''

    centered_thresholds = (thresholds[1:] + thresholds[:-1]) / 2
    dsum_mask = np.diff(sum_mask)
    idx = np.argmin(abs(dsum_mask))

    return centered_thresholds[idx]



def generate_kernel(r_min, r_max):
    '''
    Creates 3D kernel that is used for first round of filtering of the cell mask.
    '''
    r_mid = r_min + int((r_max-r_min) / 2)
    p_min = r_max - r_min
    p_mid = r_max - r_mid

    kernel = np.array([np.pad(disk(r_min), pad_width=p_min),
                       np.pad(disk(r_mid), pad_width=p_mid),
                       disk(r_max),
                       np.pad(disk(r_mid), pad_width=p_mid),
                       np.pad(disk(r_min), pad_width=p_min)])
    
    return kernel



def compute_height(cell_pred, method="sum"):
    '''
    Computes cell heights either by summing voxels or taking the difference between min and max.
    Assumes prediction voxels are 0 or 1. Returns height in units of voxels.
    '''
    assert method=="sum" or method=="diff"
    assert np.max(cell_pred) == 1

    if method=="sum":
        h = np.sum(cell_pred, axis=0)

    elif method=="diff":
        _, Z_idx, _ = np.meshgrid(np.arange(0, len(cell_pred[0])),
                                  np.arange(0, len(cell_pred)), 
                                  np.arange(0, len(cell_pred[0,0])))
        Z_idx = Z_idx * cell_pred
        Z_idx[Z_idx==0] = np.nan

        h = np.nanmax(Z_idx, axis=0) - np.nanmin(Z_idx, axis=0) + 1

    return h



def refractive_index_uint16(stack, mask, height):
    '''
    Transforms float16 to uint16. Used on refractive indices to save as uint16.
    Returns sum and mean
    '''

    # Take average
    ridx_sum  = np.sum(stack * mask, axis=0)
    ridx_avrg = np.copy(ridx_sum)
    ridx_avrg[height > 0]  = ridx_avrg[height > 0] / height[height > 0]

    # remove empty regions
    ridx_avrg[height <= 0] = 0

    return np.array(ridx_avrg, dtype=np.uint16)