'''
Functions for transforming Tomocube data to 'Holomonitor data', i.e. 2D tiffs of heights and mean refractive indices
'''

import numpy as np
import scipy as sc
from skimage.morphology import disk


def get_voxel_size_35mm():
    ''' 
    Returns the spacings from what corresponds to 35mm dish (based on Thomas' assumption) 
    '''
    
    return np.array([0.946946, 0.155433, 0.155433])


def split_tiles(stack, xsize=912, Nz=87, Nx=4):
    '''
    Split stack into tiles
    '''
    tiles = np.zeros([4, 4, Nz, Nx, Nx])

    for ix in range(Nx):
        for iy in range(Nx):
            tiles[ix, iy] = stack[:, xsize*iy:xsize*(1+iy), xsize*ix:xsize*(1+ix)]

    return tiles


def scale_refractive(n_z):
    n_cell = 1.38
    scaling_factor = n_cell / (np.mean(n_z[n_z > 0]))

    return n_z * scaling_factor



def estimate_cell_bottom(stack, mode="mean"):
    '''
    Estimates first z-slice with cells.
    Assumes it is where derivative of refractive index is max.
    dn_dz = np.diff(np.mean(n, axis=(1,2))), i.e. the derivative along z of the mean refractive index of each stack
    '''
    if mode == "mean":
        n_z   = np.mean(stack, axis=(1,2))
        dn_dz = np.diff(n_z) + 1
        z0    = np.argmax(dn_dz)

    elif mode == "plane":
        dn_dz = np.diff(stack, axis=0) + 1
        z0    = np.argmax(dn_dz, axis=0)

    else:
        print("Error: doesn't recognize mode.")

    return z0


def correct_zslice_tile(tile, z0_median):
    '''
    Corrects zslice of tile so it has same median as z0_median
    '''
    z_diff = int(z0_median - tile)
    z_pad = abs(z_diff)

    if z_pad > 0:
        npad = ((z_pad, z_pad), 
                (0, 0), 
                (0, 0))
        
        tile_padded = np.pad(tile, pad_width=npad, mode="edge")
        tile_zcorr  = np.roll(tile_padded, shift=z_diff, axis=0)[z_pad:-z_pad]

    else:
        tile_zcorr = tile

    return tile_zcorr


def plane(params, x, y):
    a, b, c = params
    return a*x + b*y + c

def residual_plane(params, X, Y, Z):
    return Z - plane(params, X, Y)


def fit_plane(Z0):
    '''
    Fit linear plane to z0 data points
    '''

    dims = np.shape(Z0)
    Y, X = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]))

    X = X.flatten()
    Y = Y.flatten()
    Z = Z0.flatten()

    flat_params = np.array([0,0, np.mean(Z0)])
    result = sc.optimize.least_squares(residual_plane, flat_params, args=(X, Y, Z))

    z_plane = plane(result.x, X, Y)

    return z_plane



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