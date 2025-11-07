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



def split_tiles(stack, tile_size=850, full_tile_size=958, Ntiles=4):
    '''
    Split a 3D stack into tiles with edge removed.

    Parameters:
    - stack: 3D numpy array of shape (Nz, Nxy, Nxy), where:
        - Nz: Number of slices in the z-dimension.
        - Nxy: Size of the xy dimensions.
    - tile_size: The size of the tile to extract (default is 850).
    - full_tile_size: The size of the full tile before edge removal (default is 958).
    - Ntiles: The number of tiles to create along each axis (default is 4).

    Returns:
    - tiles: A 5D numpy array of shape (Ntiles, Ntiles, Nz, tile_size, tile_size),
             containing the extracted tiles.
    '''

    # Get dimensions of the input stack
    Nz, Nxy, _ = np.shape(stack)  # Extract the number of z slices and size of xy dimension
    dxy = int(tile_size / 2)      # Calculate half the tile size for centered indexing

    # Define the edge indices for extracting tiles ( centering the extraction within bounds )
    xy_edge = np.array([0, Nxy / 2 - full_tile_size, Nxy / 2, Nxy / 2 + full_tile_size, Nxy], dtype=int)
    
    # Initialize an array to hold the extracted tiles
    tiles = np.zeros([Ntiles, Ntiles, Nz, tile_size, tile_size])

    # Loop over each tile's position
    for ix in range(Ntiles):
        for iy in range(Ntiles):
            # Extract a tile from the stack using the defined edge indices
            tile = stack[:, xy_edge[iy]:xy_edge[1 + iy], xy_edge[ix]:xy_edge[1 + ix]]
            
            # Get the current tile's dimensions
            dims = np.shape(tile)
            xmid = int(dims[2] / 2)  # Calculate the mid-point for x dimension
            ymid = int(dims[1] / 2)  # Calculate the mid-point for y dimension

            # Center the tile by removing the edges using calculated half-widths (dxy)
            tiles[iy, ix] = tile[:, ymid - dxy:ymid + dxy, xmid - dxy:xmid + dxy]

    return tiles  # Return the array of tiles



def combine_tiles(tile, weight_tile, Nxy=3648, full_tile_size=958, Ntiles=4):
    '''
    Combine tiles into the original stack
    '''

    # Define the edge indices for extracting tiles ( centering the extraction within bounds )
    xy_edge = np.array([0, Nxy / 2 - full_tile_size, Nxy / 2, Nxy / 2 + full_tile_size, Nxy], dtype=int)
    
    # Calculate the dimensions of the full stack
    full_image = np.zeros([Nxy, Nxy])  # Initialize the full stack array
    mid = int(full_tile_size / 2)

    # Loop through each tile to combine them into the full stack
    for ix in range(Ntiles):
        for iy in range(Ntiles):

            # Get the current tile's dimensions
            dx = int((xy_edge[ix+1] - xy_edge[ix]) / 2)
            dy = int((xy_edge[iy+1] - xy_edge[iy]) / 2)

            # Center the tile by removing the edges using calculated half-widths (dxy)
            z0_plane = tile[mid-dy:mid+dy, mid-dx:mid+dx]
            w = weight_tile[iy, ix]
            full_image[xy_edge[iy]:xy_edge[iy+1], xy_edge[ix]:xy_edge[ix+1]] = z0_plane * w

    return full_image



def scale_refractive(n_z):
    n_cell = 1.38
    scaling_factor = n_cell / (np.mean(n_z[n_z > 0]))

    return n_z * scaling_factor



def estimate_cell_bottom(stack):
    '''
    Estimates first z-slice with cells.
    Assumes it is where derivative of refractive index is max.
    dn_dz = np.diff(np.mean(n, axis=(1,2))), i.e. the derivative along z of the mean refractive index of each stack
    '''

    n_z   = np.mean(stack, axis=(1,2))
    dn_dz = np.diff(n_z) + 1
    z0    = np.argmax(dn_dz / n_z[:-1])

    return z0


def estimate_cell_bottom_reference(mask, f_crit=0.5):
    dims = np.shape(mask)

    # compute fraction of voxels that are cell along z-axis
    f_cells = np.sum(mask, axis=(1,2)) / (dims[1] * dims[2])
    zslice  = np.arange(len(f_cells))

    # use first z-slice above f_crit as reference
    zslice_mask = zslice[f_cells > f_crit]
    z0 = zslice_mask[np.argmin(zslice_mask)]

    return z0



def update_cell_mask(mask, z0, z0_plane):

    new_mask = np.copy(mask).astype(float)
    new_mask[z0] = 1
    new_mask[:z0] = 0
    new_mask[z0-1] = z0_plane
    new_mask[z0-2] = (z0_plane-1) * (z0_plane > 1)

    return new_mask



# def correct_zslice_tile(tile, z0_tile, z0_median):
#     '''
#     Corrects zslice of tile so it has same median as z0_median

#     tile:      stack of tile
#     z0_tile:   median zero-level for tile
#     z0_median: median zero-level for full stack
#     '''
    
#     z_diff = int(z0_median - z0_tile)
#     z_pad = abs(z_diff)

#     if z_pad > 0:
#         npad = ((z_pad, z_pad), 
#                 (0, 0), 
#                 (0, 0))
        
#         tile_padded = np.pad(tile, pad_width=npad, mode="edge")
#         tile_zcorr  = np.roll(tile_padded, shift=z_diff, axis=0)[z_pad:-z_pad]

#     else:
#         tile_zcorr = tile

#     return tile_zcorr



# Fit plane to cell bottom
def plane(params, x, y):
    a, b, c = params
    return a*x + b*y + c

def residual_plane(params, X, Y, Z):
    return Z - plane(params, X, Y)


def fit_plane(Z0, full_dims):
    '''
    Fit linear plane to z0 data points and returning plane for full tile
    '''

    dims = np.shape(Z0)
    Y, X = np.meshgrid(np.arange(dims[0]), 
                       np.arange(dims[1]))

    # tile without edges
    X = X.flatten()
    Y = Y.flatten()
    Z = Z0.flatten()

    flat_params = np.array([0,0, np.mean(Z0)])
    result = sc.optimize.least_squares(residual_plane, flat_params, args=(X, Y, Z))

    # tile with edges
    dim_diff = int(full_dims[0] - dims[0]) / 2
    Y, X = np.meshgrid(np.arange(full_dims[0]) - dim_diff, 
                       np.arange(full_dims[1]) - dim_diff)

    X = X.flatten()
    Y = Y.flatten()

    z_plane = plane(result.x, X, Y)

    return z_plane.reshape(full_dims[0], full_dims[1])



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