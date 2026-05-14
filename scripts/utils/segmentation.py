# import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology as morph

from skimage.filters import gaussian
from skimage.segmentation import watershed, clear_border



def smoothen_normalize_im(n_im, s_high, s_low, fig=False):
    '''
    Smoothens image using Gaussian blur and then normalizes it with respect to lowest refractive index within cells.
    I.e. avoids being dominated by empty areas.
    '''
    n_copy = np.copy(n_im)
    n_copy[n_copy == 0] = np.mean(n_copy)
    n_high_pass  = gaussian(n_copy, s_high)
    n_low_pass = gaussian(n_copy, s_low)

    n_norm = n_high_pass - n_low_pass
    n_norm = ((n_norm - n_norm.min()) / (n_norm.max() - n_norm.min()))
    n_norm[n_im == 0] = 0

    # Plots illustration of original image and blurred image
    if fig:
        fig, ax = plt.subplots(1,4, figsize=(12, 4))
        ax[0].imshow(n_im.T, origin="lower", vmin=1.33)
        ax[1].imshow(n_high_pass.T, origin="lower")
        ax[2].imshow(n_low_pass.T, origin="lower")
        ax[3].imshow(n_norm.T, origin="lower")

        ax[0].set(title="raw image")
        ax[1].set(title=f"sigma = {s_high}")
        ax[2].set(title=f"sigma = {s_low}")
        ax[3].set(title="high - low")
        fig.tight_layout()
        #fig.savefig("normalization.png")

    return n_norm



def generate_seed_mask(pos, _shape):
    '''
    Turns list of cell positions into matrix with cell labels.
    Used as seeds for watershed.
    '''
    seeds = np.zeros(_shape, dtype=int)
    i = 1
    for x, y in pos:
        seeds[x,y] = i
        i += 1

    return seeds



def get_cell_areas(im, pos, h_im, clear_edge=True):
    '''
    Uses watershed to obtain mask of labeled cell areas
    '''

    # get cell areas with watershed
    seeds = generate_seed_mask(pos, im.shape)
    areas = watershed(im, seeds, watershed_line=False, connectivity=1)


    # remove empty areas
    cell_mask = (h_im > 0)
    cell_areas = areas*cell_mask

    # remove small holes and areas
    # cell_areas = morph.remove_small_holes(cell_areas, area_threshold=100)

    if clear_edge:
        cell_areas = clear_border(cell_areas)

    return cell_areas
