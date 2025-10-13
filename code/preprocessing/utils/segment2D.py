import scipy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import Voronoi
from skimage import measure, draw
from skimage import morphology as morph

from skimage.filters import gaussian
from skimage.segmentation import watershed, clear_border

n0 = 1.33
nd = 1.4
n_cell = 1.38

def get_pixel_size():
    ''' 
    From Nigar's thesis 
    '''

    return np.array([567 / 1024, 567 / 1024])


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


def extendedmin(im, H):
    '''
    Inverse of MATLABs imextendedmax: https://se.mathworks.com/help/images/ref/imextendedmax.html
    '''
    mask   = im.copy() 
    marker = mask + H  
    hmin   =  morph.reconstruction(marker, mask, method='erosion')

    return morph.local_minima(hmin)


def find_cell_pos(im, H, min_size=20):
    '''
    Finds cell positions as centroid of extendedmin
    '''
    
    im_min = extendedmin(im, H)

    # remove very small maxima
    im_min = morph.remove_small_objects(im_min, min_size=min_size) 


    im_label = measure.label(im_min)
    reg_prop = measure.regionprops(im_label, im)
    pos = np.array([[int(r.centroid[0]), int(r.centroid[1])] for r in reg_prop])

    return pos


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


def update_pos(pos, labels):
    n = np.max([np.max(labels)+1, len(pos)])
    new_pos = np.zeros([n, 2])

    for x, y in pos:
        l = labels[x,y]
        new_pos[l] = [x,y]
    
    x = new_pos.T[0] #[new_pos.T[0] > 0]
    y = new_pos.T[1] #[new_pos.T[1] > 0]

    return np.array([x[1:],y[1:]], dtype=int).T


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
    #cell_areas = morph.remove_small_holes(cell_areas, area_threshold=100)

    if clear_edge:
        cell_areas = clear_border(cell_areas)

    return cell_areas


# def compute_polarization(reg_props):
#     '''
#     Computes radius and angle of polarization
#     Also returns planar aspect ratio and 1 / a_major for computation of normal aspect ratio 
#     '''
#     a_major = [reg.axis_major_length for reg in reg_props]
#     a_minor = [reg.axis_minor_length for reg in reg_props]
#     theta  = [reg.orientation for reg in reg_props]
#     radius = [(a_max-a_min) / np.sqrt(a_max**2+a_min**2) for a_max, a_min in zip(a_major, a_minor)]

#     return radius, theta, a_minor / a_major, 1 / a_major


def compute_cell_props(label_im, pos, h_im, n_im, xy_to_um, type='holomonitor'):
    '''
    Uses mask of labeled areas to compute cell position, area, volume and mass
    '''
    # Set conversion from pixel to um depending on type of data
    labels = []
    minor_axis, major_axis = [], []
    h_mean, h_max, n_mean  = [], [], []
    area, perimeter, volume = [], [], []
    orientation, eccentricity = [], []

    reg_prop = measure.regionprops(label_im, n_im)

    i = 0
    for l in range(label_im.max()):
        label = l+1
        mask = (label_im == label)

        # skip labels associated with empty areas 
        if np.sum(mask) == 0:
            continue

        # update position
        if np.all(pos[l]) == 0:
            x, y = reg_prop[i].centroid_weighted
            pos[l] = int(x), int(y)

        # specific to Tomocube data
        if type=='tomocube':
            n_mean.append(np.sum(mask*n_im) / np.sum(mask))

        area.append(np.sum(mask))
        h_mean.append(np.sum(mask*h_im) / np.sum(mask))
        minor_axis.append(reg_prop[i].axis_minor_length)
        major_axis.append(reg_prop[i].axis_major_length)
        orientation.append(reg_prop[i].orientation)
        eccentricity.append(reg_prop[i].eccentricity)

        labels.append(label)    # why not l?
        h_max.append(np.max(mask*h_im))
        volume.append(np.sum(mask * h_im))
        perimeter.append(reg_prop[i].perimeter)

        i += 1

    mask = (np.all(pos, axis=1) > 0)
    cells_tmp = pd.DataFrame({'x': np.array(pos.T[0][mask]) * xy_to_um,
                              'y': np.array(pos.T[1][mask]) * xy_to_um,
                              'P': np.array(perimeter)      * xy_to_um,
                              'A': np.array(area)           * xy_to_um**2, 
                              'V': np.array(volume)         * xy_to_um**2,
                              'a_min': np.array(minor_axis) * xy_to_um,
                              'a_max': np.array(major_axis) * xy_to_um,
                              'theta': np.array(orientation),
                              'ecc':   np.array(eccentricity),
                              'h_avrg': np.array(h_mean),
                              'n_avrg': np.array(np.zeros_like(h_mean)),
                              'h_max': np.array(h_max),
                              'label': np.array(labels)
                            })


    # Tomocube data
    if type=='tomocube':
        cells_tmp['n_avrg'] = n_mean
    
    return cells_tmp


def get_voronoi_ridges(pos):
    '''
    Copied from scipy.spatial.plot_voronoi_2d source code: 
    https://github.com/scipy/scipy/blob/v1.15.1/scipy/spatial/_plotutils.py#L0-L1
    '''
    vor = Voronoi(pos)
    center = vor.points.mean(axis=0)
    ptp_bound = np.ptp(vor.points, axis=0)

    voronoi_ridges = []
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            voronoi_ridges.append(vor.vertices[simplex])
        else:
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            if (vor.furthest_site):
                direction = -direction
            aspect_factor = abs(ptp_bound.max() / ptp_bound.min())
            far_point = vor.vertices[i] + direction * ptp_bound.max() * aspect_factor

            voronoi_ridges.append([vor.vertices[i], far_point])

    return voronoi_ridges


def get_voronoi_areas(labels, pos, h_im, w_edge=4, area_threshold=200, clear_edge=True):
    dims = np.shape(h_im)
    voronoi_ridges = get_voronoi_ridges(pos)
    raw_areas = np.ones_like(h_im, dtype=np.uint8)

    for ridge in voronoi_ridges:
        ridge = np.array(ridge, dtype=int)# + w_pad
        row, col = draw.line(*ridge.ravel())
        mask = (row >= 0) * (row < dims[0]) * (col >= 0) * (col < dims[1])
        raw_areas[row[mask], col[mask]] = 0
    
    # make edges thicker
    raw_edges = (raw_areas == 0)
    edges = morph.dilation(raw_edges, morph.disk(w_edge))
    areas = (edges == 0)

    # remove empty areas
    cell_mask = (h_im > 0)
    cell_areas = areas*cell_mask

    # remove small areas not corresponding to cell pos
    cell_areas = morph.remove_small_objects(cell_areas, min_size=area_threshold)
    if clear_edge: 
        cell_areas = clear_border(cell_areas)

    # label areas, taking same labels as watershed areas
    labels_tmp = measure.label(cell_areas)
    labels = morph.dilation(labels, morph.disk(2*w_edge))*(labels_tmp > 0)
    areas = np.zeros_like(labels_tmp)

    for x, y in pos:
        x = int(np.clip(x, 0, dims[0]-1))
        y = int(np.clip(y, 0, dims[1]-1))

        l_new = labels[x,y]
        l_old = labels_tmp[x,y]
        areas[labels_tmp == l_old] = int(l_new)

    return areas


def compute_volume_change(df):
    '''
    Compute change in volume of tracked cells
    '''
    df = df.sort_values(by=['particle', 'frame'])
    df['dV'] = df.groupby('particle')['V'].diff()
    df['dV'] = df['dV'].fillna(0)

    df['dV'] /= df['V']

    return df


