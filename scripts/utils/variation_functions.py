import sys
import json
import imageio
import numpy as np

from tqdm import tqdm
# from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter
# from skimage.morphology import disk

sys.path.append("scripts/utils")
from Microscopes import Holomonitor, Tomocube
from SegmentedCells import SegmentedCells
from file_operations import load_set_of_frames
# from scripts.utils.matrix_operations import global_density, average_cell_radius

data_path = "../../../../hdd_data/silja/Monolayers/"

# def spatial_variation(data):

#     mean = np.mean(data[data > 0])
#     rel_err = np.std(data[data > 0]) / mean

#     return rel_err, mean



def temporal_variation(data):
    mean    = np.ma.mean(data, axis=0)
    rel_err = np.ma.std(data,  axis=0) / mean

    return np.ma.mean(rel_err), np.ma.std(rel_err) / np.ma.sqrt(np.ma.sum(~rel_err.mask)), np.ma.mean(mean)



# compute variation
def compute_temporal_variation(datasets, measure, parameter, dt=1, dT=4):

    mean_value = []
    variation_mean = []
    variation_std  = []
    cell_densities = []

    # loop through datasets
    for dataset in datasets:

        config = json.load(open(f"configs/{dataset}.json"))
        data   = SegmentedCells(f"{data_path}/cell_features/calibrated/{dataset}_cells.p")

        ### SET MICROSCOPE ###
        if config['microscope'] == "holomonitor": microscope = Holomonitor()
        elif config['microscope'] == "tomocube":  microscope = Tomocube()

        dt_i = int(dt / microscope.frame_to_h)
        dT_i = int(dT / microscope.frame_to_h)


        # import segmented data
        cells   = SegmentedCells(f"{data_path}/cell_features/calibrated/{dataset}_cells.p")

        # loop through frames and compute spatial variation
        rho_tmp = []
        h_mean_tmp = []
        h_var_tmp  = []
        h_varerr_tmp  = []

        try:
            mask = imageio.v2.imread(f"{data_path}masks/{dataset}_mask.tiff")
            mask = mask > 0
        except:
            mask = 1

        fmin = config['cells']['fmin']
        fmax = config['cells']['fmax'] + 1

        # looping through initial frame
        for fstart in tqdm(range(fmin, fmax-dT_i, dt_i)):

            r_cell = 10**3 / (microscope.pix_to_um*np.sqrt(cells.density[int(fstart+dt_i/2)-fmin]*np.pi))
            sigma = microscope.rblur * r_cell

            im = []
            for f in range(fstart, fstart + dT_i):

                if measure == "pixel":
                    # pixel wise measure only defined for heights
                    h, _ = load_set_of_frames(f"{data_path}height_fields/calibrated/{dataset}", f, microscope)
                    h = h * mask
                    im.append(np.ma.array(h, mask=h==0))

                elif measure == "disc":

                    # bluring heights
                    h, _ = load_set_of_frames(f"{data_path}height_fields/calibrated/{dataset}", f, microscope)

                    #mask = mask * np.ones_like(h)
                    if np.sum(mask) > 1:
                        h[mask == 0] = np.mean(h[mask > 0])
                    h = gaussian_filter(h, int(sigma)) * mask
                    im.append(np.ma.array(h, mask=h==0))
                    
            if measure == "pixel" or measure == "disc":
                im = np.array(im)
                im = np.ma.array(im, mask=im==0)

            elif measure == "cell":
                if parameter == "h":
                    im = data.h[fstart - fmin: fstart - fmin + dT_i]
                if parameter == "n":
                    im = data.n[fstart - fmin: fstart - fmin + dT_i]-Tomocube().n0
                elif parameter == "A":
                    im = data.A[fstart - fmin: fstart - fmin + dT_i]
                elif parameter == "V":
                    im = data.h[fstart - fmin: fstart - fmin + dT_i] * data.A[fstart - fmin: fstart - fmin + dT_i]
                im = np.ma.mask_cols(im)


            h_var, h_varerr, h_mean = temporal_variation(im)

            h_mean_tmp.append(h_mean)
            h_var_tmp.append(h_var)
            h_varerr_tmp.append(h_varerr)
            rho_tmp.append(np.mean(cells.density[fstart-fmin:fstart-fmin+dT_i]))

        mean_value.append(h_mean_tmp)
        variation_mean.append(h_var_tmp)
        variation_std.append(h_varerr_tmp)
        cell_densities.append(rho_tmp)
            
    return variation_mean, variation_std, mean_value, cell_densities