'''
Scipt to compute mean and standard deviation of height (pixel-wise or disc-wise) over a monolayer.
'''

import sys
import json
import imageio
import argparse
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter

sys.path.append("scripts/utils/")
from Microscopes import Holomonitor, Tomocube
from SegmentedCells import SegmentedCells
from file_operations import load_stack

data_path = "../../../../hdd_data/silja/Monolayers/"


parser = argparse.ArgumentParser(description="")
parser.add_argument("path", type=str, help="Path to dataset, as config/dataset")
parser.add_argument("-t", "--type", type=str, help="Measure type, pixel or disc", default="pixel")
args = parser.parse_args()

Path(f"results/mean_evolutions/").mkdir(parents=True, exist_ok=True)

###############
# Import data #
###############

# Import pixel heights of dataset
dataset = Path(args.path).stem
config  = json.load(open(f"configs/{dataset}.json"))

h_pix = load_stack(f"{data_path}height_fields/calibrated/{dataset}/", config, param="height", data_type="cells")
try:
    mask = imageio.v2.imread(f"{data_path}masks/{dataset}_mask.tiff")
    h_pix = h_pix * mask
except:
    print("Mask not applied!")


h_pix = np.ma.array(h_pix, mask=(h_pix==0))

cells = SegmentedCells(f"{data_path}cell_features/calibrated/{dataset}_cells.p")

### SET MICROSCOPE ###
if config['microscope'] == "holomonitor": microscope = Holomonitor()
elif config['microscope'] == "tomocube":  microscope = Tomocube()



# Apply Gaussian blur if type is disc
if args.type == "disc":

    h_disc = []
    for h, rho in zip(h_pix, cells.density):

        A_mean = 1/(rho / 10**6)
        r_mean = np.sqrt(A_mean / np.pi)
        sigma  = microscope.rblur * r_mean / microscope.pix_to_um

        # Apply Gaussian blur to data
        h_disc.append(gaussian_filter(h, sigma))

    h_disc = np.ma.array(h_disc, mask=h_pix.mask)




# Compute mean, std and save
if args.type == "disc":
    h_mean = np.ma.mean(h_disc, axis=(1,2), keepdims=False)
    h_std  = np.ma.std(h_disc,  axis=(1,2), keepdims=False)

    np.save(f"results/mean_evolutions/{dataset}_h_disc_mean", h_mean.data) 
    np.save(f"results/mean_evolutions/{dataset}_h_disc_std",  h_std.data) 

elif args.type == "pixel":
    h_mean = np.ma.mean(h_pix, axis=(1,2), keepdims=False)
    h_std  = np.ma.std(h_pix,  axis=(1,2), keepdims=False)

    np.save(f"results/mean_evolutions/{dataset}_h_pixel_mean", h_mean.data) 
    np.save(f"results/mean_evolutions/{dataset}_h_pixel_std",  h_std.data) 

