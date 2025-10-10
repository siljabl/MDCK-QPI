import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from tqdm import tqdm
from pathlib import Path
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm

sys.path.append("code/preprocessing/utils")
from segment2D     import *
from segment3D     import *
from file_handling import *

from utils.data_class import SegmentationData, VariationData
from utils.helper_functions import global_density, average_cell_radius
from utils.variation_functions import spatial_variation

parser = argparse.ArgumentParser(description="")
parser.add_argument("path", type=str, help="Path to dataset, as data/experimental/processed/dataset")
parser.add_argument("--dt", type=float, help="Duration between frames taht are considered to be independent [h]", default=0.5)
args = parser.parse_args()


# get dataset and microscope type
microscope = Path(args.path).stem.split("_")[0]
dataset = Path(args.path).stem


# load data
config = json.load(open(f"data/experimental/configs/{dataset}.json"))
data = SegmentationData()
data.load(f"data/experimental/processed/{dataset}/cell_props.p")

if microscope == 'holomonitor':
    dt = int(args.dt * 12)
    rblur = 0.75
    pix_to_um = get_pixel_size()
    h_stack = import_holomonitor_stack(f"data/experimental/raw/{dataset}/", 
                                    f_min=config['segmentation']['fmin'],
                                    f_max=config['segmentation']['fmax'])
    n_stack = np.copy(h_stack)


elif microscope == 'tomocube':
    dt = int(args.dt * 4)
    rblur = 0.9
    pix_to_um = get_voxel_size_35mm()
    n_stack, h_stack = import_tomocube_stack(f"data/experimental/raw/{dataset}/", 
                                       h_scaling=pix_to_um[0], 
                                       f_min=config['segmentation']['fmin'], 
                                       f_max=config['segmentation']['fmax'])



relative_variation = np.zeros([3, len(h_stack)])
average_height     = np.zeros([3, len(h_stack)])

r_cell = average_cell_radius(data.A)
sigma = rblur * r_cell / pix_to_um[1]


i = 0
for im in h_stack:

    # blur image
    im_disk = gaussian_filter(im, int(sigma[i]))

    # compute variation
    relative_variation[0,i], average_height[0,i] = spatial_variation(im)
    relative_variation[1,i], average_height[1,i] = spatial_variation(im_disk)
    relative_variation[2,i], average_height[2,i] = spatial_variation(data.h[i])

    i += 1


# Bin data
Nframes = len(n_stack)

cell_density = global_density(data.A)
mean_density   = np.copy(cell_density[:-dt])
mean_variation = np.copy(relative_variation[:,:-dt])
mean_height    = np.copy(average_height[:,:-dt])

# Take average of neightbour bins
for i in range(1,dt):
    mean_density   += cell_density[i:-dt+i]
    mean_variation += relative_variation[:,i:-dt+i]
    mean_height    += average_height[:,i:-dt+i]

mean_density   = mean_density[int(dt/2):Nframes:dt] / 6
mean_variation = mean_variation[:,int(dt/2):Nframes:dt] / 6
mean_height    = mean_height[:,int(dt/2):Nframes:dt] / 6

# Save data
output = VariationData(f"data/experimental/processed/{dataset}/height_variations.p")
output.add_unbinned_data(mean_density, mean_height[0], mean_variation[0], 'pixel')
output.add_unbinned_data(mean_density, mean_height[1], mean_variation[1], 'disk')
output.add_unbinned_data(mean_density, mean_height[2], mean_variation[2], 'cell')
output.save()
