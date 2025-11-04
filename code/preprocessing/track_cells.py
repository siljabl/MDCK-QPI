# import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt

from pathlib  import Path
from datetime import datetime
from skimage.measure import regionprops

sys.path.append("code/preprocessing/")
from utils.file_handling import *
from utils.segment2D import get_pixel_size
from utils.segment3D import get_voxel_size_35mm

sys.path.append("code/analysis/utils")
from data_class import SegmentationData


parser = argparse.ArgumentParser(description="Usage: python cell_segmentation_Holomonitor.py dir file")
parser.add_argument("path",     type=str,  help="Path to data folder")
parser.add_argument("-r", "--search_range", type=int, help="trackpy search_range",          default=10)
parser.add_argument("-m", "--memory",       type=int, help="trackpy memory",                default=5)
parser.add_argument("-t", "--threshold",    type=int, help="trackpy threshold",             default=6)
parser.add_argument("-a", "--a_scale",      type=int, help="scale area to search_range",    default=50)
args = parser.parse_args()


# Get dataset name
dataset = Path(args.path).stem

# Load data
config    = json.load(open(f"data/experimental/configs/{dataset}.json"))
im_areas  = np.load(f"data/experimental/processed/{dataset}/im_cell_areas_corrected.npy")
im_height = import_stack(f"data/experimental/raw/{dataset}/", config)

# Compute region props of cells in all frames
cellprops = [regionprops(im_areas[i], im_height[i]) for i in range(len(im_areas))]


# Prepare dataframe for tracking
Acells = np.concatenate([[cell.area for cell in cells] for cells in cellprops])
hcells = np.concatenate([[cell.mean_intensity for cell in cells] for cells in cellprops])
Lcells = np.concatenate([[cell.label for cell in cells] for cells in cellprops])
Fcells = np.concatenate([[frame for cell in cellprops[frame]] for frame in range(len(cellprops))])
pos_cells = np.concatenate([[cell.centroid_weighted for cell in cells] for cells in cellprops])

cells_df = pd.DataFrame({'x': pos_cells.T[1],
                         'y': pos_cells.T[0],
                         'area': Acells / args.a_scale,
                         'hmean': hcells,
                         'label': Lcells, 
                         'frame': Fcells})


# Track cells
tracks = tp.link(cells_df, search_range=args.search_range, memory=args.memory, pos_columns=['x', 'y', 'hmean', 'area']);
tracks = tp.filter_stubs(tracks, threshold=args.threshold);

# Keep only frames where all cells can have tracks longer than threshold
fmin = args.threshold - 1
fmax = np.max(tracks.frame) + 1 - fmin
tracks = tracks[(tracks.frame >= fmin) * (tracks.frame < fmax)]
tracks.area *= args.a_scale


# Create new im_areas with only surviving cells
im_areas_tracked = np.copy(im_areas)[fmin:-fmin]

for i in range(len(im_areas_tracked)):
    labels = tracks[tracks.frame==fmin + i].label.values
    exclude = np.setdiff1d(np.unique(im_areas[i]), labels)

    for cell in exclude:
        mask = im_areas_tracked[i] == cell
        im_areas_tracked[i][mask] = 0

# Save im_areas
np.save(f"data/experimental/processed/{dataset}/im_cell_areas_tracked.npy", im_areas_tracked)


# # Save dataframe as masked array?
# cellprops = [regionprops(im_areas_tracked[i], im_height[fmin+i]) for i in range(len(im_areas_tracked))]

# Ncells = [len(cells) for cells in cellprops]
# Acells = [[cell.area for cell in cells] for cells in cellprops]
# hcells = [[cell.mean_intensity for cell in cells] for cells in cellprops]
# Fcells = [[frame for cell in cellprops[frame]] for frame in range(len(cellprops))]

# Acells = np.concatenate(Acells)
# hcells = np.concatenate(hcells)
# Fcells = np.concatenate(Fcells)
# Vcells = hcells * Acells

# set microscope specific parameters
microscope = Path(dataset).stem.split("_")[0]

if microscope == 'holomonitor':
    xy_to_um = get_pixel_size()[0]


elif microscope == 'tomocube':
    xy_to_um = get_voxel_size_35mm()[1]



# save as pickle
data_obj = SegmentationData(f"data/experimental/processed/{dataset}/cell_props.p")
data_obj.transform_df_to_ma(tracks, xy_to_um)
data_obj.save(f"data/experimental/processed/{dataset}/cell_props.p")

# save filter config
config['filtering']['date'] = datetime.today().strftime('%Y-%m-%d')

json.dump(config, open(f"data/experimental/configs/{dataset}.json", "w"))