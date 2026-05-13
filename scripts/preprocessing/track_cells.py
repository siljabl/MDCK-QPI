
"""
Author S. B. Låstad

Track cells with trackpy, using both position and area as tracking parameters
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
import trackpy as tp

from pathlib  import Path
from skimage.measure import regionprops

sys.path.append("scripts/utils/")
from file_operations import *
from segmentation    import *
from Microscopes     import Holomonitor, Tomocube
from SegmentedCells  import SegmentedCells


# Paths
data_path    = "../../../../hdd_data/silja/Monolayers/"

parser = argparse.ArgumentParser(description="Usage: python track_cells.py dir microscope")
parser.add_argument("path",     type=str,  help="Path to data folder. Typically '../../../../hdd_data/silja/Monolayers/height_fields/<dataset>/'")
args = parser.parse_args()



### LOAD DATA ###
dataset   = Path(args.path).stem
config    = json.load(open(f"configs/{dataset}.json"))

if config['microscope'] == "holomonitor": 
    microscope = Holomonitor()
elif config['microscope'] == "tomocube":  
    microscope = Tomocube()

im_areas  = load_label_images(f"{data_path}cell_labels/corrected/{dataset}/")
im_height = load_stack(f"{data_path}height_fields/raw/{dataset}/", config, param="height", data_type="field")

# Compute region props of cells in all frames
cellprops = [regionprops(im_areas[i], im_height[i]) for i in range(len(im_areas))]

# Prepare dataframe for tracking
Fcells = np.concatenate([[config['field']['fmin'] + frame for cell in cellprops[frame]] for frame in range(len(cellprops))])

Acells       = np.concatenate([[cell.area              for cell in cells] for cells in cellprops])
hcells       = np.concatenate([[cell.mean_intensity    for cell in cells] for cells in cellprops])
Lcells       = np.concatenate([[cell.label             for cell in cells] for cells in cellprops])
amajor       = np.concatenate([[cell.axis_major_length for cell in cells] for cells in cellprops])
aminor       = np.concatenate([[cell.axis_minor_length for cell in cells] for cells in cellprops])
orientation  = np.concatenate([[cell.orientation       for cell in cells] for cells in cellprops])
eccentricity = np.concatenate([[cell.eccentricity      for cell in cells] for cells in cellprops])
x_position   = np.concatenate([[cell.centroid_weighted[1] for cell in cells] for cells in cellprops])
y_position   = np.concatenate([[cell.centroid_weighted[0] for cell in cells] for cells in cellprops])
print(x_position)
print(y_position)

cells_df = pd.DataFrame({'x': x_position,
                         'y': y_position,
                         'area': Acells / microscope.A_scale,
                         'hmean': hcells,
                         'label': Lcells, 
                         'a_max': amajor,
                         'a_min': aminor,
                         'theta': orientation, 
                         'ecc': eccentricity,
                         'frame': Fcells})

# Add RI for Tomocube
if microscope.name == "tomocube":
    
    n_mean_arr = []
    for i in range(len(im_areas)):
        ri_field    = imageio.v2.imread(f"{data_path}ri_field/raw/{dataset}/MDCK-li_refractive_index_{i}.tiff") * microscope.ri_conversion
        n_cellprops = regionprops(im_areas[i], ri_field)

        n_mean_arr.append([cell.mean_intensity for cell in n_cellprops])

    cells_df['nmean'] = np.concatenate(n_mean_arr)


### Track cells ###
tracks = tp.link(cells_df, search_range=microscope.track_range, 
                           memory=microscope.memory, 
                           pos_columns=['x', 'y', 'hmean', 'area']);

tracks = tp.filter_stubs(tracks, threshold=microscope.threshold);

# Keep only frames where all cells can have tracks longer than threshold
fmin = config['cells']['fmin']
fmax = config['cells']['fmax']
tracks = tracks[(tracks.frame >= fmin) * (tracks.frame <= fmax)]
tracks.area *= microscope.A_scale


# Convert to maskes arrays
cells = SegmentedCells(f"{data_path}cell_features/raw/{dataset}_cells_.p")
cells.transform_df_to_ma(tracks, microscope.pix_to_um)

# Filter out based on cell size
remove_small = (cells.h > config["filtering"]["hmin"]) * (cells.A > config["filtering"]["Amin"]) * (cells.h * cells.A > config["filtering"]["Vmin"])
remove_large = (cells.h < config["filtering"]["hmax"]) * (cells.A < config["filtering"]["Amax"]) * (cells.h * cells.A < config["filtering"]["Vmax"])

cells.remove_cells(remove_small * remove_large)

cells.save(f"{data_path}cell_features/raw/{dataset}_cells.p")


# Create new im_areas with only surviving cells
im_areas_tracked = np.copy(im_areas)[fmin:-fmin]

for i in range(len(im_areas_tracked)):
    labels = tracks[tracks.frame==fmin + i].label.values
    exclude = np.setdiff1d(np.unique(im_areas[i]), labels)

    for cell in exclude:
        mask = im_areas_tracked[i] == cell
        im_areas_tracked[i][mask] = 0

# Save im_areas
Path(f"{data_path}cell_labels/tracked/{dataset}/").mkdir(parents=True, exist_ok=True)
save_id_images(im_areas_tracked, tracks, f"{data_path}cell_labels/tracked/{dataset}/", fmin=config['cells']['fmin'])

