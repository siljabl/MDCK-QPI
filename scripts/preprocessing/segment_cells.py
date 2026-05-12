
"""
Author S. B. Låstad

Segment cell areas from height and refractive index fields using peak detection and watershed.

"""

import sys
import json
import argparse
import numpy  as np
import pandas as pd
import trackpy as tp

from tqdm     import tqdm
from pathlib  import Path
from datetime import datetime
from skimage.measure import regionprops
from skimage.feature import peak_local_max

sys.path.append("scripts/utils/")
from file_handling import *
from segmentation_functions import *
from microscope_features    import Holomonitor, Tomocube


# Paths
config_path = "configs/"
height_path = "../../../../hdd_data/silja/Monolayers/height_fields/"
ri_path     = "../../../../hdd_data/silja/Monolayers/ri_fields/"
label_path  = "../../../../hdd_data/silja/Monolayers/cell_labels/"


parser = argparse.ArgumentParser(description="Usage: python segement_2D_images.py dir file")
parser.add_argument("path",         type=str, help="Path to directory containing data. Typically '~/../../hdd_data/silja/Monolayers/height_fields/<dataset>/'. ")
parser.add_argument("microscope",   type=str, help="Microscope that where used for data. 'holomonitor' or 'tomocube'.")
parser.add_argument("--clear_edge", action="store_true",   help="Should be True if monolayer is larger than FOV, otherwise False")
args = parser.parse_args()



### SET MICROSCOPE ###
assert args.microscope == 'holomonitor' or args.microscope == 'tomocube', "Error: do not recognize microscope."

if args.microscope == "holomonitor": microscope = Holomonitor()
elif args.microscope == "tomocube":  microscope = Tomocube()



### LOAD CONFIG ###
dataset = Path(args.path).stem
try:
    with open(f"{config_path}{dataset}.json", 'r') as f:
        config = json.load(f)
        config['segmentation']['date'] = datetime.today().strftime('%Y-%m-%d')
        print(f"Loading configs from {config_path}{dataset}.json")
except:
    print(f"Found no config file at {config_path}{dataset}.json")

p0   = config['detection']['particle_size']
tau  = config['detection']['tau']
fmin = config['detection']['fmin']
fmax = config['detection']['fmax']
Nframes = fmax - fmin + 1


# Define and create output
cells_df = pd.DataFrame()
Path(f"{label_path}{dataset}/raw/").mkdir(parents=True, exist_ok=True)
Path(f"{label_path}{dataset}/corrected/").mkdir(parents=True, exist_ok=True)


for i in tqdm(range(Nframes)):

    # Import frames
    h_im, n_im = load_set_of_frames(dataset, fmin+i, microscope)

    # Smoothen field
    n_norm = smoothen_normalize_im(n_im, config['detection']['s_high'], 
                                         config['detection']['s_low'])

    # Find peaks
    # Estimating average particle size from cell doubling time
    p_size   = np.round(p0 * 2 ** (-i / (2*tau / microscope.frame_to_h)))
    full_pos = np.array(peak_local_max(n_norm, min_distance=int(p_size)))

    # Remove non-confluent areas
    pos = full_pos[(full_pos[:,1] > config['segmentation']['xmin']) * (full_pos[:,1] < config['segmentation']['xmax'])]
    pos =      pos[(     pos[:,0] > config['segmentation']['ymin']) * (     pos[:,0] < config['segmentation']['ymax'])]

    # Smoothen again
    if microscope.name == 'tomocube':
        n_norm = smoothen_normalize_im(n_im, config['segmentation']['s_high'],
                                             config['segmentation']['s_low'])
        # n_norm = smoothen_normalize_im(n_im, 20, 30)
    
    # Segment cell areas using watershed
    raw_areas = get_cell_areas(-n_norm, pos, h_im, clear_edge=False)
    save_label_image(raw_areas, f"{label_path}/{dataset}/raw", fmin=fmin)

    # Get cell properties
    cell_props     = regionprops(raw_areas, h_im)
    cell_areas     = np.array([cell.area for cell in cell_props])
    cell_heights   = np.array([cell.mean_intensity for cell in cell_props])
    cell_volumes   = cell_areas * cell_heights 
    cell_positions = np.array([cell.centroid for cell in cell_props], dtype=int)

    # create mask to remove small cells
    remove_small  = (cell_heights > config['filtering']['hmin'])
    remove_small *= (cell_areas   > config['filtering']['Amin'] / microscope.pix_to_um**2) 
    remove_small *= (cell_volumes > config['filtering']['Vmin'] / microscope.pix_to_um**2)

    # Save as temporary data frame
    tmp_df = pd.DataFrame({'x': pos.T[1][remove_small],
                           'y': pos.T[0][remove_small],
                           'area': cell_areas[remove_small] / microscope.ascale,
                           'frame': i * np.ones_like(pos.T[1][remove_small])})
    
    cells_df = pd.concat([cells_df, tmp_df], ignore_index=True)



# Remove short lived detections
tracks = tp.link(cells_df, search_range=microscope.search_range, 
                           memory=microscope.memory, 
                           pos_columns=['x', 'y', 'area']);

tracks = tp.filter_stubs(tracks, threshold=config['filtering']['track_threshold']);



# Redo watershed with filtered positions
for i in tqdm(range(Nframes)):

    # Import frames
    h_im, n_im = load_set_of_frames(dataset, fmin+i, microscope)

    # smoothen
    n_norm = smoothen_normalize_im(n_im, config['segmentation']['s_high'], 
                                         config['segmentation']['s_low'])

    # get relevant frame from dataframe
    tracks_tmp = tracks[tracks.frame == i]
    x_cell = tracks_tmp.x.values
    y_cell = tracks_tmp.y.values

    pos = np.array([y_cell, x_cell]).T

    areas = get_cell_areas(-n_norm, pos, h_im, clear_edge=args.clear_edge)
    save_label_image(areas, f"{label_path}/{dataset}/corrected", fmin=fmin)

