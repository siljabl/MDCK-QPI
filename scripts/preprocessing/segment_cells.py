
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
from file_operations import *
from segmentation    import *
from Microscopes     import Holomonitor, Tomocube


# Paths
config_path = "configs/"
height_path = "../../../../hdd_data/silja/Monolayers/height_fields/"
ri_path     = "../../../../hdd_data/silja/Monolayers/ri_fields/"
label_path  = "../../../../hdd_data/silja/Monolayers/cell_labels/"


parser = argparse.ArgumentParser(description="Usage: python segement_2D_images.py dir microscope")
parser.add_argument("path",         type=str, help="Path to directory containing data. Typically '~/../../hdd_data/silja/Monolayers/height_fields/<dataset>/'. ")
args = parser.parse_args()


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
fmin = config['field']['fmin']
fmax = config['field']['fmax']
Nframes = fmax - fmin + 1

if config['boundary'] == "open":
    clear_edge = True
else:
    clear_edge = False


### SET MICROSCOPE ###
if config['microscope'] == "holomonitor": microscope = Holomonitor()
elif config['microscope'] == "tomocube":  microscope = Tomocube()




# Define and create output
cells_df = pd.DataFrame()
Path(f"{label_path}raw/{dataset}/").mkdir(parents=True, exist_ok=True)
Path(f"{label_path}corrected/{dataset}/").mkdir(parents=True, exist_ok=True)


for f in tqdm(range(fmin, fmax)):

    # Import frames
    h_im, n_im = load_set_of_frames(dataset, f, microscope)

    # Smoothen field
    n_norm = smoothen_normalize_im(n_im, config['detection']['s_high'], 
                                         config['detection']['s_low'])

    # Find peaks
    # Estimating average particle size from cell doubling time
    p_size   = np.round(p0 * 2 ** (-(f-fmin) / (2*tau / microscope.frame_to_h)))
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
    save_label_image(raw_areas, f"{label_path}raw/{dataset}/", frame=f)

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
                           'area': cell_areas[remove_small] / microscope.A_scale,
                           'frame': f * np.ones_like(pos.T[1][remove_small])})
    
    cells_df = pd.concat([cells_df, tmp_df], ignore_index=True)



# Remove short lived detections
tracks = tp.link(cells_df, search_range=microscope.search_range, 
                           memory=microscope.memory, 
                           pos_columns=['x', 'y', 'area']);

tracks = tp.filter_stubs(tracks, threshold=config['filtering']['track_threshold']);



# Redo watershed with filtered positions
for f in tqdm(range(fmin, fmax+1)):

    # Import frames
    h_im, n_im = load_set_of_frames(dataset, f, microscope)

    # smoothen
    n_norm = smoothen_normalize_im(n_im, config['segmentation']['s_high'], 
                                         config['segmentation']['s_low'])

    # get relevant frame from dataframe
    tracks_tmp = tracks[tracks.frame == f]
    x_cell = tracks_tmp.x.values
    y_cell = tracks_tmp.y.values

    pos = np.array([y_cell, x_cell]).T

    areas = get_cell_areas(-n_norm, pos, h_im, clear_edge=clear_edge)
    save_label_image(areas, f"{label_path}corrected/{dataset}/", frame=f)

