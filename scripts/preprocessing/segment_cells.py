
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
from skimage.feature import peak_local_max
from skimage.measure import regionprops

sys.path.append("scripts/utils/")
# from segment2D     import *
# from segment3D     import *
from file_handling import *
from experimental_parameters import *


# Input paths
config_path = "configs/"
height_path = "../../../../hdd_data/silja/Monolayers/height_fields/"
ri_path     = "../../../../hdd_data/silja/Monolayers/ri_fields/"

# Output paths
label_path  = "../../../../hdd_data/silja/Monolayers/cell_labels/"


parser = argparse.ArgumentParser(description="Usage: python segement_2D_images.py dir file"
    )
parser.add_argument("path",           type=str,    help="Path to directory containing data. Typically '~/../../hdd_data/silja/Monolayers/height_fields/<dataset>/'. ")
parser.add_argument("--psize",        type=float,  help="Particle radius at initial frame",                        default=13)
parser.add_argument("--tau",          type=float,  help="Doubling time of cells (h)",                              default=16)
parser.add_argument("--s_high",       type=int,    help="kernel size for Gaussian filter applied to data",         default=6)
parser.add_argument("--s_low",        type=int,    help="kernel size for Gaussian filter subtracting from data",   default=12)
parser.add_argument("--scaling",      type=int,    help="holomonitor scaling to µm",                               default=100)
parser.add_argument("--fmin",         type=int,    help="First useful frame",                                      default=1)
parser.add_argument("--fmax",         type=int,    help="Last useful frame",                                       default=337)
parser.add_argument("--clear_edge",   action="store_true",   help="Should be True if monolayer is larger than FOV, otherwise False")
args = parser.parse_args()

# Extract experiment identificators
dataset = Path(args.path).stem


# Load or create config dictionary
try:
    with open(f"{config_path}{dataset}.json", 'r') as f:
        config = json.load(f)
        config['segmentation']['date'] = datetime.today().strftime('%Y-%m-%d')
        print(f"Loading configs from {config_path}{dataset}.json")
except:
    print(f"Found no config file at {config_path}{dataset}.json")
    config = {'segmentation': {
                'date':   datetime.today().strftime('%Y-%m-%d'),
                'fmin':   args.fmin,
                'fmax':   args.fmax,
                's_low':  args.s_low,
                's_high': args.s_high,
                'tau':    args.tau,
                'particle_size':   args.psize}
}
fmin = config['segmentation']['fmin']
fmax = config['segmentation']['fmax']
Nframes = fmax - fmin + 1


# Quick fix for identifying microscope
microscope = "holomonitor"
if "2025" in dataset:
    microscope = "tomocube"

if microscope == 'holomonitor':
    pix_to_um  = get_pixel_size()
    frame_to_h = 1 / 12
    memory = 5
    search_range = 20
    ascale = 50

elif microscope == 'tomocube':
    pix_to_um = get_voxel_size_35mm()
    frame_to_h = 1 / 4
    memory = 3
    search_range = 80
    ascale = 150


# empty arrays for storing data
cells_df = pd.DataFrame()
im_areas = []
# segment cells in each frame
for i in tqdm(range(Nframes)):

    # import frame
    if microscope == "holomonitor":
        h_im = imageio.v2.imread(f"{height_path}{dataset}/MDCK-li_reg_zero_corr_fluct_{fmin+i}.tiff") / 100
        n_im = np.copy(h_im)
    else:
        n_im = imageio.v2.imread(f"{ri_path}{dataset}/MDCK-li_refractive_index_{fmin+i}.tiff") / 10_000
        h_im = imageio.v2.imread(f"{height_path}{dataset}/MDCK-li_height_{fmin+i}.tiff") / (pix_to_um[0] * 1000)

    n_norm = smoothen_normalize_im(n_im, config['segmentation']['s_high'], 
                                         config['segmentation']['s_low'])

    # estimate particle size from cell doubling time
    particle_size = config['segmentation']['particle_size'] * 2 ** (-i / (2*config['segmentation']['tau'] / frame_to_h))
    full_pos = np.array(peak_local_max(n_norm, min_distance=int(np.round(particle_size))))

    # remove non-confluent areas
    pos = full_pos[(full_pos[:,1] > config['segmentation']['xmin']) * (full_pos[:,1] < config['segmentation']['xmax'])]
    pos = full_pos[(full_pos[:,0] > config['segmentation']['ymin']) * (full_pos[:,0] < config['segmentation']['ymax'])]

    # segment cell areas using watershed
    if microscope == 'tomocube':
        n_norm = smoothen_normalize_im(n_im, 20, 30)

    raw_areas = get_cell_areas(-n_norm, pos, h_im, clear_edge=False)

    # get cell properties
    cell_props     = regionprops(raw_areas, h_im)
    cell_areas     = np.array([cell.area for cell in cell_props])
    cell_heights   = np.array([cell.mean_intensity for cell in cell_props])
    cell_positions = np.array([cell.centroid for cell in cell_props], dtype=int)

    # create mask to remove small cells
    remove_small  = (cell_heights > config['filtering']['hmin'])
    remove_small *= (cell_areas   > config['filtering']['Amin'] / pix_to_um[1]**2) 
    remove_small *= (cell_areas * cell_heights > config['filtering']['Vmin']  / pix_to_um[1]**2)

    # redo watershed without small cells
    areas = get_cell_areas(-n_norm, pos[remove_small], h_im, clear_edge=args.clear_edge)

    # save frame to temporary dataframe
    im_areas.append(raw_areas)

    tmp_df = pd.DataFrame({'x': pos.T[1][remove_small],
                           'y': pos.T[0][remove_small],
                           'area': cell_areas[remove_small] / ascale,
                           'frame': i * np.ones_like(pos.T[1][remove_small])})
    
    cells_df = pd.concat([cells_df, tmp_df], ignore_index=True)

# save areas before filtering
Path(f"{label_path}{dataset}/raw/").mkdir(parents=True, exist_ok=True)
save_label_images(im_areas, f"{label_path}/{dataset}/raw", fmin=fmin)
# np.save(f"{label_path}{dataset}/im_cell_areas_raw.npy", im_areas)
# cells_df.to_csv(f"{label_path}{dataset}/dataframe.csv")


# remove short lived detections
tracks = tp.link(cells_df, search_range=search_range, memory=memory, pos_columns=['x', 'y', 'area']);
tracks = tp.filter_stubs(tracks, threshold=config['filtering']['track_threshold']);


# redo watershed with filtered positions
im_areas = []
for i in tqdm(range(Nframes)):

    # import frame
    if microscope == "holomonitor":
        h_im = imageio.v2.imread(f"{height_path}{dataset}/MDCK-li_reg_zero_corr_fluct_{fmin+i}.tiff") / 100
        n_im = np.copy(h_im)
    else:
        n_im = imageio.v2.imread(f"{ri_path}{dataset}/MDCK-li_refractive_index_{fmin+i}.tiff") / 10_000
        h_im = imageio.v2.imread(f"{height_path}{dataset}/MDCK-li_height_{fmin+i}.tiff") / pix_to_um[0]

    # smoothen
    n_norm = smoothen_normalize_im(n_im, config['segmentation']['s_high'], 
                                         config['segmentation']['s_low'])

    # get relevant frame from dataframe
    tracks_tmp = tracks[tracks.frame == i]
    x_cell = tracks_tmp.x.values
    y_cell = tracks_tmp.y.values

    pos = np.array([y_cell, x_cell]).T

    areas = get_cell_areas(-n_norm, pos, h_im, clear_edge=args.clear_edge)
    im_areas.append(areas)


# save output
Path(f"{label_path}{dataset}/corrected/").mkdir(parents=True, exist_ok=True)
save_label_images(im_areas, f"{label_path}/{dataset}/corrected", fmin=fmin)

json.dump(config, open(f"{config_path}{dataset}.json", "w"))
