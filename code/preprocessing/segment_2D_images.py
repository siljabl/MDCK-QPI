import os
import json
import pickle
import argparse
import numpy  as np
import pandas as pd

from tqdm     import tqdm
from pathlib  import Path
from datetime import datetime
from skimage.feature import peak_local_max
from skimage.measure import regionprops

from utils.segment2D     import *
from utils.segment3D     import *
from utils.file_handling import *

import matplotlib
matplotlib.use("Agg")


config_path    = "data/experimental/configs/"
processed_path = "data/experimental/processed/"



parser = argparse.ArgumentParser(description="Usage: python segement_2D_images.py dir file"
    )
parser.add_argument("path",          type=str,    help="Path to data series")
parser.add_argument("--psize",       type=float,  help="Particle radius at initial frame",                        default=13)
parser.add_argument("--tau",         type=float,  help="Doubling time of cells (h)",                              default=16)
parser.add_argument("--s_high",      type=int,    help="kernel size for Gaussian filter applied to data",         default=6)
parser.add_argument("--s_low",       type=int,    help="kernel size for Gaussian filter subtracting from data",   default=12)
parser.add_argument("--scaling",     type=int,    help="Holomonitor scaling to µm",                               default=100)
parser.add_argument("--fmin",        type=int,    help="First useful frame",                                      default=1)
parser.add_argument("--fmax",        type=int,    help="First useful frame",                                      default=337)
parser.add_argument("--clear_edge",  action="store_true",   help="Should be True if monolayer is larger than FOV, otherwise False")
parser.add_argument("--enforce_confluency",       help="Converts false empty regions in data to non-empty for segmentation", action="store_true")
args = parser.parse_args()

microscope = Path(args.path).stem.split("_")[0]


# check if config exists and load
fname = Path(args.path).stem
Path(f"{processed_path}{fname}").mkdir(parents=True, exist_ok=True)


try:
    with open(f"data/experimental/configs/{fname}.json", 'r') as f:
        config = json.load(f)
        config['segmentation']['date'] = datetime.today().strftime('%Y-%m-%d')
        print(f"Loading configs from data/experimental/configs/{fname}.json")
except:
    print(f"Found no config file at data/experimental/configs/{fname}.json")
    config = {'segmentation': {
                'date':   datetime.today().strftime('%Y-%m-%d'),
                'fmin':   args.fmin,
                'fmax':   args.fmax,
                's_low':  args.s_low,
                's_high': args.s_high,
                'tau':    args.tau,
                'particle_size':   args.psize}
}


if microscope == 'holomonitor':
    pix_to_um  = get_pixel_size()
    frame_to_h = 1 / 12
    h_im = import_holomonitor_stack(args.path, 
                                    f_min=config['segmentation']['fmin'],
                                    f_max=config['segmentation']['fmax'])[:, :940, :940]
    n_im = np.copy(h_im)

elif microscope == 'tomocube':
    pix_to_um = get_voxel_size_35mm()
    frame_to_h = 1 / 4
    n_im, h_im = import_tomocube_stack(args.path, 
                                       h_scaling=pix_to_um[0], 
                                       f_min=config['segmentation']['fmin'], 
                                       f_max=config['segmentation']['fmax'])

# empty arrays for storing data--bio",  "
cells_df = pd.DataFrame()
im_areas = []
regprops = []

# linear array of H for imextendmax
#H_arr = np.linspace(config['segmentation']['Hmax'],        config['segmentation']['Hmin'],      len(h_im), endpoint=True)
#s_arr = np.linspace(config['segmentation']['s_low_start'], config['segmentation']['s_low_end'], len(h_im), endpoint=True)


# segment cells in each frame
for i in tqdm(range(len(h_im))):
    
    # smoothen image and remove large scale variation
    if args.enforce_confluency:
        n_blur = np.copy(n_im[i])
        n_blur[n_blur < 1.33] = 1.37 #np.mean(im_blur)
        n_blur = sc.ndimage.gaussian_filter(n_blur, 40)

        n_im[i][n_im[i] < 1.33] = n_blur[n_im[i] < 1.33]

    n_norm = smoothen_normalize_im(n_im[i], config['segmentation']['s_high'], 
                                            config['segmentation']['s_low'])

    # estimate particle size from cell doubling time
    particle_size = config['segmentation']['particle_size'] * 2 ** (-i / (2*config['segmentation']['tau'] / frame_to_h))
    pos = np.array(peak_local_max(n_norm, min_distance=int(np.round(particle_size))))

    #pos = pos[pos[:,0]>12]

    # segment cell areas using watershed
    if microscope == 'tomocube':
        n_norm = smoothen_normalize_im(n_im[i], 10, 15)
    areas = get_cell_areas(-n_norm, pos, h_im[i], clear_edge=args.clear_edge)

    # save frame
    regprops.append(regionprops(areas, h_im[i]))
    im_areas.append(areas)

    #pos = update_pos(pos, areas)

    # compute cell properties
    #tmp_df = compute_cell_props(areas, pos, h_im[i], n_im[i], pix_to_um[1], type=microscope)
    #tmp_df['frame'] = i

    # save to df and list
    #cells_df = pd.concat([cells_df, tmp_df], ignore_index=True)


# filter out small cells
#cells_df = cells_df[cells_df.A >= 100] # change
#cells_df.to_csv(f"data/experimental/processed/{fname}/dataframe_unfiltered.csv", index=False)

if microscope == "tomocube":
    with open(f"{f"data/experimental/processed/{fname}/raw_cell_props_0-20.p"}", 'wb') as f: 
        pickle.dump(regprops[0:], f)

    with open(f"{f"data/experimental/processed/{fname}/raw_cell_props_21-40.p"}", 'wb') as f: 
        pickle.dump(regprops[21:], f)

else:
    with open(f"{f"data/experimental/processed/{fname}/raw_cell_props.p"}", 'wb') as f: 
            pickle.dump(regprops, f)

np.save(f"data/experimental/processed/{fname}/im_cell_areas.npy", im_areas)
json.dump(config, open(f"data/experimental/configs/{fname}.json", "w"))
