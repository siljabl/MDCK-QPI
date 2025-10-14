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
dataset = Path(args.path).stem
Path(f"{processed_path}{dataset}").mkdir(parents=True, exist_ok=True)


try:
    with open(f"data/experimental/configs/{dataset}.json", 'r') as f:
        config = json.load(f)
        config['segmentation']['date'] = datetime.today().strftime('%Y-%m-%d')
        print(f"Loading configs from data/experimental/configs/{dataset}.json")
except:
    print(f"Found no config file at data/experimental/configs/{dataset}.json")
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

if microscope == 'holomonitor':
    pix_to_um  = get_pixel_size()
    frame_to_h = 1 / 12

elif microscope == 'tomocube':
    pix_to_um = get_voxel_size_35mm()
    frame_to_h = 1 / 4

# empty arrays for storing data
cells_df = pd.DataFrame()
im_areas = []


# segment cells in each frame
for i in tqdm(range(Nframes)):

    # import frame
    if microscope == "holomonitor":
        h_im = imageio.v2.imread(f"data/experimental/raw/{dataset}/MDCK-li_reg_zero_corr_fluct_{fmin+i}.tiff") / 100
        n_im = np.copy(h_im)
    else:
        n_im = imageio.v2.imread(f"data/experimental/raw/{dataset}/MDCK-li_refractive_index_{fmin+i}.tiff") / 10_000
        h_im = imageio.v2.imread(f"data/experimental/raw/{dataset}/MDCK-li_height_{fmin+i}.tiff") / pix_to_um[0]

   
    # smoothen image and remove large scale variation
    if args.enforce_confluency:
        n_blur = np.copy(n_im)
        n_blur[n_blur < 1.33] = 1.37 #np.mean(im_blur)
        n_blur = sc.ndimage.gaussian_filter(n_blur, 40)

        n_im[n_im < 1.33] = n_blur[n_im < 1.33]

    n_norm = smoothen_normalize_im(n_im, config['segmentation']['s_high'], 
                                         config['segmentation']['s_low'])

    # estimate particle size from cell doubling time
    particle_size = config['segmentation']['particle_size'] * 2 ** (-i / (2*config['segmentation']['tau'] / frame_to_h))
    pos = np.array(peak_local_max(n_norm, min_distance=int(np.round(particle_size))))

    # remove non-confluent areas
    pos = pos[(pos[:,1] > config['segmentation']['xmin']) * (pos[:,1] < config['segmentation']['xmax'])]
    pos = pos[(pos[:,0] > config['segmentation']['ymin']) * (pos[:,0] < config['segmentation']['ymax'])]

    # segment cell areas using watershed
    if microscope == 'tomocube':
        n_norm = smoothen_normalize_im(n_im, 20, 30)
    areas = get_cell_areas(-n_norm, pos, h_im, clear_edge=args.clear_edge)

    # save frame
    im_areas.append(areas)


# save output
np.save(f"data/experimental/processed/{dataset}/im_cell_areas.npy", im_areas)
json.dump(config, open(f"data/experimental/configs/{dataset}.json", "w"))
