import sys
import json
import imageio
import argparse
import numpy  as np

from pathlib  import Path
from skimage.measure import regionprops
from skimage.feature import peak_local_max

sys.path.append("scripts/utils/")
from file_operations import *
from SegmentedCells  import SegmentedCells
from Microscopes     import Holomonitor, Tomocube



def poly2(p, x):
    return p[0] + p[1] * x + p[2] * x**2

# Paths
data_path = "../../../../hdd_data/silja/Monolayers/"

parser = argparse.ArgumentParser(description="Usage: python track_cells.py dir microscope")
parser.add_argument("path",     type=str,  help="Path to data folder. Typically 'config/<dataset>/'")
args = parser.parse_args()



### LOAD DATA ###
dataset = Path(args.path).stem
config  = json.load(open(f"configs/{dataset}.json"))
cells   = SegmentedCells(f"{data_path}cell_features/raw/{dataset}_cells.p")


if config['microscope'] == "holomonitor": 
    microscope = Holomonitor()
elif config['microscope'] == "tomocube":  
    microscope = Tomocube()


### 2D QPI ###
if microscope.name == "holomonitor":

    # calibrate cell features
    cells.h = cells.h + config["calibration"]["hshift"]
    cells.save(f"{data_path}cell_features/calibrated/{dataset}_cells.p")

    N = config["field"]["fmax"] + 1 - config["field"]["fmin"]
    density = np.linspace(cells.density[0], cells.density[-1], N)

    # Create folder
    Path(f"{data_path}height_fields/calibrated/{dataset}/").mkdir(parents=True, exist_ok=True)

    # Save field
    for f, rho in zip(range(config["field"]["fmin"], config["field"]["fmax"]), 
                      density):

        h_im, _ = load_set_of_frames(f"{data_path}height_fields/raw/{dataset}", f, microscope)

        # Shift according to reference
        h_im_calibrated = (h_im + config["calibration"]["hshift"])
        h_im_calibrated[h_im_calibrated < 0] = 0

        # Divide by calibrated ri contrast
        dn = poly2(microscope.a, rho)
        h_im_calibrated = h_im_calibrated * microscope.dn0 / dn
        
        imageio.imwrite(f"{data_path}height_fields/calibrated/{dataset}/MDCK-li_reg_zero_corr_fluct_{f}.tiff", np.array(h_im_calibrated / microscope.h_to_um, dtype=np.uint16))



### 3D QPI ###
if microscope.name == "tomocube":

    # calibrate cell features
    cells.n = cells.n + config["calibration"]["nshift"]
    cells.save(f"{data_path}cell_features/calibrated/{dataset}_cells.p")

    # Create folder
    Path(f"{data_path}ri_fields/calibrated/{dataset}/").mkdir(parents=True, exist_ok=True)
    Path(f"{data_path}height_fields/calibrated/{dataset}/").mkdir(parents=True, exist_ok=True)

    # Save field
    for f in range(config["field"]["fmin"], 
                   config["field"]["fmax"]):

        # Load field at one instant 
        h_im, n_im = load_set_of_frames(f"{data_path}height_fields/raw/{dataset}", f, microscope)

        # Update
        n_im_calibrated = n_im + config["calibration"]["nshift"]
        h_im_calibrated = h_im
        
        # Save
        imageio.imwrite(f"{data_path}ri_fields/calibrated/{dataset}/MDCK-li_refractive_index_{f}.tiff", np.array(n_im_calibrated / microscope.ri_conversion, dtype=np.uint16))
        imageio.imwrite(f"{data_path}height_fields/calibrated/{dataset}/MDCK-li_height_{f}.tiff",       np.array(h_im_calibrated / microscope.h_to_um,       dtype=np.uint16))
