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

if microscope.name == "holomonitor":

    cells.h = cells.h + config["calibration"]["hshift"]
    cells.save(f"{data_path}cell_features/calibrated/{dataset}_cells.p")

    # Correct field
    stack = load_stack(f"{data_path}height_fields/raw/{dataset}/", config, param="height", data_type="field")

    # Create 
    Path(f"{data_path}height_fields/calibrated/{dataset}/").mkdir(parents=True, exist_ok=True)

    # Save field
    for i in range(len(stack)):
        frame = (stack[i] + config["calibration"]["hshift"]) / microscope.h_to_um

        imageio.imwrite(f"{data_path}height_fields/calibrated/{dataset}/MDCK-li_reg_zero_corr_fluct_{i}.tiff", np.array(frame, dtype=np.uint16))


