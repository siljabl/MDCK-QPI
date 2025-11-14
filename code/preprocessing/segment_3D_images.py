'''
Create 3D mask out of 3D probabilities from catBioM_MlM
'''

import os
import pickle
import argparse
import tifffile
import numpy as np

from pathlib import Path
from datetime import datetime

from skimage.filters import median
from skimage.morphology import disk

from utils.plots     import *
from utils.segment3D import *
from src.ImUtils     import commonStackReader

import matplotlib
matplotlib.use("Agg")

parser = argparse.ArgumentParser(description='Segment cell from MlM probabilities')
parser.add_argument('dir',       type=str, help="path to main dir")
parser.add_argument('--r1_min',  type=int, help="min radius of kernel 1", default='3')
parser.add_argument('--r1_max',  type=int, help="max radius of kernel 1", default='11')
parser.add_argument('--r2',      type=int, help="radius of kernel 2", default='25')
parser.add_argument('--Nframes', type=int, help="Number of frames")
args = parser.parse_args()


# create folders
in_dir = f"{args.dir}predictions"
assert os.path.exists(in_dir), "In path does not exist, maybe because of missing '/' at the end of path"
print(in_dir)

path = Path(in_dir)
out_dir  = f"{args.dir}segmentation"
mhds_dir = f"{out_dir}{os.sep}mhds"
corrected_dir = f"{out_dir}{os.sep}z_correction"

try:
    os.mkdir(out_dir)
    os.mkdir(mhds_dir)
    os.mkdir(corrected_dir)
except:
    pass


# get experiment specific dataseries names
experiment = []
for file in path.glob("*HT3D_0_prob.npy"):
    experiment.append(file.stem.split("HT3D_0_prob")[0])


# create arrays for prediction and filtering
thresholds = np.linspace(0.5, 1, 20, endpoint=False)
kernel_1 = generate_kernel(args.r1_min, args.r1_max)
kernel_2 = np.array([disk(args.r2)])



# saving MlM threshold of each file
new_log = 1
logfile = f"{mhds_dir}{os.sep}log.txt"
if os.path.exists(logfile):
    new_log = 0



with open(logfile, "a") as log:
    if new_log:
        log.write("# file, zero-level, MlM threshold, r1_min, r1_max, r2\n")
    log.write(f"date: {datetime.today().strftime('%Y-%m-%d')}\n")

    # sort by experiment
    for exp in experiment:
        print(exp)

        #threshold = np.zeros(args.Nframes)

        # compute list for determination of zero level
        print(f"\nDetermining zero-level for experiment {exp} ...")
        for file in path.glob(f"{exp}*_prob.npy"):

            # get file name and frame for printing
            stack_name = f"{path.parent}{os.sep}{file.name.split('_prob.npy')[0]}.tiff"
            frame = int(stack_name.split('_')[-1].split('.tiff')[0])
            print(f"Frame {frame}")

            # load stacks and ML-predictions
            stack = commonStackReader(stack_name)
            MlM_probabilities = np.load(file)
            cell_prob = MlM_probabilities[:,:,:,1]

            z0 = estimate_cell_bottom(stack)
                
            # compute array for determination of MlM threshold
            sum_above = np.zeros_like(thresholds)

            for i in range(len(thresholds)):
                mask = (cell_prob > thresholds[i])
                sum_above[i] = np.sum(mask[z0:])

            # compute threshold
            threshold = determine_threshold(thresholds, sum_above)
            log.write(f"{frame}, z0: {z0}, p_thres: {threshold}\n")

            # apply threshold and split 
            cell_pred  = (cell_prob > threshold)

            # filter mask
            cell_mask = median(cell_pred, kernel_1)
            #cell_mask = median(cell_mask,  kernel_2)

            # save mask
            out_mask = f"{out_dir}{os.sep}{file.name.split('_prob.npy')[0]}_mask.tiff"
            tifffile.imwrite(out_mask, np.array(cell_mask, dtype=np.uint8), bigtiff=True)

