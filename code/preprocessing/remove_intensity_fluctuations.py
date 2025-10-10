'''
Transform images to tiff and fit mean intensity to growth curve
'''
import os
import json
import imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.stats   import linregress
from utils.file_handling import *

parser = argparse.ArgumentParser(description="")
parser.add_argument("path", type=str, help="Path to dataset, as data/experimental/processed/dataset")
parser.add_argument("--linear",       help="Remove fluctuations by ",  action="store_true")
parser.add_argument("--gaussian",       action="store_true")
args = parser.parse_args()


# Paths
dataset = Path(args.path).stem
Path(f"data/experimental/PIV/{dataset}/").mkdir(parents=True, exist_ok=True)  


# Load data
config = json.load(open(f"data/experimental/configs/{dataset}.json"))
stack  = import_stack(f"data/experimental/raw/{dataset}/", config)
stack  = np.ma.array(stack, mask = stack == 0)

# compute mean
frames = np.arange(1, 182)
mean_intensity = np.ma.mean(stack, axis=(1,2))


if args.linear:
    fit = linregress(frames, mean_intensity)
    corrected_mean = frames * fit.slope + fit.intercept

if args.gaussian:
    corrected_mean = gaussian_filter(mean_intensity, sigma=10)

else:
    corrected_mean = mean_intensity

# New images
stack_fluct = stack.data * (corrected_mean / mean_intensity)[:,np.newaxis, np.newaxis]

#stack_bit  = np.array(stack_fluct.data * (2**8 / np.max(stack_fluct)), dtype=np.uint8)
stack = np.array(stack_fluct, dtype=np.uint16)

for i in range(2):
    if args.no_fluct:
        imageio.v2.imwrite(f"data/experimental/raw/{dataset}/MDCK-li_reg_zero_corr_fluct_{frames[i]}.tiff", stack[i],  dtype=np.uint8)
        #imageio.v2.imwrite(f"test8bit_{frames[i]}.tiff", stack_8bit[i],  dtype=np.uint8)

