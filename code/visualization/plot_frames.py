"""
Script to plot frames that illustrates data analysis
func:
- cell_detection

"""

import os
import sys
import json
import pickle
import imageio
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import skimage.morphology as morph

# module_path = os.path.abspath(os.path.join(''))
# if module_path not in sys.path:
#     sys.path.append(module_path)

from tqdm import tqdm
from pathlib import Path
from matplotlib_scalebar.scalebar import ScaleBar

import matplotlib
matplotlib.use("Agg")

sys.path.append("code/preprocessing/utils")
from segment2D     import *
from segment3D     import *
from file_handling import *

sys.path.append("code/analysis/utils")
from data_class import SegmentationData




######################
# Plotting functions #
######################

def plot_cell_detection(ax, frame, pos, vmin=0, vmax=20, label='h (µm)'):
    """ 
    Plotting raw tiff with detected cell centers on top

    Parameters:
    ax:    ax object to plot data on
    frame: frame of raw data to plot as heatmap
    pos:   positions of cell centers as [x, y]
    vmin:  min intensity on heatmap
    vmax:  max intensity on heatmap
    """

    sns.heatmap(frame, ax=ax, square=True, cmap="gray", vmin=vmin, vmax=vmax, 
                xticklabels=False, yticklabels=False, cbar=True, cbar_kws={'label':label})
    
    ax.plot(pos.T[1], pos.T[0], 'r.', ms=5)
    ax.set(title=f"#cells: {len(pos)}")
    #ax.set(title=f"#cells: {np.sum(pos[0].mask==False)}")




def plot_cell_height():
    return 0


def plot_cell_volume():
    return 0


def plot_cell_velocity():
    return 0


def plot_field_velocity():
    return 0



####################
# Perform plotting #
####################

def main():
    parser = argparse.ArgumentParser(description="Usage: python cell_segmentation_Holomonitor.py dir file")
    parser.add_argument("path",     type=str,  help="Path to data folder")
    parser.add_argument("func",     type=str,  help="Plotting function")
    parser.add_argument("-edges",   type=bool, help="Plot edges", default=False)
    parser.add_argument("--raw", action="store_true", help="plotting raw result in stead of final")
    args = parser.parse_args()


    # Decompose input
    microscope = Path(args.path).stem.split("_")[0]
    dataset    = Path(args.path).stem

    if microscope == 'holomonitor':
        pix_to_um = get_pixel_size()

    elif microscope == 'tomocube':
        pix_to_um = get_voxel_size_35mm()


    # Load data
    config   = json.load(open(f"data/experimental/configs/{dataset}.json"))
    fmin     = config['segmentation']['fmin']
    if args.raw:
        with open(f"data/experimental/processed/{dataset}/raw_cell_props.p", 'rb') as f:
            cellprop = pickle.load(f)
        Nframes  = len(cellprop)
    else:
        cellprop = SegmentationData(f"data/experimental/processed/{dataset}/cell_props.p")
        Nframes  = len(cellprop.x)


    # Define path for output
    outdir = f"figs/frames/{args.func}/{dataset}/"
    Path(outdir).mkdir(parents=True, exist_ok=True) 


    # set value range
    vmin = 0
    vmax = 20


    # loop through frames
    for frame in tqdm(range(Nframes)):


        # Import raw image
        if args.func == "cell_detection":# or args.func == "cell_velocity" or arg.

            if microscope == "holomonitor":
                #im = import_holomonitor_stack(f"data/experimental/raw/{dataset}/", f_min=fmin + frame, f_max=fmin + frame)[0]
                im = imageio.v2.imread(f"data/experimental/raw/{dataset}/MDCK-li_reg_zero_corr_fluct_{fmin+frame}.tiff") / 100
            else:
                #im = import_tomocube_stack(f"data/experimental/raw/{dataset}/", h_scaling=pix_to_um[0], f_min=fmin + frame, f_max=fmin + frame)[0][0]
                im = imageio.v2.imread(f"data/experimental/raw/{dataset}/MDCK-li_reg_zero_corr_fluct_{fmin+frame}.tiff") / pix_to_um[0]


        
        # plot
        fig, ax = plt.subplots(1,1, figsize=(10,8))

        if args.func == "cell_detection":

            if args.raw:
                positions = np.array([cell.centroid_weighted for cell in cellprop[frame]])
            else:
                positions = [cellprop.x[frame] / pix_to_um[1], cellprop.y[frame] / pix_to_um[1]]

            plot_cell_detection(ax, im, positions, vmin=vmin, vmax=vmax)


        elif args.func == "cell_height":
            plot_cell_height()


        elif args.func == "cell_volume":
            plot_cell_volume()


        elif args.func == "cell_velocity":
            plot_cell_velocity()


        elif args.func == "field_velocity":
            plot_field_velocity()

        else:
            print("Error: func not recognized.")



        # add scalebar
        sb = ScaleBar(pix_to_um[-1], 'um', box_alpha=0, color="w", height_fraction=2e-2, scale_loc="none", fixed_value=100)
        sb.location = 'lower left'
        ax.add_artist(sb)

        # save
        if args.raw:
            prefix = "raw_"
        else:
            prefix = ""
        fig.tight_layout()
        plt.savefig(f"{outdir}/{prefix}frame_{frame}.png", dpi=300);
        plt.close()




if __name__ == "__main__":
    main()
