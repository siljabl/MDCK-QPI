"""
Plot population parameter as function of time

PARAMETERS:
- path: path to dataset
- param: population parameter to plot. 

"""

import sys
import json
import pickle
import argparse
import numpy as np
from pathlib import Path

# Avoid localhost error on my machine
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')

sys.path.append("code/analysis")
from utils.data_class import SegmentationData
from utils.variation_functions import global_density

sys.path.append("code/preprocessing/utils")
from file_handling import *



def plot_cell_number(Ncells, dataset, frame_to_h = 1/12, savefig=True, prefix="filtered"):
    """ Plot global cell density """

    hour = np.arange(len(Ncells)) * frame_to_h

    plt.figure(figsize=(6, 4), dpi=300)
    plt.xlabel("t (h)")
    plt.ylabel(r"$N_{cell}$")
    plt.plot(hour, Ncells, '.')
    plt.tight_layout()
    if savefig:
        plt.savefig(f"results/{dataset}/{prefix}_cell_number.png")



def plot_cell_density(data, dataset):
    """ Plot global cell density """

    cell_density = global_density(data.A)

    plt.figure(figsize=(6, 4), dpi=300)
    plt.xlabel("frame")
    plt.ylabel(r"$\rho_{cell}~(mm^{-2})$")
    plt.plot(cell_density, '.')
    plt.tight_layout()
    plt.savefig(f"results/{dataset}/cell_density.png")


def plot_intensity(stack, dataset):
    """ Plot mean intensity/height in units of µm """

    mean = np.mean(stack, axis=(1,2))

    plt.figure(figsize=(6, 4), dpi=300)
    plt.xlabel("frame")
    plt.ylabel(r"$\langle h \rangle~(µm)$")
    plt.plot(mean, '.')
    plt.tight_layout()
    plt.savefig(f"results/{dataset}/mean_intensity.png")
    


def main():
    parser = argparse.ArgumentParser(description="Plot time evolutions")
    parser.add_argument('path',  type=str, help="Path to files to plot")
    parser.add_argument('param', type=str, help="Population parameter to plot.")
    parser.add_argument('--raw',           action="store_true")
    args = parser.parse_args()

    # Assert correct input
    param_list = ["density", "intensity", "number", "height"]
    assert args.param in param_list, f"Must chose parameter among {param_list}."

    # Make ouput dir
    dataset    = Path(args.path).stem
    microscope = dataset.split("_")[0]
    Path(f"results/{dataset}/").mkdir(parents=True, exist_ok=True)  

    # Plot global cell density
    if args.param == "density":
        data = SegmentationData(f"data/experimental/processed/{dataset}/cell_props.p")

        plot_cell_density(data, dataset)

    # Plot total number of cells
    if args.param == "number":
        if args.raw:
            im_cell_areas = np.load(f"data/experimental/processed/{dataset}/raw_im_cell_areas.npy")
            prefix = "raw"
        else:
            im_cell_areas = np.load(f"data/experimental/processed/{dataset}/im_cell_areas.npy")
            prefix = "filtered"

        if microscope == "holomonitor":
            frame_to_h = 1/12
        else:
            frame_to_h = 1/4

        Ncells = [len(np.unique(im_area))-1 for im_area in im_cell_areas]
        plot_cell_number(Ncells, dataset, frame_to_h, prefix=prefix)

    # Plot mean intensity/height
    if args.param == "intensity" or args.param == "height":
        config = json.load(open(f"data/experimental/configs/{dataset}.json"))
        #data = SegmentationData(f"data/experimental/processed/{dataset}/cell_props.p")

        # if microscope == 'holomonitor':
        stack = import_holomonitor_stack(f"data/experimental/raw/{dataset}/", 
                                        f_min=config['segmentation']['fmin'],
                                        f_max=config['segmentation']['fmax'])
        
        plot_intensity(stack, dataset)
        





if __name__ == "__main__":
    main()


