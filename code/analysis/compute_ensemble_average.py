import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

from glob import glob
from tqdm import tqdm
from pathlib import Path
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm

sys.path.append("code/preprocessing/utils")
from segment2D     import *
from segment3D     import *
from file_handling import *

from utils.data_class import SegmentationData, VariationData, AutocorrelationData
from utils.variation_functions import spatial_variation, global_density



def weighted_average(data, err, ax, empty_val=0):
    """ 
    Computes weighted averages over binned datasets (Typically binned by density). 

    PARAMETERS:
    data:       Array of mean values in bins. Data entries without data
    err:        Array of standard deviations in bins
    ax:         Index of axis to take average over, i.e. dimension of dataset
    empty_val:  Value that indicates that the bin is empty

    RETURNS:
    wmean:      Weighted mean of each bin
    wstd:       Weighted uncertainty on the mean
    """

    # Mask entries without data
    mask = (data==empty_val)
    data = np.ma.array(data, mask=mask)
    err  = np.ma.array(err,  mask=mask)

    # Compute weights
    weights = np.zeros_like(err.data)
    weights[err > 0] = 1 / err.data[err > 0]**2
    weights = np.ma.array(weights, mask=mask)

    # Number of non-zero weights
    N = np.sum(weights!=0, axis=ax)

    # Compute weighted mean and error on the mean
    wmean = np.ma.average(data, weights=weights, axis=ax)
    wstd  = np.sqrt(np.ma.average((data-wmean)**2, weights=weights, axis=ax) / (N-1))

    return wmean, wstd




def compute_variation_average(args):
    
    # Collect variation data
    binned_pixel = []
    binned_disk  = []
    binned_cell  = []

    for path in args.paths:

        # Load individual variation data
        dataset = Path(path).stem
        data    = VariationData(f"data/experimental/processed/{dataset}/height_variations.p")

        # For unbinned data, i.e. technical replicates
        if data.datatype == "unbinned":
            tmp_pixel = data.bin_data(data.var_pixel, bin_size=args.bin_size)
            tmp_disk  = data.bin_data(data.var_disk,  bin_size=args.bin_size)
            tmp_cell  = data.bin_data(data.var_cell,  bin_size=args.bin_size)
            binned_pixel.append(tmp_pixel)
            binned_disk.append(tmp_disk)
            binned_cell.append(tmp_cell)
        
        # For binned data, i.e. averages of technical replicates 
        elif data.datatype == "binned":
            bin_size = data.density[1] - data.density[0]
            bins = np.arange(data.density[0]-bin_size/2, data.density[-1]+bin_size*3/2, bin_size)
            tmp_pixel = {"density_bins": bins, "mean": data.var_pixel, "std": data.svar_pixel}
            tmp_disk  = {"density_bins": bins, "mean": data.var_disk, "std": data.svar_disk}
            tmp_cell  = {"density_bins": bins, "mean": data.var_cell, "std": data.svar_cell}
            binned_pixel.append(tmp_pixel)
            binned_disk.append(tmp_disk)
            binned_cell.append(tmp_cell)


    # Sort by density
    bins  = np.unique(np.concatenate([binned['density_bins'] for binned in binned_pixel]))
    means = np.zeros([3, len(bins)-1, len(binned_pixel)])
    stds  = np.zeros([3, len(bins)-1, len(binned_pixel)])

    i = 0
    for pixel, disk, cell in zip(binned_pixel, binned_disk, binned_cell):
        density_idx = np.digitize(pixel['density_bins'], bins)

        means[0, density_idx[:-1]-1, i] = pixel['mean']
        means[1, density_idx[:-1]-1, i] = disk['mean']
        means[2, density_idx[:-1]-1, i] = cell['mean']
        stds[0, density_idx[:-1]-1, i]  = pixel['std']
        stds[1, density_idx[:-1]-1, i]  = disk['std']
        stds[2, density_idx[:-1]-1, i]  = cell['std']

        i += 1

    # Take weighted mean
    wmean, wstd = weighted_average(means, stds, ax=2)

    # Save
    dataset = Path(args.outdir).stem
    output = VariationData(f"data/experimental/processed/{dataset}/height_variations.p")

    cell_density  = (bins[1:] + bins[:-1]) / 2

    output.add_binned_data(cell_density, wmean[0], wstd[0], "pixel")
    output.add_binned_data(cell_density, wmean[1], wstd[1], "disk")
    output.add_binned_data(cell_density, wmean[2], wstd[2], "cell")
    output.save()




def compute_correlation_average(args):
    """
    Computes average of correlation data.

    PARAMETERS IN ARGS:
    paths:  List of paths to datasets to take average over
    field:  Boolian that is true if we should take average over field correlations
    cell:   Boolian that is true if we should take average over cell correlations
    """
    
    binned_corr = []
    x_arrays = []

    # Collect (and bin) correlation data
    for path in args.paths:

        # Load individual correlation data
        dataset = Path(path).stem
        if args.field:
            data = AutocorrelationData(f"data/experimental/processed/{dataset}/field_autocorr.p")

        else:
            data = AutocorrelationData(f"data/experimental/processed/{dataset}/cell_autocorr.p")


        # For unbinned data, i.e. technical replicates
        tmp_corr = data.bin_data(args.var, args.param)


        if args.var == "r":
            x_arrays.append(data.r_array[args.param])
            binned_corr.append(tmp_corr)

        elif args.var == "t":
            x_arrays.append(data.t_array[args.param])
            binned_corr.append(tmp_corr)
        

    #Full x-array of all datasets
    x_array = np.unique(np.concatenate([x for x in x_arrays]))

    # Sort by density
    bins  = np.unique(np.concatenate([binned['density_bins'] for binned in binned_corr]))
    means = np.zeros([len(binned_corr), len(bins)-1, len(x_array)])
    stds  = np.zeros([len(binned_corr), len(bins)-1, len(x_array)])

    i = 0
    for corr in binned_corr:

        density_idx = np.digitize(corr['density_bins'], bins)

        means[i, density_idx[:-1]-1, :len(x_arrays[i])] = corr['mean']
        stds[i,  density_idx[:-1]-1, :len(x_arrays[i])] = corr['std'] / corr['N_in_bin'][:,np.newaxis]

        i += 1

    # Take weighted average
    wmean, wstd = weighted_average(means, stds, ax=0)


    # Save
    dataset = Path(args.outdir).stem
    if args.field:
        output = AutocorrelationData(f"data/experimental/processed/{dataset}/field_autocorr.p")
    else:
        output = AutocorrelationData(f"data/experimental/processed/{dataset}/cell_autocorr.p")

    cell_density  = (bins[1:] + bins[:-1]) / 2
    output.density = cell_density

    if args.var == "r":
        output.r_array[args.param] = x_array
        output.spatial[args.param] = wmean

    elif args.var == "t":
        output.t_array[args.param]  = x_array
        output.temporal[args.param] = wmean

    output.log[args.var][args.param] = datetime.today().strftime('%Y/%m/%d_%H:%M')
    output.save()



def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("outdir",           type=str, help="Path to dataset, as data/experimental/processed/dataset-outdir")
    parser.add_argument("variable",         type=str, help="Variable to take average over")
    parser.add_argument("paths",            nargs='*', help="List of dataset.")
    parser.add_argument("--field", action="store_true")
    parser.add_argument("--cell",  action="store_true")
    parser.add_argument("-p", "--param", default="")
    parser.add_argument("-v", "--var", default="")
    #parser.add_argument("--technical",      action="store_true", help="Take average of technical replicates")
    #parser.add_argument("--biological",     action="store_true", help="Take average of biological replicates")
    parser.add_argument("-b", "--bin_size", type=int, help="Path to dataset, as data/experimental/processed/dataset-outdir", default=200)
    #parser.add_argument("measure", type=str, help="Measure to take average of")
    args = parser.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # Get files that represent biological or technical replicates
    assert len(args.paths) > 0, "Must pass at least two datasets!"
    #assert args.technical or args.biological, "Must pass either --technical or --biological"

    # paths = []
    # for dataset in args.datasets:
    #     paths.append(glob(f"data/experimental/raw{}"))

    # if args.technical:
    #     paths = glob(f"{args.outdir}*-*")

    # else:
    #     paths = ["data/experimental/processed/holomonitor_20240301_B2-5/",
    #              "data/experimental/processed/holomonitor_20240319_A1/",
    #              "data/experimental/processed/holomonitor_20240319_B1-11/",
    #              "data/experimental/processed/holomonitor_20240516_B1-9/"]


    if args.variable == "variation":
        compute_variation_average(args)

    if args.variable == "correlation":
        compute_correlation_average(args) 



if __name__ == "__main__":
    main()
