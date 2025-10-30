import os
import sys
import json
import argparse
import numpy as np

from pathlib import Path

sys.path.append("code/preprocessing/utils")
from segment2D     import *
from segment3D     import *
from file_handling import *

# import from analysis/utils/
from utils.data_class import SegmentationData, AutocorrelationData
#from utils.helper_functions import global_density, average_cell_radius



def remove_field_mean(stack):

    variation = np.copy(stack)

    i = 0
    for frame in stack:
        field_mean = np.mean(frame[frame != 0])
        variation[i][frame != 0] -= field_mean

        i += 1

    return variation



def compute_cell_correlation(dataset, args):

    # Load cell properties
    cellprop = SegmentationData(f"data/experimental/processed/{dataset}/cell_props.p")

    # Get cell properties
    positions  = [cellprop.x, cellprop.y]
    heights    =  cellprop.h
    areas      =  cellprop.A
    volumes    =  cellprop.h * cellprop.A
    vpositions = [(cellprop.x[:-1] + cellprop.x[1:]) / 2, (cellprop.y[:-1] + cellprop.y[1:]) / 2]
    velocities = [cellprop.dx, cellprop.dy]


    # Define mean variable axis
    if args.mean_var == 'r':
        mean_var = 1
    elif args.mean_var == 'cell': 
        mean_var = 0

    # Subtract mean
    h_variation = np.ma.array(heights - np.mean(heights, axis=mean_var, keepdims=True), mask=False)
    A_variation = np.ma.array(areas   - np.mean(areas,   axis=mean_var, keepdims=True), mask=False)
    V_variation = np.ma.array(volumes - np.mean(volumes, axis=mean_var, keepdims=True), mask=False)
    v_variation = np.ma.array([velocities[0] - np.mean(velocities[0], axis=mean_var, keepdims=True),
                               velocities[1] - np.mean(velocities[1], axis=mean_var, keepdims=True)])

    print(np.shape(positions), np.shape(velocities))

    # Initialize correlation object
    autocorr_obj = AutocorrelationData(f"data/experimental/processed/{dataset}/cell_autocorr.p")
    autocorr_obj.add_density(cellprop.A)

    if args.var == 'r' or args.var == 'all':
        # Upper limit on distance
        rmax = np.max([np.max(positions[0]), np.max(positions[1])]) * args.rfrac

        # Compute spatial autocorrelations
        if args.param == 'hh' or args.param == 'all':
            autocorr_obj.compute_spatial(positions, h_variation, 'hh', args.dr, rmax, t_avrg=args.t_avrg, overwrite=args.overwrite)
        if args.param == 'AA' or args.param == 'all':
            autocorr_obj.compute_spatial(positions, A_variation, 'AA', args.dr, rmax, t_avrg=args.t_avrg, overwrite=args.overwrite)
        if args.param == 'VV' or args.param == 'all':
            autocorr_obj.compute_spatial(positions, V_variation, 'VV', args.dr, rmax, t_avrg=args.t_avrg, overwrite=args.overwrite)
        if args.param == 'vv' or args.param == 'all':
            autocorr_obj.compute_spatial(vpositions, v_variation, 'vv', args.dr, rmax, t_avrg=args.t_avrg, overwrite=args.overwrite) 


    if args.var == 't' or args.var == 'all':
        # Upper limit on t ime difference
        tmax = int(len(cellprop.h) * args.tfrac)

        # Compute temporal autocorrelations
        if args.param == 'hh' or args.param == 'all':
            autocorr_obj.compute_temporal(h_variation, 'hh', tmax, mean_var=args.mean_var, t_avrg=args.t_avrg, overwrite=args.overwrite)
        if args.param == 'AA' or args.param == 'all':
            autocorr_obj.compute_temporal(A_variation, 'AA', tmax, mean_var=args.mean_var, t_avrg=args.t_avrg, overwrite=args.overwrite)
        if args.param == 'VV' or args.param == 'all':
            autocorr_obj.compute_temporal(V_variation, 'VV', tmax, mean_var=args.mean_var, t_avrg=args.t_avrg, overwrite=args.overwrite)
        if args.param == 'vv' or args.param == 'all':
            print(args.mean_var)
            autocorr_obj.compute_temporal(v_variation, 'vv', tmax, mean_var=args.mean_var, t_avrg=args.t_avrg, overwrite=args.overwrite)

    # Save autocorrelation as .autocorr
    autocorr_obj.save()




def compute_field_correlations(dataset, config, args, xy_to_um):
    """
    Compute correlations on unsegmented data

    Parameters:
    dataset: Name of dataset folder
    h_field: Height field, stack of heights
    v_field: Velocity field from PIV
    v_positions: Positions of velocity field
    args: arguments that specify what to plot and how. Defined in main()
    xy_to_um: conversion factor
    """

    # Initialize correlation object, where correlations will be saved.
    autocorr_obj = AutocorrelationData(f"data/experimental/processed/{dataset}/field_autocorr.p")

    # import height fiels
    print(f"data/experimental/raw/{dataset}/")
    h_field = import_stack(f"data/experimental/raw/{dataset}/", config)
    h_dims = np.shape(h_field)

    # Upper limits on x-axis
    rmax = len(h_field[0]) * xy_to_um * args.rfrac
    tmax = int(len(h_field) * args.tfrac)


    # Compute height autocorrelation
    if args.param == 'hh' or args.param == 'all':
        
        # Reshape input array to form that is expected by correlation functions (frames x positions)
        h_variation =  remove_field_mean(h_field)
        h_variation = np.reshape(h_variation, (h_dims[0], h_dims[1]*h_dims[2]))
        h_variation = np.ma.array(h_variation, mask=h_variation==0)

        # Spatial autocorrelation
        if args.var == 'r' or args.var =='all':

            # Reshape position arrays to (frames x positions)
            x = np.arange(h_dims[1]) * xy_to_um
            y = np.arange(h_dims[2]) * xy_to_um
            Y, X = np.meshgrid(y, x)

            X = X.reshape(h_dims[1]*h_dims[2])
            Y = Y.reshape(h_dims[1]*h_dims[2])

            positions = [np.ma.array(np.tile(X, h_dims[0]).reshape(h_dims[0], h_dims[1]*h_dims[2]), mask=h_variation.mask), 
                         np.ma.array(np.tile(Y, h_dims[0]).reshape(h_dims[0], h_dims[1]*h_dims[2]), mask=h_variation.mask)]
            
            # Compute correlation
            autocorr_obj.compute_spatial(positions, h_variation, 'hh', args.dr, rmax, t_avrg=args.t_avrg, overwrite=args.overwrite)

        # Temporal autocorrelation
        if args.var == 't' or args.var =='all':

            # Compute correlation
            autocorr_obj.compute_temporal(h_variation, 'hh', tmax, mean_var=args.mean_var, t_avrg=args.t_avrg, overwrite=args.overwrite)

    
    # Compute velocity autocorrelation
    if args.param == 'vv' or args.param == 'all':

        # Load data
        positions, v_field = import_PIV_velocity(f"data/experimental/PIV/{dataset}/", config, h_field.shape)

       # Subtract mean value from fields
        v_variation = [remove_field_mean(v_field[0]), 
                       remove_field_mean(v_field[1])]

        # Reshape position arrays to (frames x positions)  
        v_dims = np.shape(v_variation)
        v_variation = np.reshape(v_variation, (v_dims[0], v_dims[1], v_dims[2]*v_dims[3]))
        v_variation = np.ma.array(v_variation, mask=v_variation==0)

        # Spatial autocorrelation
        if args.var == 'r' or args.var =='all':

            # Reshape position arrays to (frames x positions)
            positions = positions * xy_to_um
            positions = np.ma.array(positions.reshape(v_dims[0], v_dims[1], v_dims[2]*v_dims[3]), mask=v_variation.mask)

            # Compute correlation
            autocorr_obj.compute_spatial(positions, v_variation, 'vv', args.dr, rmax, t_avrg=args.t_avrg, overwrite=args.overwrite)

        # Temporal autocorrelation
        if args.var == 't' or args.var =='all':

            # Compure correlation
            autocorr_obj.compute_temporal(v_variation, 'vv', tmax, mean_var=args.mean_var, t_avrg=args.t_avrg, overwrite=args.overwrite)

    # Save object
    autocorr_obj.save()
        




# relative_variation = np.zeros([3, len(h_stack)])
# average_height     = np.zeros([3, len(h_stack)])

# r_cell = average_cell_radius(cellprop.A)
# sigma = rblur * r_cell / pix_to_um[1]


# i = 0
# for im in h_stack:

#     # blur image
#     im_disk = gaussian_filter(im, int(sigma[i]))

#     # compute variation
#     relative_variation[0,i], average_height[0,i] = spatial_variation(im)
#     relative_variation[1,i], average_height[1,i] = spatial_variation(im_disk)
#     relative_variation[2,i], average_height[2,i] = spatial_variation(cellprop.h[i])

#     i += 1


# # Bin data
# Nframes = len(n_stack)

# cell_density = global_density(cellprop.A)
# mean_density   = np.copy(cell_density[:-dt])
# mean_variation = np.copy(relative_variation[:,:-dt])
# mean_height    = np.copy(average_height[:,:-dt])

# # Take average of neightbour bins
# for i in range(1,dt):
#     mean_density   += cell_density[i:-dt+i]
#     mean_variation += relative_variation[:,i:-dt+i]
#     mean_height    += average_height[:,i:-dt+i]

# mean_density   = mean_density[int(dt/2):Nframes:dt] / 6
# mean_variation = mean_variation[:,int(dt/2):Nframes:dt] / 6
# mean_height    = mean_height[:,int(dt/2):Nframes:dt] / 6

# # Save data
# output = VariationData(f"data/experimental/processed/{dataset}/height_variations.p")
# output.add_unbinned_data(mean_density, mean_height[0], mean_variation[0], 'pixel')
# output.add_unbinned_data(mean_density, mean_height[1], mean_variation[1], 'disk')
# output.add_unbinned_data(mean_density, mean_height[2], mean_variation[2], 'cell')
# output.save()


def main():

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("path",          type=str,   help="Path to dataset, as data/experimental/processed/dataset")
    parser.add_argument('-p', '--param', type=str,   help="Parameter to plot correlation of (varvar)",                          default="all")
    parser.add_argument('-v', '--var',   type=str,   help="Correlation variable (t or r)",                                      default="all")
    parser.add_argument('-o','--overwrite',          help="Overwrite previous computations",                                    action='store_true')
    parser.add_argument('--dr',          type=float, help="Spatial step size (float)",                                          default=20)
    parser.add_argument('--rfrac',       type=float, help="Max distance to compute correlation for (float)",                    default=0.5)
    parser.add_argument('--tfrac',       type=float, help="Fraction of total duration to compute correlation for (float)",      default=0.5)
    parser.add_argument('--mean_var',    type=str,   help="Variable to take mean over in <x - <x>_var> (r or cell). Default: r",  default='r')
    parser.add_argument('--t_avrg',                  help="Take average over all starting times (if steady state)", action="store_true")
    parser.add_argument("--field",                   help="Compute correlations on height and velocity fields.",                action="store_true")
    parser.add_argument("--cells",                   help="Compute correlations on segmented cell properties.",                 action="store_true")
    args = parser.parse_args()

    assert args.field or args.cells, "Must pass either --field or --cells"

    # Load config
    dataset = Path(args.path).stem
    config = json.load(open(f"data/experimental/configs/{dataset}.json"))
    Path(f"data/experimental/processed/{dataset}/").mkdir(parents=True, exist_ok=True)

    # Compute correlation on cell properties
    if args.cells:

        # Compute correlation
        compute_cell_correlation(dataset, args)

    # Compute correlation on field
    if args.field:

        # Load conversion factor
        microscope = dataset.split("_")[0]

        if microscope == 'holomonitor':
            xy_to_um = get_pixel_size()[0]

        elif microscope == 'tomocube':
            xy_to_um = get_voxel_size_35mm()[1]

        # compute correlation
        compute_field_correlations(dataset, config, args, xy_to_um)


if __name__ == "__main__":
    main()

