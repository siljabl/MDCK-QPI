"""
Script to plot frames that illustrates data analysis
func:
- cell_detection
- field_velocity or PIV
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
import matplotlib as mpl
import matplotlib.pyplot as plt
import skimage.morphology as morph
from skimage.measure import regionprops

from tqdm import tqdm
from pathlib import Path
from scipy.stats import linregress
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




def plot_cell_height(ax, frame, cellprop, hmin=0, hmax=11):
    """ 
    Plotting imshow of segmented cell areas with faces colors by average height

    Parameters:
    ax:       ax object to plot data on
    frame:    frame of raw data to plot as heatmap
    cellprop: list of regionprop of all cells in frame
    hmin:     min height in heatmap
    hmax:     max height in heatmap
    """
    
    h_cmap = sns.color_palette("Blues",   as_cmap=True)
    e_cmap = mpl.colors.ListedColormap(['none', 'w'])

    h_mean = np.zeros_like(frame, dtype=np.float64)
    e_im   = np.zeros_like(frame, dtype=int)

    for cell in cellprop:

        # isolate cell
        cell_mask = (frame == cell.label)

        # cell heights
        h_mean[cell_mask] = cell.mean_intensity

        # cell edges
        cell_interior = morph.erosion(cell_mask, footprint=morph.disk(1))
        edge = cell_mask ^ cell_interior
        e_im += edge

    e_im += (h_mean == 0)

    sns.heatmap(h_mean, ax=ax, square=True, cmap=h_cmap, vmin=hmin, vmax=hmax, xticklabels=False, yticklabels=False, cbar=True)
    sns.heatmap(e_im, ax=ax, cmap=e_cmap,  xticklabels=False, yticklabels=False, cbar=False)
    ax.set(title=f"Average cell height (µm)")

    return 0



def plot_cell_area(ax, frame, cellprop, Ascale, Amin=-1.1, Amax=1.1, xy_to_um=0.15):
    """ 
    Plotting imshow of segmented cell areas with faces colors by average height

    Parameters:
    ax:       ax object to plot data on
    frame:    frame of raw data to plot as heatmap
    cellprop: list of regionprop of all cells in frame
    hmin:     min height in heatmap
    hmax:     max height in heatmap
    """
    
    A_cmap = sns.color_palette("Greens",   as_cmap=True)
    e_cmap = mpl.colors.ListedColormap(['none', 'w'])

    A_mean = np.zeros_like(frame, dtype=np.float64)
    e_im   = np.zeros_like(frame, dtype=int)

    for cell in cellprop:

        # isolate cell
        cell_mask = (frame == cell.label)

        # cell areas
        A_mean[cell_mask] = np.log(cell.area * xy_to_um**2 / Ascale)

        # cell edges
        cell_interior = morph.erosion(cell_mask, footprint=morph.disk(1))
        edge = cell_mask ^ cell_interior
        e_im += edge

    e_im += (A_mean == 0)

    sns.heatmap(A_mean, ax=ax, square=True, cmap=A_cmap, vmin=Amin, vmax=Amax, xticklabels=False, yticklabels=False, cbar=True)
    sns.heatmap(e_im, ax=ax, cmap=e_cmap,  xticklabels=False, yticklabels=False, cbar=False)
    ax.set(title=r"$\log{[A_{cell} ~/~ \langle A \rangle_{t, linear}]}$")

    return 0



def plot_cell_volume(ax, frame, cellprop, Vmin=600, Vmax=8000, xy_to_um=0.15):
    """ 
    Plotting imshow of segmented cell areas with faces colors by the cell volume

    Parameters:
    ax:       ax object to plot data on
    frame:    frame of raw data to plot as heatmap
    cellprop: list of regionprop of all cells in frame
    Vmin:     min volume in heatmap
    Vmax:     max volume in heatmap
    """
        
    V_cmap = sns.color_palette("Oranges",   as_cmap=True)
    e_cmap = mpl.colors.ListedColormap(['none', 'w'])

    V_mean = np.zeros_like(frame, dtype=np.float64)
    e_im   = np.zeros_like(frame, dtype=int)

    for cell in cellprop:

        # isolate cell
        cell_mask = (frame == cell.label)

        # cell heights
        V_mean[cell_mask] = cell.mean_intensity * cell.area * xy_to_um**2

        # cell edges
        cell_interior = morph.erosion(cell_mask, footprint=morph.disk(1))
        edge = cell_mask ^ cell_interior
        e_im += edge

    e_im += (V_mean == 0)

    sns.heatmap(V_mean, ax=ax, square=True, cmap=V_cmap, vmin=Vmin, vmax=Vmax, xticklabels=False, yticklabels=False, cbar=True)
    sns.heatmap(e_im, ax=ax, cmap=e_cmap,  xticklabels=False, yticklabels=False, cbar=False)
    ax.set(title=f"Cell volume (µm³)")
    return 0


def plot_cell_velocity(ax, frame, velocity, pos, vmin=0, vmax=20, label='h (µm)'):

    sns.heatmap(frame, ax=ax, square=True, cmap="gray", vmin=vmin, vmax=vmax, 
                xticklabels=False, yticklabels=False, cbar=True, cbar_kws={'label':label})
    ax.quiver(pos[0], pos[1], velocity[0], -velocity[1], color="cyan", alpha=0.8, scale_units="xy", scale=0.15)

    return 0


def plot_field_velocity(ax, frame, v_field, pos, vmin=0, vmax=20, label='h (µm)'):

    #sns.heatmap(frame, ax=ax, square=True, cmap="gray", vmin=vmin, vmax=vmax, 
    #            xticklabels=False, yticklabels=False, cbar=True, cbar_kws={'label':label})
    ax.imshow(frame)
    ax.quiver(pos[0], pos[1], v_field[0], v_field[1], color="cyan", alpha=0.4)
    
    return 0



####################
# Perform plotting #
####################

def main():
    parser = argparse.ArgumentParser(description="Usage: python cell_segmentation_Holomonitor.py dir file")
    parser.add_argument("path",          type=str,   help="Path to data folder")
    parser.add_argument("func",          type=str,   help="Plotting function")
    parser.add_argument("-t", "--track_tresh", type=int,   help="first frame of tracked data (corresponds to threshold-1)", default=0)
    parser.add_argument("--fmin",        type=int,   help="First frame to plot", default=0)
    parser.add_argument("--Nframes",     type=int,   help="Number of frames to plot", default=9999)
    parser.add_argument("--figscale",    type=float, help="Scaleing of figure size. Default size is (10,8).", default=1)
    parser.add_argument("--edges",       type=bool,  help="Plot edges", default=False)
    parser.add_argument("--raw",         action="store_true", help="plotting raw data")
    parser.add_argument("--corrected",   action="store_true", help="plotting corrected data")
    parser.add_argument("--tracked",     action="store_true", help="plotting tracked data")
    parser.add_argument("--final",       action="store_true", help="plotting final data")
    parser.add_argument("--scale_area",  action="store_true", help="scale area with <A>_glattet")
    #parser.add_argument("--frames_to_hour", type=float, help="Conversion factor from frames to hours", default=1/12)
    parser.add_argument("-o", "--outdir",   type=str,   help="Output directory", default="")
    args = parser.parse_args()

    assert args.track_tresh > 0 or args.raw or args.corrected, "must provide track_thres"


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
    fmax     = config['segmentation']['fmax']
    Nframes  = fmax - fmin + 1

    if args.Nframes < Nframes:
        Nframes = args.Nframes

    if args.raw:
        im_cell_areas = np.load(f"data/experimental/processed/{dataset}/im_cell_areas_raw.npy")

    elif args.corrected:
        im_cell_areas = np.load(f"data/experimental/processed/{dataset}/im_cell_areas_corrected.npy")

    elif args.tracked:
        im_cell_areas = np.load(f"data/experimental/processed/{dataset}/im_cell_areas_tracked.npy")

    else:
        cellprop      = SegmentationData(f"data/experimental/processed/{dataset}/cell_props.p")
        im_cell_areas = np.load(f"data/experimental/processed/{dataset}/im_cell_areas_tracked.npy")


        if args.scale_area:
            A_mean_t = np.ma.mean(cellprop.A, axis=1)
            frames = np.arange(len(A_mean_t)) #* args.frame_to_hour
            fit = linregress(frames, A_mean_t)

            A_scale = fit.slope*frames + fit.intercept

            cellprop.A /= A_scale[:,np.newaxis]



    # Define path for output
    if args.outdir == "":
        outdir = f"figs/frames/{args.func}/{dataset}/"
        Path(outdir).mkdir(parents=True, exist_ok=True)
    else:
        outdir = args.outdir


    # set value range
    if microscope == "holomonitor":
        vmin = 0
        vmax = 20
    else:
        vmin = 1.33
        vmax = 1.4


    # loop through frames
    for frame in tqdm(range(args.fmin, Nframes+args.fmin)):

        # Import raw image
        if microscope == "holomonitor":
            im = imageio.v2.imread(f"data/experimental/raw/{dataset}/MDCK-li_reg_zero_corr_fluct_{fmin+frame+args.track_tresh}.tiff") / 100
        else:
            im = imageio.v2.imread(f"data/experimental/raw/{dataset}/MDCK-li_refractive_index_{fmin+frame}.tiff") / 10_000

        # Compute reigion props
        if args.raw or args.corrected or args.tracked:
            cellprop = regionprops(im_cell_areas[frame], im)
        

        # plot
        fig, ax = plt.subplots(1,1, figsize=(10*args.figscale, 8*args.figscale))

        if args.func == "cell_detection":

            if args.raw or args.corrected:
                positions = np.array([cell.centroid_weighted for cell in cellprop])
            else:
                positions = [cellprop.x[frame] / pix_to_um[1], cellprop.y[frame] / pix_to_um[1]]

            plot_cell_detection(ax, im, positions, vmin=vmin, vmax=vmax)


        elif args.func == "cell_height":

            plot_cell_height(ax, im_cell_areas[frame], cellprop)


        elif args.func == "cell_area":

            cellprop = regionprops(im_cell_areas[frame], im)
            plot_cell_area(ax, im_cell_areas[frame], cellprop, Ascale=A_scale[frame-args.fmin], xy_to_um=pix_to_um[1])


        elif args.func == "cell_volume":

            plot_cell_volume(ax, im_cell_areas[frame], cellprop, xy_to_um=pix_to_um[1])


        elif args.func == "cell_velocity":
            positions  = [cellprop.x[frame]  / pix_to_um[1], cellprop.y[frame]  / pix_to_um[1]]
            velocities = [cellprop.dx[frame] / pix_to_um[1], cellprop.dy[frame] / pix_to_um[1]]

            plot_cell_velocity(ax, im, velocities, positions)


        elif args.func == "field_velocity" or "PIV":
            # Load data
            positions, v_field = import_PIV_frame(f"data/experimental/PIV/{dataset}/", im, frame)

            plot_field_velocity(ax, im, v_field, positions)

        else:
            print("Error: func not recognized.")



        # add scalebar
        sb = ScaleBar(pix_to_um[-1], 'um', box_alpha=0, color="w", height_fraction=2e-2, scale_loc="none", fixed_value=100)
        sb.location = 'lower left'
        ax.add_artist(sb)

        # save
        if args.raw:
            sufix = "_raw"
        elif args.corrected:
            sufix = "_corrected"
        elif args.tracked:
            sufix = "_tracked"
        else:
            sufix = ""
        fig.tight_layout()
        plt.savefig(f"{outdir}/frame_{frame+args.track_tresh}{sufix}.png", dpi=300);
        plt.close()




if __name__ == "__main__":
    main()
