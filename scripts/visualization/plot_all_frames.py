"""
Script to plot frames that illustrates data analysis
func:
- cell_detection
- field_velocity or PIV
- orientation
"""

# import os
import sys
import json
# import pickle
# import imageio
import argparse
import numpy as np
# import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import skimage.morphology as morph
from skimage.measure import regionprops
from matplotlib_scalebar.scalebar import ScaleBar


from tqdm import tqdm
from pathlib import Path
from cmcrameri import cm

sys.path.append("scripts/utils")
from file_operations import *
from Microscopes     import Tomocube, Holomonitor
from SegmentedCells  import SegmentedCells


data_path = "../../../../hdd_data/silja/Monolayers/"

# area colourbar
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm,\
    LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection, LineCollection
cmap_yketa = (                                                               # colourmap
    LinearSegmentedColormap.from_list("aroace", (                           # https://en.wikipedia.org/wiki/Pride_flag#/media/File:Aroace_flag.svg
        (0/4, (0.125, 0.220, 0.337)),
        (1/4, (0.384, 0.682, 0.863)),
        (2/4, (1.000, 1.000, 1.000)),
        (3/4, (0.925, 0.804, 0.000)),
        (4/4, (0.886, 0.549, 0.000))))
    or plt.cm.bwr)
norm_area = Normalize(0, 10)                                                # interval of value represented by colourmap
scalarMap_yketa = ScalarMappable(norm_area, cmap_yketa)

VMAX = 14

######################
# Plotting functions #
######################

def plot_cell_detection(ax, frame, pos, vmin=0, vmax=VMAX, label='h (µm)'):
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
    
    ax.plot(pos[0].T, pos[1].T, 'r.', ms=5)
    # ax.set(title=f"#cells: {len(pos[0])}")
    #ax.set(title=f"#cells: {np.sum(pos[0].mask==False)}")



def plot_height_field(ax, frame, time, microscope, vmin=0, vmax=VMAX):
    
    sns.set_theme(style='ticks', palette='bright', font_scale=4)
    sns.heatmap(frame, square=True, cmap=cm.batlowW, xticklabels=False, yticklabels=False, vmin=vmin, vmax=vmax)

    sb = ScaleBar(microscope.pix_to_um, 'um', box_alpha=0, color="w", height_fraction=2e-2, scale_loc="none", fixed_value=100)
    sb.location = 'lower left'
    ax.add_artist(sb)
    ax.set(title=rf"h(x,y) µm,    t = {time * microscope.frame_to_h:0.2f} h")



def plot_ri_field(ax, frame, time, microscope, vmin=1.33, vmax=1.43):
    
    sns.set_theme(style='ticks', palette='bright', font_scale=4)
    sns.heatmap(frame, square=True, cmap="rocket", xticklabels=False, yticklabels=False, vmin=vmin, vmax=vmax)

    sb = ScaleBar(microscope.pix_to_um, 'um', box_alpha=0, color="w", height_fraction=2e-2, scale_loc="none", fixed_value=100)
    sb.location = 'lower left'
    ax.add_artist(sb)
    ax.set(title=rf"n(x,y),    t = {time * microscope.frame_to_h:0.2f} h")



def plot_3D_height(ax, X, Y, Z, vmin=0, vmax=VMAX):

    ax.set(zlim=(0,vmax+1))
    ax.set_zticks([])

    ax.plot_surface(X, 
                    Y, 
                    Z,
                    rstride=10,
                    cstride=10,
                    antialiased=True,
                    linewidth=0, 
                    cmap=cm.batlowW,
                    vmin=vmin,
                    vmax=vmax)





# def plot_cell_height(ax, frame, cellprop, hmean=0, hmin=0, hmax=10):
#     """ 
#     Plotting imshow of segmented cell areas with faces colors by average height

#     Parameters:
#     ax:       ax object to plot data on
#     frame:    frame of raw data to plot as heatmap
#     cellprop: list of regionprop of all cells in frame
#     hmin:     min height in heatmap
#     hmax:     max height in heatmap
#     """
    
#     h_cmap = sns.color_palette("Blues",   as_cmap=True)
#     e_cmap = mpl.colors.ListedColormap(['none', 'w'])

#     h_mean = np.ones_like(frame, dtype=np.float64) * hmean
#     e_im   = np.zeros_like(frame, dtype=int)

#     for cell in cellprop:

#         # isolate cell
#         cell_mask = (frame == cell.label)

#         # cell heights
#         h_mean[cell_mask] = cell.mean_intensity

#         # cell edges
#         cell_interior = morph.erosion(cell_mask, footprint=morph.disk(1))
#         edge = cell_mask ^ cell_interior
#         e_im += edge

#     e_im += (h_mean == 0)

#     #scalarMap_yketa.set_array(h_mean)
#     #sns.heatmap(h_mean, ax=ax, square=True, vmin=hmin, cmap=scalarMap_yketa.cmap, vmax=hmax, xticklabels=False, yticklabels=False, cbar=True)
#     sns.heatmap(h_mean, ax=ax, square=True, vmin=hmin, cmap=h_cmap, vmax=hmax, xticklabels=False, yticklabels=False, cbar=True)
#     sns.heatmap(e_im, ax=ax, cmap=e_cmap,  xticklabels=False, yticklabels=False, cbar=False)
#     ax.set(title=f"Average cell height (µm)")

#     return 0



# def plot_cell_mass_concentration(ax, frame, cellprop, cdmean=0.12, cdmin=0, cdmax=0.25):
#     """ 
#     Plotting imshow of segmented cell areas with faces colors by average height

#     Parameters:
#     ax:       ax object to plot data on
#     frame:    frame of raw data to plot as heatmap
#     cellprop: list of regionprop of all cells in frame
#     hmin:     min height in heatmap
#     hmax:     max height in heatmap
#     """
    
#     cd_cmap = sns.color_palette("rocket",   as_cmap=True)
#     e_cmap  = mpl.colors.ListedColormap(['none', 'w'])

#     cd_mean = np.ones_like(frame, dtype=np.float64) * cdmean
#     e_im    = np.zeros_like(frame, dtype=int)

#     for cell in cellprop:

#         # isolate cell
#         cell_mask = (frame == cell.label)

#         # cell heights
#         cd_mean[cell_mask] = cell.mean_intensity

#         # cell edges
#         cell_interior = morph.erosion(cell_mask, footprint=morph.disk(1))
#         edge = cell_mask ^ cell_interior
#         e_im += edge

#     e_im += (cd_mean == 0)

#     #scalarMap_yketa.set_array(h_mean)
#     #sns.heatmap(h_mean, ax=ax, square=True, vmin=hmin, cmap=scalarMap_yketa.cmap, vmax=hmax, xticklabels=False, yticklabels=False, cbar=True)
#     sns.heatmap(cd_mean, ax=ax, square=True, vmin=cdmin, cmap=cd_cmap, vmax=cdmax, xticklabels=False, yticklabels=False, cbar=True)
#     sns.heatmap(e_im, ax=ax, cmap=e_cmap,  xticklabels=False, yticklabels=False, cbar=False)
#     ax.set(title=f"Average mass concentration (g/ml)")

#     return 0



# def plot_cell_area(ax, frame, cellprop, Ascale, Amean=0, Amin=-1.1, Amax=1.1, xy_to_um=0.15):
#     """ 
#     Plotting imshow of segmented cell areas with faces colors by average height

#     Parameters:
#     ax:       ax object to plot data on
#     frame:    frame of raw data to plot as heatmap
#     cellprop: list of regionprop of all cells in frame
#     hmin:     min height in heatmap
#     hmax:     max height in heatmap
#     """
    
#     A_cmap = sns.color_palette("Oranges",   as_cmap=True)
#     e_cmap = mpl.colors.ListedColormap(['none', 'w'])

#     A_cell = np.ones_like(frame, dtype=np.float64) * Amean / Ascale
#     e_im   = np.zeros_like(frame, dtype=int)

#     for cell in cellprop:

#         # isolate cell
#         cell_mask = (frame == cell.label)

#         # cell areas
#         A_cell[cell_mask] = np.log(cell.area * xy_to_um**2 / Ascale)

#         # cell edges
#         cell_interior = morph.erosion(cell_mask, footprint=morph.disk(1))
#         edge = cell_mask ^ cell_interior
#         e_im += edge

#     e_im += (A_cell == 0)

#     sns.heatmap(A_cell, ax=ax, square=True, cmap=A_cmap, vmin=Amin, vmax=Amax, xticklabels=False, yticklabels=False, cbar=True)
#     sns.heatmap(e_im, ax=ax, cmap=e_cmap,  xticklabels=False, yticklabels=False, cbar=False)
#     ax.set(title=r"$\log{[A_{cell} ~/~ \langle A \rangle_{t, linear}]}$")

#     return 0



# def plot_cell_volume(ax, frame, cellprop, Vmean=0, Vmin=600, Vmax=8000, xy_to_um=0.15):
#     """ 
#     Plotting imshow of segmented cell areas with faces colors by the cell volume

#     Parameters:
#     ax:       ax object to plot data on
#     frame:    frame of raw data to plot as heatmap
#     cellprop: list of regionprop of all cells in frame
#     Vmin:     min volume in heatmap
#     Vmax:     max volume in heatmap
#     """
        
#     V_cmap = sns.color_palette("Greens",   as_cmap=True)
#     e_cmap = mpl.colors.ListedColormap(['none', 'w'])

#     V_cell = np.ones_like(frame, dtype=np.float64) * Vmean
#     e_im   = np.zeros_like(frame, dtype=int)

#     for cell in cellprop:

#         # isolate cell
#         cell_mask = (frame == cell.label)

#         # cell heights
#         V_cell[cell_mask] = cell.mean_intensity * cell.area * xy_to_um**2

#         # cell edges
#         cell_interior = morph.erosion(cell_mask, footprint=morph.disk(1))
#         edge = cell_mask ^ cell_interior
#         e_im += edge

#     e_im += (V_cell == 0)

#     sns.heatmap(V_cell, ax=ax, square=True, cmap=V_cmap, vmin=Vmin, vmax=Vmax, xticklabels=False, yticklabels=False, cbar=True)
#     sns.heatmap(e_im, ax=ax, cmap=e_cmap,  xticklabels=False, yticklabels=False, cbar=False)
#     ax.set(title=f"Cell volume (µm³)")
#     return 0


# def plot_cell_velocity(ax, frame, velocity, pos, vmin=0, vmax=20, label='h (µm)'):

#     sns.heatmap(frame, ax=ax, square=True, cmap="gray", vmin=vmin, vmax=vmax, 
#                 xticklabels=False, yticklabels=False, cbar=True, cbar_kws={'label':label})
#     ax.quiver(pos[0], pos[1], velocity[0], -velocity[1], color="cyan", alpha=0.8, scale_units="xy", scale=0.15)

#     return 0



# def plot_orientation(ax, im_raw, im_seg, im_cellprop, orientation, pos, vmin=0, vmax=20, vmean=0, label='h (µm)', background="im"):

#     if background == "im":
#         sns.heatmap(im_raw, ax=ax, square=True, cmap="gray", vmin=vmin, vmax=vmax, 
#                     xticklabels=False, yticklabels=False, cbar=True, cbar_kws={'label':label})
        
#     else:
#         h_cmap = sns.color_palette("Grays_r",   as_cmap=True)
#         e_cmap = mpl.colors.ListedColormap(['none', 'w'])

#         h_mean = np.ones_like(im_seg, dtype=np.float64) * vmean
#         e_im   = np.zeros_like(im_seg, dtype=int)

#         for cell in im_cellprop:

#             # isolate cell
#             cell_mask = (im_seg == cell.label)

#             # cell heights
#             h_mean[cell_mask] = cell.mean_intensity

#             # cell edges
#             cell_interior = morph.erosion(cell_mask, footprint=morph.disk(1))
#             edge = cell_mask ^ cell_interior
#             e_im += edge

#         e_im += (h_mean == 0)

#         sns.heatmap(h_mean, ax=ax, square=True, cmap=h_cmap, vmin=vmin, vmax=vmax, xticklabels=False, yticklabels=False, cbar=True)
#         sns.heatmap(e_im, ax=ax, cmap=e_cmap,  xticklabels=False, yticklabels=False, cbar=False)

    
#     ax.quiver(pos[0], pos[1], -orientation[0], orientation[1], color="cyan", alpha=0.8, scale_units="dots", scale=1/20, headaxislength=0, headlength=0, pivot="mid")

#     return 0



# def plot_field_velocity(ax, frame, v_field, pos, vmin=0, vmax=20, label='h (µm)'):

#     #sns.heatmap(frame, ax=ax, square=True, cmap="gray", vmin=vmin, vmax=vmax, 
#     #            xticklabels=False, yticklabels=False, cbar=True, cbar_kws={'label':label})
#     ax.imshow(frame)
#     ax.quiver(pos[0], pos[1], v_field[0], v_field[1], color="cyan", alpha=0.4)
    
#     return 0



####################
# Perform plotting #
####################

def main():
    parser = argparse.ArgumentParser(description="Usage: python cell_segmentation_Holomonitor.py dir file")
    parser.add_argument("path",          type=str,   help="path to config file")
    parser.add_argument("func",          type=str,   help="Plotting function")
    parser.add_argument("--hmax",        type=float,   help="Upper limit on colorbar", default=10)
    parser.add_argument("--figscale",    type=float, help="Scaleing of figure size. Default size is (10,8).", default=1)
    parser.add_argument("--edges",       type=bool,  help="Plot edges", default=False)
    parser.add_argument("--scale_area",  action="store_true", help="scale area with <A>_glattet")
    #parser.add_argument("--frames_to_hour", type=float, help="Conversion factor from frames to hours", default=1/12)
    parser.add_argument("-o", "--outdir",   type=str,   help="Output directory", default="")
    args = parser.parse_args()



    # Decompose input
    dataset    = Path(args.path).stem
    config   = json.load(open(f"configs/{dataset}.json"))

    ### SET MICROSCOPE ###
    if config['microscope'] == "holomonitor": microscope = Holomonitor()
    elif config['microscope'] == "tomocube":  microscope = Tomocube()


    cells = SegmentedCells(f"{data_path}cell_features/calibrated/{dataset}_cells.p")


    # set value range
    if microscope.name == "holomonitor":
        vmin = 0
        vmax = 20
    else:
        vmin = 1.35
        vmax = 1.39


    # loop through frames
    for frame in tqdm(range(config['cells']['fmin'],
                            config['cells']['fmax'])):
        f_idx = frame - config['cells']['fmin']
        # Import raw image
        h_im, n_im = load_set_of_frames(f"{data_path}height_fields/calibrated/{dataset}", frame, microscope)


        # plot
        if args.func != "3D":
            fig, ax = plt.subplots(1,1, figsize=(10*args.figscale, 8*args.figscale))

        if args.func == "cell_detection":
            positions = np.array([cells.x[f_idx] / microscope.pix_to_um, cells.y[f_idx] / microscope.pix_to_um])
            plot_cell_detection(ax, n_im, positions, vmin=vmin, vmax=vmax)


        if args.func == "height_field":
            plot_height_field(ax, h_im, frame, microscope, vmax=VMAX)


        if args.func == "ri_field":
            plot_ri_field(ax, n_im, frame, microscope, vmin=1.34, vmax=1.40)


        if args.func == "3D":
            font  = {'size': 24}
            mpl.rc('font', **font)
            fig, ax = plt.subplots(figsize=(32,18), subplot_kw={"projection": "3d"})
            Z    = h_im
            X, Y = np.meshgrid(np.arange(3648) * microscope.pix_to_um,
                               np.arange(3648) * microscope.pix_to_um)

            ax.set_box_aspect(((np.ptp(X)), (np.ptp(Y)), VMAX+1))
            plot_3D_height(ax, X, Y, Z)

            ax.set_xlabel(r"$x~(µm)$", fontsize=24, labelpad=50)
            ax.set_ylabel(r"$y~(µm)$", fontsize=24, labelpad=50)
            
            m = mpl.cm.ScalarMappable(cmap=cm.batlowW)
            m.set_array(np.linspace(0, VMAX, 100))

            cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(m, cax=cbar_ax)
            cbar.set_label(label=r"$h ~(µm)$", fontsize=28)
            fig.subplots_adjust(left=-0.25, right=1.2, top=1.25, bottom=-0.25)



        # elif args.func == "cell_height":

        #     plot_cell_height(ax, im_cell_areas[frame], cellprop, hmax=args.hmax)


        # elif args.func == "cell_concentration":

        #     im_cellprop = regionprops(im_cell_areas[frame], (im-1.33) / (np.mean(im)-1.33) - 1)
        #     plot_cell_mass_concentration(ax, im_cell_areas[frame], im_cellprop, cdmean=0, cdmin=-.1, cdmax=0.1)


        # elif args.func == "cell_area":

        #     im_cellprop = regionprops(im_cell_areas[frame], im)
        #     plot_cell_area(ax, im_cell_areas[frame], im_cellprop, Amean=np.ma.mean(cellprop.A[frame]), Ascale=A_scale[frame-args.fmin], xy_to_um=pix_to_um[1])


        # elif args.func == "cell_volume":

        #     plot_cell_volume(ax, im_cell_areas[frame], cellprop, xy_to_um=pix_to_um[1])


        else:
            print("Error: func not recognized.")


        fig.tight_layout()
        plt.savefig(f"figs/frames/frame_{frame}.png", dpi=300);
        plt.close()




if __name__ == "__main__":
    main()
