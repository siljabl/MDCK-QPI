import os
import sys
import pickle
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("Agg")

from pathlib import Path
from cmcrameri import cm

# Append the path of relative_parent directories
sys.path.append("code/analysis/utils")
from data_class import SegmentationData
from variation_functions import global_density

sys.path.append("code/preprocessing/utils")
from segment2D import get_pixel_size





def hist_to_curve(arr, bins=0, hist_range=None):
    if hist_range == None:
        hist_range = (np.ma.min(arr), np.ma.max(arr))

    if bins == 0:
        bins  = int(np.max(arr))

    y, x = np.histogram(arr, bins=bins, range=hist_range, density=True)

    return 0.5*(x[1:] + x[:-1]), y, bins



parser = argparse.ArgumentParser(description='Plot data set')
parser.add_argument('dir',            type=str,   help="Path to file to plot. Typically 'data/experimental/processed/<dataset>")
parser.add_argument('--bin_size',     type=int, help="data set as listed in holo_dict", default=200)
parser.add_argument('-b', '--binned',  action="store_true", help="bin data by density")
parser.add_argument('-r', '--rescale', action="store_true", help="rescale data")
parser.add_argument('--ylog',          action="store_true", help="rescale data")
args = parser.parse_args()



#######################
# import and bin data #
#######################
dataset = Path(args.dir).stem
cellprop = SegmentationData(f"{args.dir}cell_props.p")


# format conversion
pix_to_um = get_pixel_size()

# round density range to nearest 100
density = global_density(cellprop.A)

if args.binned == True:
    min_density = int(density.min() / 100) *100
    max_density = int(density.max() / 100) *100 + args.bin_size

    density_bins = np.arange(min_density, max_density, args.bin_size)

    # number of bins
    Nbins = len(density_bins) - 1

    height = []
    area   = []
    volume = []
    shape  = []


    for i in range(Nbins):

        # define low and high boundary on bin
        low_lim  = density_bins[i]   #min_density + args.bin_size * i
        high_lim = density_bins[i+1] #min_density + args.bin_size *(i+1)
        
        # mask relevant densities
        density_mask = (density >= low_lim) * (density < high_lim)

        heights_in_bin = cellprop.h[density_mask]
        areas_in_bin   = cellprop.A[density_mask]
        volumes_in_bin = heights_in_bin * areas_in_bin
        #shapes_in_bin  = ((data['major_axis'] / data['minor_axis']))[density_mask]

        height.append(heights_in_bin.ravel())
        area.  append(areas_in_bin.ravel())
        volume.append(volumes_in_bin.ravel())
        #shape. append(shapes_in_bin.ravel())
            
else:
    height = cellprop.h
    area   = cellprop.A
    volume = height * area

######################
# plot distributions #
######################
N = len(height)
colors  = cm.roma_r(np.linspace(0, 1, N))
hlow, hhigh = 0, 12
Alow, Ahigh = 0, 1500
Vlow, Vhigh = 0, 6000
bins = 20

fig, ax = plt.subplots(1,3, figsize=(12,3.5))

for i in range(len(height)):

    if args.rescale:
        height[i] = (height[i] - np.ma.mean(height[i], keepdims=True)) / np.ma.std(height[i], keepdims=True)
        area[i]   = (area[i]   - np.ma.mean(area[i],   keepdims=True)) / np.ma.std(area[i],   keepdims=True)
        volume[i] = (volume[i] - np.ma.mean(volume[i], keepdims=True)) / np.ma.std(volume[i], keepdims=True)

        hlow, hhigh = -3, 5
        Alow, Ahigh = -3, 5
        Vlow, Vhigh = -3, 5

        bins = 20

    h_x, h_y, bins = hist_to_curve(height[i], bins=bins, hist_range=(hlow, hhigh))
    A_x, A_y, bins = hist_to_curve(area[i],   bins=bins, hist_range=(Alow, Ahigh))
    V_x, V_y, bins = hist_to_curve(volume[i], bins=bins, hist_range=(Vlow, Vhigh))
    #p_x, p_y, bins = hist_to_curve(shape[i],  bins=30)

    ax[0].plot(h_x, h_y, '-', color=colors[i])
    ax[1].plot(A_x, A_y, '-', color=colors[i])
    ax[2].plot(V_x, V_y, '-', color=colors[i])
    #ax[1,1].plot(p_x, p_y, '-', color=colors[i])

   
if args.rescale:
    ax[0].set(xlabel=r"$(h-\bar{h}) ~/~ \sigma_h$",   ylabel="PDF")
    ax[1].set(xlabel=r"$(A-\bar{A}) ~/~ \sigma_A$")
    ax[2].set(xlabel=r"$(V-\bar{V}) ~/~ \sigma_V$")

else:
    ax[0].set(xlabel=r"$h ~[µm]$",   ylabel="PDF")
    ax[1].set(xlabel=r"$A ~[µm^2]$")
    ax[2].set(xlabel=r"$V ~[µm^3]$")

if args.ylog:
    for i in range(3):
        ax[i].set(yscale="log")

fig.tight_layout()
fig.subplots_adjust(right=0.85)

cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
sm      = plt.cm.ScalarMappable(cmap=cm.roma_r, norm=plt.Normalize(vmin=density.min(), vmax=density.max()))

fig.colorbar(sm, cax=cbar_ax, label=r"$\rho_{cell}(t) ~[mm^{-2}]$")
fig.savefig(f"results/{dataset}/height_area_volume_distributions.png", dpi=300)



# ##################
# # Save as pickle #
# ##################
# out_dict = {'area':    area,
#             'height':  height,
#             'volume':  volume,
#             'shape':   shape,
#             'density': density_bins}



# # if not given, use input folder for outpur also
# if args.out_path == None:
#     args.out_path = args.in_path

# # create figure folder
# try:
#     os.mkdir(f"{args.out_path}/figs")
# except:
#     None

# # save as pickle
# with open(f"{args.out_path}/hAV_distributions.pkl", 'wb') as handle:
#     pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

