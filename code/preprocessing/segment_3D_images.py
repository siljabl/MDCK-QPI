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

from utils.plots import *
from utils.segment3D  import *
from src.ImUtils import commonStackReader

import matplotlib
matplotlib.use("Agg")

parser = argparse.ArgumentParser(description='Segment cell from MlM probabilities')
parser.add_argument('dir',      type=str, help="path to main dir")
parser.add_argument('--r1_min',  type=int, help="min radius of kernel 1", default='3')
parser.add_argument('--r1_max',  type=int, help="max radius of kernel 1", default='5')
parser.add_argument('--r2',      type=int, help="radius of kernel 2", default='25')
args = parser.parse_args()

Nx = 912
Nz = 78

# create folders
in_dir = f"{args.dir}predictions"
print(in_dir)
assert os.path.exists(in_dir), "In path does not exist, maybe because of missing '/' at the end of path"
path = Path(in_dir)
out_dir  = f"{args.dir}segmentation"
mhds_dir = f"{out_dir}{os.sep}mhds"

try:
    os.mkdir(out_dir)
    os.mkdir(mhds_dir)
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
        ri_z_list   = []
        dri_dz_list = []
        
        # array for taking mean at specific tile over all frames
        mean_tiles = np.zeros([4, 4, Nz, Nx, Nx])
        z0_tiles   = np.zeros([4, 4])

        sum_above = np.zeros_like(thresholds)
        sum_below = np.zeros_like(thresholds)

        # compute list for determination of zero level
        print(f"\nDetermining zero-level for experiment {exp} ...")
        for file in path.glob(f"{exp}*_prob.npy"):

            stack_name = f"{path.parent}{os.sep}{file.name.split('_prob.npy')[0]}.tiff"
            frame = stack_name.split('_')[-1].split('.tiff')[0]
            print(f"Frame {frame}")

            # load stacks
            stack = commonStackReader(stack_name)
            MlM_probabilities = np.load(file)
            cell_prob = MlM_probabilities[:,:,:,1]

            # split up in tiles
            tiles = split_tiles(stack, mean_tiles)
            mean_tiles += tiles

            # compute base zero-level 
            mean_tile = np.zeros([Nz, Nx, Nx])
            for ix in range(4):
                for iy in range(4):
                    z0_tiles[ix,iy] = estimate_cell_bottom(mean_tiles[ix,iy])
            
            # collect tiles in one average tile
            z0_median = np.median(np.round(z0_tiles))

            for ix in range(4):
                for iy in range(4):
                    z_diff = int(z0_median - z0_tiles[ix,iy])
                    z_pad = abs(z_diff)

                    if z_pad > 0:
                        npad = ((z_pad, z_pad), (0, 0), (0, 0))
                        tile_zcorr = np.roll(np.pad(mean_tiles[ix,iy], pad_width=npad, mode="edge"), shift=z_diff, axis=0)[z_pad:-z_pad]

                        mean_tile += tile_zcorr / 16

                    elif z_pad == 0:
                        mean_tile += mean_tiles[ix,iy] / 16
            
            # detect tilt of dish
            z0_plane = estimate_cell_bottom(mean_tile, mode="plane")
            print(np.shape(z0_plane), z0_plane)


        # # compute zero level. same for entire experiment
        # z_0 = estimate_cell_bottom(dri_xdz_list, dri_ydz_list)
        # z_0 = median(z_0, disk(3))
        # plt.imshow(z_0)
        # plt.colorbar()
        # #fig = plot_z_profile([ri_xz_list, dri_xdz_list], stack, z_0)
        # plt.savefig(f"{mhds_dir}{os.sep}{Path(exp).name}_zero_level.png", dpi=300)
        # print(f"zero-level: {z_0}\n")


        # # determine threshold
        # print(f"Creating masks for:")
        # for file in path.glob(f"{exp}*_prob.npy"):
        #     out_mask = f"{out_dir}{os.sep}{file.name.split('_prob.npy')[0]}_mask.tiff"
        #     if os.path.exists(out_mask):
        #         continue
        #     print(file)

        #     # load probabilities
        #     MlM_probabilities = np.load(file)
        #     cell_prob = MlM_probabilities[:,:,:,1]

        #     # compute array for determination of MlM threshold
        #     for i in range(len(thresholds)):
        #         mask = (cell_prob > thresholds[i])
        #         sum_above[i] = np.sum(mask[z_0:])
        #         sum_below[i] = np.sum(mask[:z_0])
            
        #     # apply threshold
        #     threshold = determine_threshold(thresholds, sum_above)
        #     cell_pred = (cell_prob > threshold)
        #     log.write(f'{file.name}, {np.mean(z_0)}, {threshold}, {args.r1_min}, {args.r1_max}, {args.r2}\n')

        #     # filter mask
        #     tmp_mask = median(cell_pred[np.min(z_0):], kernel_1)
        #     tmp_mask = median(tmp_mask,  kernel_2)

        #     # make mask with heterogeneous zero-level
        #     cell_mask = np.zeros_like(cell_pred)
        #     for i in range(len(stack[0])):
        #         for j in range(len(stack[0,0])):
        #             cell_mask[:int(z_0[i,j]),i,j] = 0
        #             cell_mask[i,j] *= tmp_mask[i,j]
            
        #     # save mask
        #     basename = file.stem.split('_prob')[0]
        #     tifffile.imwrite(out_mask, np.array(cell_mask, dtype=np.uint8), bigtiff=True)


        # # plot illustration of MlM threshold and final mask
        # fig = plot_threshold(thresholds, [sum_above, sum_below], cell_prob.shape, np.mean(z_0))
        # fig.savefig(f"{mhds_dir}{os.sep}{file.name.split('_prob.npy')[0]}_threshold.png", dpi=300)

        # # save as pickle
        # out_dict = {'ri_xz_list':       ri_xz_list,
        #             'ri_yz_list':       ri_yz_list,
        #             'dri_xdz_list':     dri_xdz_list,
        #             'dri_ydz_list':     dri_ydz_list,
        #             'thresholds':       thresholds,
        #             'sum_above':        sum_above,
        #             'sum_below':        sum_below,
        #             'cell_prob_shape':  cell_prob.shape}
        
        # with open(f"{mhds_dir}{os.sep}/lists_for_plotting.pkl", 'wb') as handle:
        #     pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        


