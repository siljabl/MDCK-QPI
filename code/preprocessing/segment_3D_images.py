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
Nframes = 41

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
        mean_tiles     = np.zeros([Nframes, Nz, Nx, Nx])        # The average of all 16 tiles in one frame
        mean_prob_tile = np.zeros([Nz, Nx, Nx])                 # The average of all 16 probability tiles in one frame
        mean_tile      = np.zeros([Nz, Nx, Nx])                 # The average of the average tile of all frames
        z0_tiles       = np.zeros([Nframes, 4, 4])              # The average zero-level in all tiles in one frame
        z0_arr         = np.zeros(Nframes)                      # The average zero-level in all frames

        sum_above = np.zeros_like(thresholds)
        sum_below = np.zeros_like(thresholds)
        threshold = np.zeros(Nframes)

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

            # split up in tiles for individual detection of zero-level
            tiles      = split_tiles(stack,     Nz=Nz, Nx=Nx)
            prob_tiles = split_tiles(cell_prob, Nz=Nz, Nx=Nx)

            # find mean zero level of each tile
            for ix in range(4):
                for iy in range(4):
                    z0_tiles[frame, ix,iy] = estimate_cell_bottom(tiles[ix,iy])

            # adjust zslice between tiles so all have same zero-level as z0_median[frame]
            z0_arr[frame] = int(np.round(np.median(z0_tiles[frame])))

            for ix in range(4):
                for iy in range(4):
                    tile_zcorr      = correct_zslice_tile(tiles[ix,iy],      z0_tiles[frame, ix, iy], z0_arr[frame])
                    tile_prob_zcorr = correct_zslice_tile(prob_tiles[ix,iy], z0_tiles[frame, ix, iy], z0_arr[frame])

                    mean_tiles[frame] += tile_zcorr / 16
                    mean_prob_tile    += tile_prob_zcorr / 16
            
                
            # compute array for determination of MlM threshold
            for i in range(len(thresholds)):
                mask = (mean_prob_tile > thresholds[i])
                print(z0_arr[frame])
                sum_above[i] = np.sum(mask[int(z0_arr[frame]):])
                sum_below[i] = np.sum(mask[:int(z0_arr[frame])])

            # compute threshold
            print(np.shape(threshold))
            threshold[frame] = determine_threshold(thresholds, sum_above)


        # adjust zslice between frames so all have same zero-level as z0
        z0 = np.round(np.median(z0_arr))

        for f in range(Nframes):
            tile_zcorr = correct_zslice_tile(mean_tiles[f], z0_arr[frame], z0)
            mean_tile += tile_zcorr / Nframes

        # detect tilt of dish
        z0_points = estimate_cell_bottom(mean_tile, mode="plane")
        z0_plane  = fit_plane(z0_points).reshape(Nx, Nx)

        # saving where?
        np.save("z0_plane.npy", z0_plane)
        log.write(f'{file.name}, {np.mean(z0_arr[frame])}, {threshold}, {args.r1_min}, {args.r1_max}, {args.r2}\n')

        # finding lower limit on zero-level
        z0_cutoff_tiles = z0_tiles - z0 + np.min(np.floor(z0_plane))

        # # compute zero level. same for entire experiment
        # z_0 = estimate_cell_bottom(dri_xdz_list, dri_ydz_list)
        # z_0 = median(z_0, disk(3))
        # plt.imshow(z_0)
        # plt.colorbar()
        # #fig = plot_z_profile([ri_xz_list, dri_xdz_list], stack, z_0)
        # plt.savefig(f"{mhds_dir}{os.sep}{Path(exp).name}_zero_level.png", dpi=300)
        # print(f"zero-level: {z_0}\n")


        # determine threshold
        print(f"Creating masks for:")
        for file in path.glob(f"{exp}*_prob.npy"):
            out_mask = f"{out_dir}{os.sep}{file.name.split('_prob.npy')[0]}_mask.tiff"
            if os.path.exists(out_mask):
                continue
            frame = int(out_mask.split('_')[-2])
            print(file)
            print(f"Frame {frame}")

            # load probabilities
            MlM_probabilities = np.load(file)
            cell_prob  = MlM_probabilities[:,:,:,1]

            # apply threshold and split 
            cell_pred  = (cell_prob > threshold)
            tiles_pred = split_tiles(cell_pred, Nz=Nz, Nx=Nx)

            # filter mask
            cell_mask = np.zeros_like(cell_prob)

            for ix in range(4):
                for iy in range(4):

                    # finding lower limit on zero-level
                    z0_cutoff = int(z0_tiles[frame, ix, iy] - z0 + np.min(np.floor(z0_plane)))
                    print(z0_cutoff)

                    # apply median filter twice (?)
                    tmp_mask = median(tiles_pred[ix, iy, z0_cutoff:], kernel_1)
                    tmp_mask = median(tmp_mask,  kernel_2)

                    # make mask with heterogeneous zero-level
                    cell_mask[:z0_cutoff, ix*Nx:(ix+1)*Nx, iy*Nx:(iy+1)*Nx] = 0
                    cell_mask[:,          ix*Nx:(ix+1)*Nx, iy*Nx:(iy+1)*Nx] *= tmp_mask
                    
            # save mask
            basename = file.stem.split('_prob')[0]
            tifffile.imwrite(out_mask, np.array(cell_mask, dtype=np.uint8), bigtiff=True)


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
        


