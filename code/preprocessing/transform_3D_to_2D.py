'''
Transform 3D mask to 2D tiff of height and mean refractive index.
'''
import os
import imageio
import argparse
import numpy as np

from pathlib  import Path
from datetime import datetime

from src.ImUtils import commonStackReader
from utils.plots import *
from utils.segment3D  import *


def correct_cell_bottom(mask, full_tile_size=958):
        
        # determine zero-level
        z0_ref = estimate_cell_bottom(mask)
        
        # splitting stack in tiles
        tiles = split_tiles(mask) 

        tile_dims    = np.shape(tiles)
        mean_tile_z0 = np.zeros([tile_dims[3], tile_dims[4]])
        weight_tile  = np.zeros([tile_dims[0], tile_dims[1]])

        # computing mean tile for fitting plane to cell bottom and weights to scale adjustment with amount of data below z0_ref
        for iy in range(4):
            for ix in range(4):
                mean_tile_z0 += tiles[iy, ix, z0_ref-1] / 16
                weight_tile[iy, ix] = np.sum(tiles[iy, ix, z0_ref-2:z0_ref]) / (tile_dims[3] * tile_dims[4])

        
        # combine z0_plane of each tile to one image
        z0_plane = fit_plane(mean_tile_z0, [full_tile_size, full_tile_size])
        z0_image = combine_tiles(z0_plane, weight_tile)

        new_mask = update_cell_mask(mask, z0_ref, z0_image)

        return new_mask



parser = argparse.ArgumentParser(description='Compute cell properties from segmented data')
parser.add_argument('dir',             type=str, help="path to main dir")
parser.add_argument('-m', '--method',  type=str, help="method for computing heights. 'sum' or 'diff'", default='sum')
parser.add_argument('-s', '--scaling', type=int, help="value that Tomocube data is scaled with",       default=10_000)
args = parser.parse_args()

n_cell = 1.38

# Folders
in_dir = f"{args.dir}segmentation"
assert os.path.exists(in_dir)
path = Path(in_dir)

fig_dir    = f"{path.parent}{os.sep}analysis"
height_dir = f"{path.parent}{os.sep}heights"
n_dir   = f"{path.parent}{os.sep}refractive_index"

try:
    os.mkdir(fig_dir)
    os.mkdir(n_dir)
    os.mkdir(height_dir)
except:
    pass


vox_to_um = get_voxel_size_35mm()

# get position specific dataseries names
positions = []
for file in path.glob("*_HT3D_0_mask.tiff"):
    positions.append(file.stem.split("_HT3D")[0])


# loop through all positions in folder
assert len(positions) > 0
for pos in positions:

    print(f"\nComputing histogram for {pos} ...")
    for file in path.glob(f"{pos}*"):
        stack_name = f"{path.parent}{os.sep}{file.name.split('_mask.tiff')[0]}.tiff"
        out_name   = file.name.split("_mask.tiff")[0]

        # load stacks
        stack = commonStackReader(stack_name)
        mask  = commonStackReader(file)
        mask  = correct_cell_bottom(mask) 

        # compute and save height
        im_heights = compute_height(mask, method=args.method)
        imageio.imwrite(f"{height_dir}{os.sep}{out_name}_heights.tiff", np.array(im_heights, dtype=np.uint8))

        # compute and save refractive index average in z
        im_n_avrg = refractive_index_uint16(stack, mask, im_heights)
        im_n_avrg = im_n_avrg - np.mean(im_n_avrg) + n_cell * args.scaling      # scale to have mean around 1.38
        imageio.imwrite(f"{n_dir}{os.sep}{out_name}_mean_refractive.tiff", np.array(im_n_avrg,  dtype=np.uint16))

config = {"date": datetime.today().strftime('%Y-%m-%d'),
          "vox_to_um": vox_to_um,
          "dims": im_heights.shape}

with open(f"{height_dir}/config.txt", 'w') as f:
    for key, value in config.items():  
        f.write('%s:%s\n' % (key, value))
