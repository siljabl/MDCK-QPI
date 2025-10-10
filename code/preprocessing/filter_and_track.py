import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import trackpy as tp

from pathlib import Path
from datetime import datetime

from utils.file_handling import *
from utils.segment3D import get_voxel_size_35mm
from utils.segment2D import get_pixel_size

sys.path.append("code/analysis/utils")
from data_class import SegmentationData


parser = argparse.ArgumentParser(description="Usage: python segement_2D_images.py dir file")
parser.add_argument("path", type=str,    help="Path to dataset, typically data/experimental/processed/dataset/")
args = parser.parse_args()


# load sata
dataset = Path(args.path).stem
df     = pd.read_csv(f"data/experimental/processed/{dataset}/dataframe_unfiltered.csv")
config = json.load(open(f"data/experimental/configs/{dataset}.json"))


# set microscope specific parameters
microscope = Path(dataset).stem.split("_")[0]

if microscope == 'holomonitor':
    search_range = 10
    xy_to_um = get_pixel_size()[0]


elif microscope == 'tomocube':
    search_range = 50
    xy_to_um = get_voxel_size_35mm()[1]


# get filter parameters
A_min = config['filtering']['Amin']
A_max = config['filtering']['Amax']
V_min = config['filtering']['Vmin']
V_max = config['filtering']['Vmax']
h_min = config['filtering']['hmin']
h_max = config['filtering']['hmax']

# create mask
h_mask = (df.h_avrg > h_min) * (df.h_max < h_max) 
A_mask = (df.A > A_min)      * (df.A < A_max)
V_mask = (df.V > V_min)      * (df.V < V_max)

mask = h_mask * A_mask * V_mask

print(f"Before filtering: {len(df)} cells")
print(f"After filtering:  {len(df[mask])} cells")

# track cells
tracks = tp.link(df[mask], search_range=search_range, memory=5);
tracks = tp.filter_stubs(tracks, threshold=5);

# save as pickle
data_obj = SegmentationData()
data_obj.transform_df_to_ma(tracks, xy_to_um)
data_obj.save(f"data/experimental/processed/{dataset}/cell_props.p")

# save filter config
config['filtering']['date'] = datetime.today().strftime('%Y-%m-%d')

json.dump(config, open(f"data/experimental/configs/{dataset}.json", "w"))