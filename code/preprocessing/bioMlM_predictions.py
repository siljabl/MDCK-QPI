#!/usr/bin/python3


""" 
Author T. Combriat
Modified by: S.B. Låstad

Script to detect and generate biological matter and fake fluorescence in Tomocube images using a pre-trained model.
This performs the same than bioM_FL_img_segmenter except that it takes a full MmModel as an input.
This is the future
"""

import os
import pickle
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

from src.ImUtils import *
from src.MlModels import *
from src.FormatConversions import *

print(" * Welcome to the bioM Finders (keepers)!!\n\n")

parser = argparse.ArgumentParser(description="Usage: ./bioM_img_segmenter file model_name")
parser.add_argument("dir",           type=str, help="Path to data folder")
parser.add_argument("model_name",    type=str, help="Name of trained model to be used. *.pic")
parser.add_argument("--file_ending", type=str, help="Prompt to also return prediction probabilities", default="*HT3D_*.tiff")
args = parser.parse_args()

directory   = args.dir
model_name  = args.model_name
return_prob = True


####### ML Model loading
model_file = open(model_name,"rb")
model = pickle.load(model_file)
clf = model.clf


path = Path(directory)
output_dir = f"{directory}{os.sep}predictions"
try:
    os.mkdir(output_dir)
except:
    pass

with open(f"{output_dir}{os.sep}log.txt", "a") as log:
    log.write(f"date: {datetime.today().strftime('%Y-%m-%d')}\n")

    for file in path.glob(args.file_ending):
        # check if bioMlM has already been performed on file
        out_file = f"{output_dir}{os.sep}{os.path.splitext(os.path.basename(file))[0]}_prob.npy"
        if os.path.exists(out_file):
            print(f"{out_file} has already been classified!")
            continue
        
        # Import
        print(file)
        stack = commonStackReader(file)
        stack = convert_to_new_format(stack)
        stack = np.array(stack)

        # bioMlM
        cell_mask, cell_prob = detect_bioM_FL_MlMod_lowRam(stack, model, return_prob=return_prob)

        # Save
        np.save(out_file, np.array(cell_prob, dtype=np.float16))
        log.write(f"{os.path.basename(file)}, {model.thresholds}\n")
