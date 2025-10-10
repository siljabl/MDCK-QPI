import os
import imageio
import numpy as np

from pathlib import Path
from segment3D import get_voxel_size_35mm


def import_holomonitor_stack(dir, h_scaling=100, f_min=1, f_max=180):
    """
    Import tiff files to one list

    Parameters:
    - dir: directory of tiff files.
    - h_scaling: converts height to µm
    - f_min: first frame
    - f_max: last frame
    """

    # mask that set area outside cells to zero
    try:
        mask = (imageio.v2.imread(f"{dir}/mask.tiff") > 0)
    except:
        try:
            mask = np.ones_like(imageio.v2.imread(f"{dir}/MDCK-li_reg_zero_corr_fluct_{1}.tiff"))
        except:
            mask = np.ones_like(imageio.v2.imread(f"{dir}/MDCK-li_reg_zero_corr_{1}.tiff"))
    
    stack = []
    for f in range(f_min, f_max+1):

        try:
            frame = imageio.v2.imread(f"{dir}/MDCK-li_reg_zero_corr_fluct_{f}.tiff")
        except:
            frame = imageio.v2.imread(f"{dir}/MDCK-li_reg_zero_corr_{f}.tiff")
            print("fail", f)

        stack.append(frame * mask)

    stack = np.array(stack) / h_scaling

    return stack



def import_tomocube_stack(dir, h_scaling, f_min=0, f_max=40, n_cell=1.38):
    """
    Import tiff files to one list

    Parameters:
    - dir: directory of tiff files.
    - h_scaling: converts height to µm
    - f_min: first frame
    - f_max: last frame
    - n_cell: mean refractive index of cells
    """

    n_stack = []
    h_stack = []
    for f in range(f_min, f_max+1):
        h_frame = imageio.v2.imread(f"{dir}/MDCK-li_height_{f}.tiff")
        n_frame = imageio.v2.imread(f"{dir}/MDCK-li_refractive_index_{f}.tiff")

        h_stack.append(h_frame)
        n_stack.append(n_frame)

    # scale data
    h_stack = np.array(h_stack) * h_scaling
    n_stack = np.array(n_stack)
    n_mean  = np.mean(n_stack[h_stack > 0])
    n_stack = n_stack * n_cell / n_mean

    return n_stack, h_stack



def import_txt_with_NaN(input_file, header_rows=3):
    with open(input_file, 'r') as file:

        # Skip the first three lines
        for _ in range(header_rows):
            next(file)
        
        # Read the remaining lines and replace 'NaN'
        output_file = []

        # Process the remaining rows
        for line in file:
            # Split the line into values (assuming a whitespace delimiter)
            values = line.strip().split(",")

            # Replace "NaN" with the specified replacement value and convert to float
            processed_row = [
                float(value) if value != "NaN" else 9999.0
                for value in values
            ]
            output_file.append(processed_row)
    
    output_file = np.array(output_file)
    output_file = np.ma.array(output_file, mask=output_file==9999.0)

    return output_file




def import_stack(dir, config):
    microscope = Path(dir).stem.split("_")[0]

    f_min = config['segmentation']['fmin']
    f_max = config['segmentation']['fmax']

    if microscope == 'holomonitor':
        h_stack = import_holomonitor_stack(dir, f_min=f_min, f_max=f_max)

    elif microscope == 'tomocube':
        pix_to_um = get_voxel_size_35mm()
        _, h_stack = import_tomocube_stack(dir, h_scaling=pix_to_um[0], f_min=f_min, f_max=f_max)

    else:
        print("microscope not recognized: ", microscope)
        h_stack = 0
        
    return h_stack




def import_PIV_velocity(path, config, _shape):

    # import PIV velocity field
    data_PIV = import_txt_with_NaN(f"{path}/PIVlab_0001.txt", header_rows=3)
    x = np.ma.array(data_PIV[:, 0], dtype=int)
    y = np.ma.array(data_PIV[:, 1], dtype=int)

    # import heights to mask image
    dataset = Path(path).stem
    stack = import_stack(f"data/experimental/raw/{dataset}", config)

    # transform PIV position to matrix entries
    dx   = y[1] - y[0]
    x0   = y[0] - dx
    xmax = int((np.ma.max(y) - x0) / dx) + 1

    x_tmp = np.ma.array((x - x0) / dx, dtype=int)
    y_tmp = np.ma.array((y - x0) / dx, dtype=int)

    position = np.ma.zeros((2, _shape[0], xmax, xmax), dtype=np.float64)
    velocity = np.ma.zeros((2, _shape[0], xmax, xmax), dtype=np.float64)

    h_mask   = np.ma.zeros((_shape[0], xmax, xmax), dtype=np.float64)

    i = 0
    for i in range(1, _shape[0]):
        
        # Load data
        data_PIV = import_txt_with_NaN(f"{path}/PIVlab_{i:04d}.txt", header_rows=3)

        # Extract values
        u = np.ma.array(data_PIV[:, 2], dtype=np.float64)
        v = np.ma.array(data_PIV[:, 3], dtype=np.float64)

        # masking didn't work properly, so probably mean is affected by outside data points
        position[:, i-1, x_tmp, y_tmp] = [x, y]
        velocity[:, i-1, x_tmp, y_tmp] = [u, v]
        
        h_mask[i-1, x_tmp, y_tmp] = stack[i-1, x, y]

    velocity[0][h_mask <= 0] = 0
    velocity[1][h_mask <= 0] = 0


    return position, velocity
