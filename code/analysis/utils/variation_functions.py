import numpy as np


def spatial_variation(data):

    mean = np.mean(data[data > 0])
    rel_err = np.std(data[data > 0]) / mean

    return rel_err, mean


def global_density(masked_areas):
    N_cells = np.sum(masked_areas.mask==False, axis=1)
    A_cells = np.sum(masked_areas, axis=1)
    density = 10**6 * N_cells / A_cells

    return density


def average_cell_radius(df, frame, vox_to_um, blur_factor=1.5):
    frame_mask = (df.frame == frame)

    # total volume of all cells in pixels
    A_cells = np.sum(df[frame_mask].A) / (vox_to_um[-2]*vox_to_um[-1])
    N_cells = np.sum(frame_mask)
    A_cell = A_cells / N_cells

    # typical cell area x blur factor
    r_cell = int(blur_factor*np.sqrt(A_cell/np.pi))

    return r_cell