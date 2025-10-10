import numpy as np


def average_cell_radius(masked_areas):
    """ In µm """

    # total volume of all cells in pixels
    N_cells = np.sum(masked_areas.mask==False, axis=1)
    A_cells = np.sum(masked_areas, axis=1)
    A_cell = A_cells / N_cells

    # typical cell area
    r_cell = [np.sqrt(A/np.pi) for A in A_cell]

    return np.array(r_cell)


def global_density(masked_areas):
    N_cells = np.sum(masked_areas.mask==False, axis=1)
    A_cells = np.sum(masked_areas, axis=1)
    density = 10**6 * N_cells / A_cells

    return density