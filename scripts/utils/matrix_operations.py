import numpy as np
from scipy.stats import linregress


# def average_cell_radius(masked_areas):
#     """ In µm """

#     # total volume of all cells in pixels
#     dims = np.shape(masked_areas)

#     if len(dims) > 1:
#         N_cells = np.sum(masked_areas.mask==False, axis=1)
#         A_cells = np.sum(masked_areas, axis=1)

#         A_cell = A_cells / N_cells
#         r_cell = [np.sqrt(A/np.pi) for A in A_cell]


#     else:
#         N_cells = np.sum(masked_areas.mask==False)
#         A_cells = np.sum(masked_areas)

#         A_cell = A_cells / N_cells
#         r_cell = np.sqrt(A_cell/np.pi)

#     return np.array(r_cell)


# def global_density(masked_areas):
#     dims = np.shape(masked_areas)

#     if len(dims) > 1:
#         N_cells = np.sum(masked_areas.mask==False, axis=1)
#         A_cells = np.sum(masked_areas, axis=1)

#     else:
#         N_cells = np.sum(masked_areas.mask==False)
#         A_cells = np.sum(masked_areas)

#     density = 10**6 * N_cells / A_cells

#     return density




def detrend_array(t_arr, arr, keepdims=False):
    fit = linregress(t_arr[~arr.mask], arr.data[~arr.mask])

    if keepdims == False:
        lin_fit = t_arr[~arr.mask]*fit.slope
        return arr.data[~arr.mask]-lin_fit, np.ma.std(arr.data[~arr.mask]-lin_fit) / np.ma.mean(arr.data[~arr.mask])
    
    else:
        lin_fit = t_arr*fit.slope + fit.intercept
        return arr + np.ma.mean(arr) - lin_fit, np.ma.std(arr-lin_fit) / np.ma.mean(arr)
    


def detrend_entire_matrix(arr):
    
    time = np.arange(len(arr))
    detrended_arr = []

    for cell_area in arr.T:
        A_detrended, _  = detrend_array(time, cell_area, keepdims=True)

        detrended_arr.append(A_detrended)

    detrended_arr = np.ma.array(detrended_arr).T

    return detrended_arr
