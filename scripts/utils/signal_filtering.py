import numpy as np
import scipy as sc

from scipy.signal import butter, filtfilt


# def replace_nan(areas, mask):
#     ''' Used to prepare masked arrays for filtering '''
#     for cell in range(len(areas[0])):
#         areas[:,cell][mask[:,cell]] = np.interp(np.flatnonzero(mask[:,cell]), np.flatnonzero(~mask[:,cell]), areas[:,cell][~mask[:,cell]])
    
#     return areas

def replace_nan(areas, mask):
    """Used to prepare masked arrays for filtering.

    areas: 2D array, shape (T, N)
    mask:  2D boolean array of same shape, True = needs filling
    """
    areas = np.asarray(areas).copy()
    mask = np.asarray(mask)

    n_cells = areas.shape[1]

    for cell in range(n_cells):
        col_mask = mask[:, cell]
        col = areas[:, cell]

        missing_idx = np.flatnonzero(col_mask)
        if missing_idx.size == 0:
            # nothing to fill in this column
            continue

        known_idx = np.flatnonzero(~col_mask)
        if known_idx.size == 0:
            # all values are masked; decide what to do
            # example: set to 0 or leave as is
            # here: leave them unchanged (or set to np.nan if you prefer)
            # areas[missing_idx, cell] = np.nan
            continue

        # If only one known point, interpolation is formally allowed but gives a flat line
        # If that’s okay, you can keep this; otherwise add a second check.
        areas[missing_idx, cell] = np.interp(
            missing_idx,
            known_idx,
            col[known_idx]
        )

    return areas


def butter_lowpass(cutoff, fs, order):
    ''' Return filter coefficient for lowpass filtering '''
    return butter(order, cutoff, fs=fs, btype='low', analog=False)


def butter_lowpass_filter(data, cutoff, fs, order):
    ''' Perform lowpass filtering. filtfilt does not pad data to zero. '''
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y


def apply_lowpass(areas, mask, cutoff, fs=12, order=6):
    ''' Collects three previous functions in one '''

    areas = np.ma.copy(areas)
    mask  = np.ma.copy(mask)
    
    # replace nans with linear interpolation
    areas = replace_nan(areas, mask)

    # Filter the data, and plot both the original and filtered signals.
    lowpass_filtered = butter_lowpass_filter(areas, cutoff, fs, order)
    lowpass_filtered = np.ma.array(lowpass_filtered, mask=mask)

    return lowpass_filtered
