import numpy as np
import scipy as sc

from scipy.signal import butter, filtfilt


def replace_nan(areas, mask):
    ''' Used to prepare masked arrays for filtering '''
    for cell in range(len(areas[0])):
        areas[:,cell][mask[:,cell]] = np.interp(np.flatnonzero(mask[:,cell]), np.flatnonzero(~mask[:,cell]), areas[:,cell][~mask[:,cell]])
    
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
