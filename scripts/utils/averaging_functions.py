import numpy as np
from scipy.ndimage import gaussian_filter



def weighted_average(data, err):
    """ 
    Computes weighted averages over binned datasets (Typically binned by density). 

    PARAMETERS:
    data:       Array of mean values in bins. Data entries without data
    err:        Array of standard deviations in bins

    RETURNS:
    wmean:      Weighted mean of each bin
    wstd:       Weighted uncertainty on the mean
    """

    # Mask entries without data
    mask = data == 0
    data = np.ma.array(data, mask=mask)
    err  = np.ma.array(err,  mask=mask)

    # Compute weights
    weights = np.ones_like(err.data)
    weights[err > 0] = 1 / err.data[err > 0]**2
    weights = np.ma.array(weights, mask=mask)

    # Number of non-zero weights
    N = np.ma.sum(weights!=0)

    # Compute weighted mean and error on the mean
    wmean = np.ma.average(data, weights=weights, axis=0)
    if N > 1:
        wstd  = np.ma.sqrt(np.ma.average((data-wmean)**2, weights=weights, axis=0) / np.sqrt(N))
    else:
        wstd = 0

    return wmean, wstd



def bin_data_by_density(variable, errvariable, density, bin_range, bin_size=100, weighted=True):

    min_bin = bin_range[0]
    max_bin = bin_range[1]
    bins = np.arange(min_bin, max_bin + bin_size+1, bin_size)

    mean_variable = np.zeros(len(bins) - 1)
    std_variable  = np.zeros(len(bins) - 1)
    counts        = np.zeros(len(bins) - 1)

    density_idx = np.digitize(density, bins)

    for i in range(1, len(bins)):
        idx_in_bin = np.where(density_idx == i)[0]
        counts[i-1] = len(idx_in_bin)

        if counts[i-1] > 0:
            if weighted:
                mean_variable[i-1], std_variable[i-1] = weighted_average(variable[idx_in_bin], errvariable[idx_in_bin])#np.average(variable[idx_in_bin], weights=1/errvariable[idx_in_bin])
            else:
                mean_variable[i-1], std_variable[i-1] = np.ma.mean(variable[idx_in_bin]), np.ma.std(variable[idx_in_bin]) / np.ma.sqrt(len(idx_in_bin)) #np.average(variable[idx_in_bin], weights=1/errvariable[idx_in_bin])

        #if counts[i-1] > 1:
        #    std_variable[i-1]  = np.std(variable[idx_in_bin], ddof=1)

    return mean_variable, std_variable, counts



def bin_distribution(data, density, bin_edges):
    """
    Function that takes bins distribution by density.

    Parameters:
    data:       Raw distribution as function of density. On form (density X samples). Any variable of cellprop is on this form
    density:    The corresponding densities
    bin_edges:  The desired bin edges

    Returns:
    binned_distribution: List of distriutions that are sorted as bin_edges
    """

    binned_distribution = []
    Nbins = len(bin_edges) - 1   # Total number of bins

    for i in range(Nbins):
            
        # mask relevant densities
        density_mask = (density >= bin_edges[i]) * (density < bin_edges[i+1])

        # collect data in bin
        data_in_bin  = data[density_mask]
        binned_distribution.append(data_in_bin.ravel().compressed())

    return binned_distribution



# def compute_derivative_of_cdf(arr, sigma):
#     """
#     Function to compute PDF from derivatice of CDF

#     Parameters:
#     arr:   Samples from distribution
#     sigma: Size of smoothing kernel

#     Returns:
#     y_pdf: Probability density function
#     x_pdf:   Corresponding x value
#     """
#     x = np.asarray(arr).ravel()
#     n = x.size

#     x = np.ma.sort(x)
#     x = gaussian_filter(x, sigma=sigma)

#     F = np.ma.cumsum(np.ones(n) / n)
#     F = gaussian_filter(F, sigma=sigma)

#     y_pdf = np.diff(F) / np.diff(x)
#     x_pdf = 0.5 * (x[1:] + x[:-1])

#     return x_pdf, y_pdf




def compute_pdf_from_bins(arr, scale_to_int, sigma=0):
    """
    Computing PDF from fine binning of data.

    Parameters:
    arr: Array of samples from distribution
    scale_to_int: factor to multiply with data before binning as integers
    sigma: size of smoothing kernel

    Returns:
    y_pdf: Probability density function
    x_pdf:   Corresponding x value
    """

    counts = np.bincount((scale_to_int * arr).astype("int"))

    x_pdf  = np.ma.array(np.arange(len(counts)) / scale_to_int, mask=counts==0)

    y_pdf  = counts / counts.sum()
    y_pdf = gaussian_filter(y_pdf, sigma=sigma)

    return x_pdf, y_pdf