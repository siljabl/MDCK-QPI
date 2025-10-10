'''
Functions that returns figures
'''

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from matplotlib.colors import LogNorm
from matplotlib_scalebar.scalebar import ScaleBar

from utils.segment3D import determine_threshold


# normalizing
def normalize(mean, std):
        '''
        Normalize mean and std
        '''
        new_mean = (np.array(mean) - np.min(mean)) / (np.max(mean) - np.min(mean))
        new_std  = std / (np.max(mean) - np.min(mean))

        return new_mean, new_std



def plot_z_profile(n_z, p_z, stack, p, z_0):
    '''
    Plot to illustrate estimation of zero-level. Plotting refrative index and MlM-probability along z-axis, as well as their derivatives.
    n_z and p_z are two dimension arrays, with [0] being n and p along z, and [1] being their derivatives.
    '''
    mean_n = np.mean(n_z[0], axis=0)
    mean_p = np.mean(p_z[0],  axis=0)

    mean_dn = np.mean(n_z[1], axis=0)
    mean_dp = np.mean(p_z[1],  axis=0)

    std_n = np.std(n_z[0], axis=0)
    std_p = np.std(p_z[0],  axis=0)

    std_dn = np.std(n_z[1], axis=0)
    std_dp = np.std(p_z[1],  axis=0)

    mean_n, std_n = normalize(mean_n,   std_n)
    mean_p, std_p = normalize(mean_p, std_p)

    mean_dn, std_dn = normalize(mean_dn,   std_dn)
    mean_dp, std_dp = normalize(mean_dp, std_dp)

    xz_mean = np.mean(stack, axis=2)
    yz_mean = np.mean(stack, axis=1)

    pxz_mean = np.mean(p, axis=2)
    pyz_mean = np.mean(p, axis=1)

    mosaic = [['mean', 'dmean', 'xz'], 
              ['mean', 'dmean', 'pxz'], 
              ['mean', 'dmean', 'yz'],
              ['mean', 'dmean', 'pyz']]
    fig, ax = plt.subplot_mosaic(mosaic)#, sharey=True)

    z0 = np.arange(len(mean_n))
    z1 = np.arange(len(mean_dn))

    ax['mean'].errorbar(mean_n, z0, xerr=std_n, color="c", label="n")
    ax['mean'].errorbar(mean_p, z0, xerr=std_p, color="k", label=r"$p_{MlM}$")
    ax['mean'].hlines(z_0, 0, 1, 'r',ls="dashed", label=r"$z_0$")

    ax['dmean'].errorbar(mean_dn, z1, xerr=std_dn, color="c", label="n")
    ax['dmean'].errorbar(mean_dp, z1, xerr=std_dp, color="k", label=r"$p_{MlM}$")
    ax['dmean'].hlines(z_0, 0, 1, 'r', ls="dashed", label=r"$z_0$")

    ax['mean'].set(ylabel="z [voxel]", title="Mean along z")
    ax['dmean'].set(title="Mean derivative along z")
    ax['mean'].legend()

    ax['xz'].imshow(xz_mean, origin="lower", aspect="auto")
    ax['yz'].imshow(yz_mean, origin="lower", aspect="auto")
    ax['xz'].hlines(z_0, 0, len(xz_mean[0])-1, 'r', lw=1, ls="dashed")
    ax['yz'].hlines(z_0, 0, len(yz_mean[0])-1, 'r', lw=1, ls="dashed")

    ax['xz'].set(title=r"$\langle n(x,y,z) \rangle_y$")
    ax['yz'].set(title=r"$\langle n(x,y,z) \rangle_x$")
    
    ax['pxz'].imshow(pxz_mean, origin="lower", aspect="auto")
    ax['pyz'].imshow(pyz_mean, origin="lower", aspect="auto")
    ax['pxz'].hlines(z_0, 0, len(pxz_mean[0])-1, 'r', lw=1, ls="dashed")
    ax['pyz'].hlines(z_0, 0, len(pyz_mean[0])-1, 'r', lw=1, ls="dashed")

    ax['pxz'].set(title=r"$\langle p_{MlM}(x,y,z) \rangle_y$")
    ax['pyz'].set(title=r"$\langle p_{MlM}(x,y,z) \rangle_x$")

    ax['xz'].set_axis_off()
    ax['yz'].set_axis_off()
    ax['pxz'].set_axis_off()
    ax['pyz'].set_axis_off()
    

    fig.tight_layout()
    return fig


def plot_threshold(thresholds, sums, dims, z_0):
    '''
    Plot to illustrate determination of threshold. Plotting sum of voxels classified as cells and its derivative as function of threshold on MlM-probability.
    '''
    n_above = dims[1] * dims[2] * (dims[0] - z_0)
    n_total = dims[1] * dims[2] * dims[0]

    frac_above = sums[0] / n_above
    frac_total = (sums[0] + sums[1]) / n_total

    centered_thresholds = (thresholds[1:] + thresholds[:-1]) / 2

    fig, ax = plt.subplots(1,2, figsize=(4,3.5), sharex=True)
    sns.set_theme(style='ticks', palette='deep', font_scale=1.1)

    ax[0].plot(thresholds, frac_above, '-', label=r"$z>z_0$")
    ax[0].plot(thresholds, frac_total, '-', label=r"all z")

    ax[1].plot(centered_thresholds, abs(np.diff(frac_above)), '.-')
    ax[1].plot(centered_thresholds, abs(np.diff(frac_total)), '.-')

    ymin = np.min(abs(np.concatenate([np.diff(frac_above), np.diff(frac_total)])))
    ymax = np.max(abs(np.concatenate([np.diff(frac_above), np.diff(frac_total)])))

    threshold_above = np.round(determine_threshold(thresholds, frac_above), 3)
    threshold_total = np.round(determine_threshold(thresholds, frac_total), 3)

    ax[0].vlines(threshold_above, ymin, ymax, color="tab:blue",   ls="dashed")
    ax[0].vlines(threshold_total, ymin, ymax, color="tab:green",  ls="dashed")
    ax[1].vlines(threshold_above, ymin, ymax, color="tab:blue",   ls="dashed", label=threshold_above)
    ax[1].vlines(threshold_total, ymin, ymax, color="tab:green",  ls="dashed", label=threshold_total)

    ax[0].set(xlabel=r"$p_{c}$", title=r"$f_{cell}(p_{c})$")
    ax[1].set(xlabel=r"$p_{c}$", title=r"$df_{cell}/dp_{c}$")
    ax[0].legend()
    ax[1].legend()

    sns.despine()
    fig.tight_layout()
    return fig


def compare_raw_to_segmentation(stack, n_z, vox_to_um):
    '''
    Plotting mean of raw Tomocube data along z and segmented n_z for visual comparison
    '''
    #n_z = np.array(n_z, dtype=np.float16)
    n_z[n_z == 0] = np.NaN

    scalebar_raw  = ScaleBar(vox_to_um[2], 'um', box_alpha=0, color="w")
    scalebar_cell = ScaleBar(vox_to_um[2], 'um', box_alpha=0, color="w")
    scalebar_raw.location  = 'lower left'
    scalebar_cell.location = 'lower left'

    fig, ax = plt.subplots(1,2, figsize=(7,3))
    fig.suptitle("Refractive index")

    im_raw = ax[0].imshow(np.mean(stack, axis=0))
    ax[1].imshow(np.zeros_like(n_z), vmin=0)
    im = ax[1].imshow(n_z)

    ax[0].set(title="Raw data")
    ax[1].set(title="Segmented cell")

    ax[0].add_artist(scalebar_raw)
    ax[1].add_artist(scalebar_cell)

    ax[0].set_axis_off()
    ax[1].set_axis_off()

    fig.colorbar(im_raw, ax=ax[0]);
    fig.colorbar(im, ax=ax[1]);

    fig.tight_layout()
    return fig


def plot_distribution(h_counts, n_counts, im_heights, im_ndx, vox_to_um):
    '''
    Plot mean of height and refractive index histograms.
    '''
    im_ndx = np.array(im_ndx, dtype=np.float16) / 10_000
    im_ndx[im_ndx==0]=np.NaN

    # prepare plotting arrays
    h_bins    = max(map(len, h_counts)) + 0
    h_counts  = np.array([np.pad(arr, (0, h_bins-len(arr)), 'constant') for arr in h_counts])

    # compute average
    mean_h_counts = np.mean(h_counts, axis=0)
    std_h_counts  = np.std(h_counts,  axis=0)
    heights       = (np.arange(h_bins) + 1) * vox_to_um[0]

    mean_n_counts = np.mean(n_counts, axis=0)
    std_n_counts  = np.std(n_counts,  axis=0)
    r_idx         = np.linspace(1.33, 1.38, len(n_counts[0]), endpoint=True)

    # dimensions of inset
    scalebar_h = ScaleBar(vox_to_um[2], 'um', box_alpha=0, color="w")
    scalebar_n = ScaleBar(vox_to_um[2], 'um', box_alpha=0, color="w")
    scalebar_h.location = 'lower left'
    scalebar_n.location = 'lower left'

    fig, ax = plt.subplots(2,1 ,figsize=(6,6))

    ax[0].errorbar(heights[1:], mean_h_counts[1:], yerr=std_h_counts[1:], fmt="o", ms=5, lw=1, color="k", capsize=3, capthick=1)
    ax[0].set(xlabel="Cell height [µm]", ylabel="Density")
    ax[0].set(yscale="linear")

    ax[1].errorbar(r_idx, mean_n_counts, yerr=std_n_counts, fmt="o", ms=5, lw=1, color="k", capsize=3, capthick=1)
    ax[1].set(xlabel="Refractive index", ylabel="Density")
    ax[1].set(yscale="linear")

    fig.tight_layout()
    
    left, bottom, width, height = [0.66, 0.73, 0.25, 0.23]
    imh = fig.add_axes([left, bottom, width, height])
    img=imh.imshow(im_heights)
    imh.add_artist(scalebar_h)
    imh.set_axis_off()
    fig.colorbar(img, ax=imh)

    left, bottom, width, height = [0.66, 0.23, 0.25, 0.23]
    imn = fig.add_axes([left, bottom, width, height])
    img=imn.imshow(np.zeros_like(im_ndx))
    img=imn.imshow(im_ndx)
    imn.add_artist(scalebar_n)
    imn.set_axis_off()
    fig.colorbar(img, ax=imn)

    return fig


def plot_fluctuations_in_time(arr, var, outlier=0):
    '''
    Plot histograms of time average and relative error of variable
    '''
    # compute time average and uncertainty
    mean = np.mean(arr, axis=0, dtype=np.float32)
    std  = np.std(arr,  axis=0, dtype=np.float32)
    tot = np.sum(mean)

    # compute relative uncertainty
    rel_err = np.copy(std)
    rel_err[mean > 0]  = rel_err[mean > 0] / mean[mean > 0]
    rel_err[mean <= 0] = 0
    mask = (mean > outlier)
    rel_mean = np.mean(rel_err[mask])
    rel_std  = np.std(rel_err[mask])

    # prepare arrays for histogram
    mean_arr = mean[mask].ravel()
    std_arr  = std[mask].ravel()
    rel_arr  = rel_err[mask].ravel()

    # plotting
    fig, ax = plt.subplots(1, 2, figsize=(6,3))
    fig.suptitle(f"$\hat{{\sigma}}_{{{var}}}$ - relative error = ({rel_mean*100:0.1f} ± {rel_std*100:0.1f})%")

    ax[0].hist(mean_arr, bins=40, density=True)
    ax[1].hist(rel_arr,  bins=40, density=True)
    ax[0].set(yscale="log", ylabel="density", title=f"${{{var}}}(s)$  [µm]")
    ax[1].set(yscale="log", ylabel="density", title=f"$\hat{{\sigma}}_{{{var}}}(s)$")

    fig.tight_layout()

    return fig, rel_err


def correlations(n_arr, h_arr, vox_to_um):
    '''
    Plot 2D histogram of height and mean refractive index
    '''
    mask = (h_arr > 0)
    h_arr = h_arr[mask].ravel() / vox_to_um
    n_arr = n_arr[mask].ravel()

    r_spearman, p_spearman = stats.spearmanr(h_arr, n_arr)
    r_pearson,  p_pearson  = stats.pearsonr(h_arr,  n_arr)

    bins = int(np.max(h_arr))

    fig, ax = plt.subplots(1,1)
    h=ax.hist2d(n_arr, h_arr * vox_to_um, bins=[bins,bins], density=True, norm=LogNorm())
    ax.set(xlabel=r"$\alpha n_z(x,y,t)$", 
           ylabel=r"$h(x,y,t)$  [µm]",
           title=f"Pearson: {r_pearson:0.2f}, Spearman: {r_spearman:0.2f}")
    fig.colorbar(h[3], ax=ax)
    
    return fig


def plot_cell_dataframe(cells_df):
    '''
    Plot number of cells, area, volume and mass from single cell segmentation (cell tracking)
    '''
    n_frames = cells_df['frame'].max()
    frames  = np.arange(cells_df['frame'].min(), n_frames+1)

    fig, ax = plt.subplots(2,2)

    ax[0,0].plot(frames, cells_df.groupby('frame').size(), '.')
    ax[0,1].hist(cells_df['A'].values, bins=32, density=True)
    ax[1,0].hist(cells_df['V'].values, bins=32, density=True)
    try:
        ax[1,1].hist(cells_df['m'].values, bins=32, density=True)
    except:
        None

    ax[0,0].set(xlabel="frame", ylabel="# cells")
    ax[0,1].set(xlabel="area [µm²]")
    ax[1,0].set(xlabel="volume [µm³]")
    ax[1,1].set(xlabel="mass [a.u.]")

    fig.tight_layout()
    return fig


def compare_voronoi_watershed_areas(h_im, watershed_areas, voronoi_areas, vox_to_um):
    overlap = []
    w_area = []
    v_area = []

    # Tomocube data
    if len(vox_to_um) == 3:
        vox_area = vox_to_um[1]*vox_to_um[2]
    # Holomonitor data
    elif len(vox_to_um) == 2:
        vox_area = vox_to_um[0]*vox_to_um[1]

    for f in range(len(watershed_areas)):
        for l in range(1,watershed_areas[f].max()):
            w_area.append(np.sum(watershed_areas[f] == l) * vox_area)
            v_area.append(np.sum(voronoi_areas[f] == l)   * vox_area)
            intersection = np.sum((watershed_areas[f]==l)*(voronoi_areas[f]==l))

            if intersection == 0:
                overlap.append(0)

            else:
                overlap.append(2*intersection / (w_area[-1] + v_area[-1]))

    line = np.arange(np.max([w_area, v_area]))
    fig, ax = plt.subplots(1,3, figsize=(12,4))

    h0 = ax[0].hist2d(w_area, v_area, bins=[32,32])
    ax[0].plot(line, line, 'r--')
    ax[1].imshow(((watershed_areas[0] > 0)*h_im[0]).T, origin="lower", vmin=1.33)
    ax[2].imshow(((voronoi_areas[0] > 0)*h_im[0]).T, origin="lower", vmin=1.33)

    ax[0].set(xlabel="watershed [µm²]", ylabel="voronoi [µm²]")
    ax[1].set(title="Watershed")
    ax[2].set(title="Voronoi")
    fig.colorbar(h0[3], ax=ax[0])

    return fig


def vector_distance(U_w, V_w, U_v, V_v):
    U_w = np.array(U_w)
    V_w = np.array(V_w)
    U_v = np.array(U_v)
    V_v = np.array(V_v)

    return np.sqrt((U_w-U_v)**2 + (V_w-V_v)**2)


def compare_voronoi_watershed_polarisation(n_im, watershed_df, voronoi_df, frame):
    idx_v  = (voronoi_df['frame']==frame)
    idx_w  = (watershed_df['frame']==frame)
    common_labels = np.intersect1d(watershed_df[idx_w]['label'], voronoi_df[idx_v]['label'])

    df_w = watershed_df[idx_w * watershed_df['label'].isin(common_labels)]
    df_v = voronoi_df[idx_v * voronoi_df['label'].isin(common_labels)]

    X = df_w['x']
    Y = df_w['y']
    U_w = [r * np.cos(phi) for r, phi in zip(df_w['magnitude'], df_w['angle'])]
    V_w = [r * np.sin(phi) for r, phi in zip(df_w['magnitude'], df_w['angle'])]
    U_v = [r * np.cos(phi) for r, phi in zip(df_v['magnitude'], df_v['angle'])]
    V_v = [r * np.sin(phi) for r, phi in zip(df_v['magnitude'], df_v['angle'])]


    dist = vector_distance(U_w, V_w, U_v, V_v)
    r_line   = np.linspace(0, 1, 10)
    phi_line = np.linspace(-np.pi/2, np.pi/2, 10)

    mosaic = [['imw', 'imw', 'imv', 'imv', 'r', 'phi'], 
              ['imw', 'imw', 'imv', 'imv', 'hist', 'hist']]

    fig, ax = plt.subplot_mosaic(mosaic, figsize=(12,4))

    ax['imw'].imshow(n_im[frame].T, origin="lower", vmin=1.33)
    ax['imw'].quiver(X, Y, U_w, V_w, pivot="mid", headlength=0, headaxislength=0, width=0.007)

    ax['imv'].imshow(n_im[frame].T, origin="lower", vmin=1.33)
    ax['imv'].quiver(X, Y, U_v, V_v, pivot="mid", headlength=0, headaxislength=0, width=0.007)

    ax['imw'].set(title="Watershed")
    ax['imv'].set(title="Voronoi")

    ax['r'].hist2d(df_w['magnitude'], df_v['magnitude']);
    ax['phi'].hist2d(df_w['angle'], df_v['angle']);
    ax['r'].plot(r_line, r_line, 'r--')
    ax['phi'].plot(phi_line, phi_line, 'r--')
    ax['hist'].hist(dist, density='True')

    ax['r'].set(title="magnitude", xlabel="watershed", ylabel="voronoi")
    ax['phi'].set(title="angle", xlabel="watershed", ylabel="voronoi")
    ax['hist'].set(xlabel="vector distance")
    fig.tight_layout()

    return fig