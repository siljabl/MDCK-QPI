import sys
import numpy as np
import imageio.v2 as imageio
import matplotlib as mpl
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.signal.windows import hann, tukey
from scipy.ndimage import gaussian_filter

# ============================================================
# CONFIG
# ============================================================

DX = 0.155433 * 64     # µm per pixel (your voxel_to_um(2) * 64)
DY = 0.155433 * 64     # µm per pixel
DT = 1.0 / 4.0         # hours per frame
N_steps = 100

GAUSS_FILTER_95_WIDTH_UM      = 5.5 * DX
GAUSS_FILTER_95_WIDTH_FRAMES  = 1.6


SYMLIN_LINTHRESH_FACTOR = 1e-3   # factor for symlog linthresh


RADIAL_BINS = None     # None => min(nx,ny)//2
DETREND = True         # match MATLAB if detrend=true
WINDOW_XY = "hann"     # "hann" | "tukey" | "none"
WINDOW_T  = "hann"     # "hann" | "tukey" | "none"
TUKEY_ALPHA = 0.25
NORMALIZE = "density"  # "density" | "amplitude" | "none"

CMAP = "RdBu_r"
VMIN, VMAX = -4, 4
FIGSIZE  = (8, 4)
FONTSIZE = 24

# parameter sweeps
SPATIAL_WIDTHS_UM   = np.linspace(0, 100, N_steps) # µm
TEMP_WIDTHS_FRAMES  = np.linspace(0, 4,   N_steps) # frames

# ============================================================
# UTILS
# ============================================================

def load_tiff_stack(folder_path: str) -> np.ndarray:
    """Load all .tif/.tiff images in a folder into a stack (nt, ny, nx)."""
    folder = Path(folder_path)
    files = sorted(folder.glob("*.tif")) + sorted(folder.glob("*.tiff"))
    if not files:
        raise FileNotFoundError(f"No TIFF files found in {folder}")

    first = imageio.imread(files[0]).astype(float)
    ny, nx = first.shape[:2]
    nt = len(files)

    stack = np.zeros((nt, ny, nx), dtype=float)
    stack[0] = first
    for i, f in enumerate(files[1:], start=1):
        stack[i] = imageio.imread(f).astype(float)

    return stack


def symlog(x: np.ndarray, linthresh: float | None = None) -> np.ndarray:
    """Symmetric log-like transform for signed data."""
    x = np.asarray(x, dtype=float)
    if linthresh is None:
        linthresh = SYMLIN_LINTHRESH_FACTOR * (np.max(np.abs(x)) + np.finfo(float).eps)

    z = np.zeros_like(x)
    mask_lin = np.abs(x) <= linthresh
    z[mask_lin] = x[mask_lin] / linthresh

    mask_log = ~mask_lin
    z[mask_log] = np.sign(x[mask_log]) * (
        1.0 + np.log10(np.abs(x[mask_log]) / linthresh)
    )
    return z


def _make_1d_window(kind: str, N: int, alpha: float) -> np.ndarray:
    kind = kind.lower()
    if kind == "hann":
        return hann(N, sym=True)
    if kind == "tukey":
        return tukey(N, alpha=alpha, sym=True)
    if kind == "none":
        return np.ones(N)
    raise ValueError(f"Unknown window type: {kind}")


def smoothaverage_python(S3: np.ndarray,
                         om: np.ndarray,
                         targetOmega: float,
                         wsmooth: int):
    """
    Python version of MATLAB smoothaverage for S3(qx,qy,omega).

    Parameters
    ----------
    S3 : array, shape (ny, nx, n_omega)
        3D structure factor or similar quantity.
    om : 1D array, length n_omega
        Frequency values corresponding to the third axis of S3.
    targetOmega : float
        Target frequency at which to take the 2D slice.
    wsmooth : int
        Window length for 1D moving-average smoothing.

    Returns
    -------
    Sout : 1D array
        Smoothed radial profile (averaged over up/down/left/right).
    cx, cy : int
        Center indices in x and y.
    l : int
        Number of points in each radial direction from the center.
    """
    # closest omega index
    k = int(np.argmin(np.abs(om - targetOmega)))
    S = S3[:, :, k]           # 2D slice (ny, nx)
    ny, nx = S.shape

    lx = nx // 2
    rx = nx % 2               # 0 or 1
    ly = ny // 2
    ry = ny % 2               # 0 or 1
    cx = lx + rx
    cy = ly + ry
    l  = min(lx, ly)

    il = np.arange(cx, cx - l, -1)        # left indices (x)
    ir = np.arange(cx + 1, cx + l + 1)    # right indices (x)
    iu = np.arange(cy, cy - l, -1)        # up indices (y)
    idd = np.arange(cy + 1, cy + l + 1)   # down indices (y)

    # Extract four 1D cuts through the center:
    up    = S[iu, cx]
    down  = S[idd, cx]
    left  = S[cy, il]
    right = S[cy, ir]

    radial_profile = (up + down + left + right) / 4.0

    # simple 1D moving-average smoothing over window wsmooth
    if wsmooth > 1:
        wsmooth = int(wsmooth)
        kernel = np.ones(wsmooth, dtype=float) / wsmooth
        Sout = np.convolve(radial_profile, kernel, mode="same")
    else:
        Sout = radial_profile.copy()

    return Sout, cx, cy, l


# ============================================================
# CORE: SPATIO-TEMPORAL AUTOCORRELATION
# ============================================================

def spatiotemporal_autocorrelation(
    stack: np.ndarray,
    dx: float,
    dy: float,
    dt: float,
    radial_bins: int | None = None,
    detrend: bool = True,
    windowXY: str = "hann",
    windowT: str = "hann",
    tukeyAlpha: float = 0.25,
    normalize: str = "density",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute radially averaged spatio-temporal autocorrelation C(r, t_lag)
    from a stack with shape (nt, ny, nx).

    Returns
    -------
    r_bins : 1D array, radii in µm
    t_lags : 1D array, time lags in hours
    C_rt   : 2D array, shape (N_r, N_t), autocorrelation vs r and t_lag
    """
    nt, ny, nx = stack.shape
    if radial_bins is None:
        radial_bins = min(nx, ny) // 2

    data = stack.astype(float)

    # ------- detrend -------
    if detrend:
        mu_t = np.nanmean(data, axis=(1, 2), keepdims=True)
        data = data - mu_t

        # mean over t per pixel
        # mu_xy = np.nanmean(data, axis=(0), keepdims=True)
        # data = data - mu_xy

    # ------- windows -------
    wx = _make_1d_window(windowXY, nx, tukeyAlpha)
    wy = _make_1d_window(windowXY, ny, tukeyAlpha)
    wt = _make_1d_window(windowT, nt, tukeyAlpha)

    Wxy = wy[None, :, None] * wx[None, None, :]
    W = wt[:, None, None] * Wxy
    data_w = data * W

    # ------- 3D FFT and power spectrum -------
    F = np.fft.fftn(data_w, axes=(0, 1, 2))
    Ntot = nt * ny * nx

    normalize = normalize.lower()
    if normalize == "density":
        P = np.abs(F) ** 2 / Ntot
    elif normalize == "amplitude":
        P = np.abs(F / np.sqrt(Ntot)) ** 2
    elif normalize == "none":
        P = np.abs(F) ** 2
    else:
        raise ValueError(f"Unknown normalize setting: {normalize}")

    # ------- autocorrelation via inverse FFT -------
    C = np.fft.ifftn(P, axes=(0, 1, 2)).real
    C_spatial_shifted = np.fft.fftshift(C, axes=(1, 2))  # shift y,x only

    # positive time lags
    nt_half = nt // 2
    C_pos = C_spatial_shifted[:nt_half + 1, :, :]  # (nt_half+1, ny, nx)

    # ------- radial averaging in (y,x) -------
    y_pix = np.arange(ny) - ny // 2
    x_pix = np.arange(nx) - nx // 2
    X_pix, Y_pix = np.meshgrid(x_pix, y_pix)
    X = X_pix * dx
    Y = Y_pix * dy
    R = np.sqrt(X ** 2 + Y ** 2)  # µm

    # use radius up to half-width (here we used full extent)
    r_max = np.max(X)
    r_edges = np.linspace(0.0, r_max, radial_bins + 1)
    r_bins = 0.5 * (r_edges[:-1] + r_edges[1:])

    R_flat = R.ravel()
    C_flat = C_pos.reshape(nt_half + 1, -1)

    bin_idx = np.digitize(R_flat, r_edges) - 1
    bin_idx[bin_idx < 0] = 0
    bin_idx[bin_idx >= radial_bins] = radial_bins - 1

    C_rt = np.zeros((radial_bins, nt_half + 1), dtype=float)
    for b in range(radial_bins):
        mask = (bin_idx == b)
        if np.any(mask):
            C_rt[b, :] = C_flat[:, mask].mean(axis=1)

    t_lags = np.arange(nt_half + 1) * dt

    # normalize
    C_rt = C_rt / C_rt[0,0]
    return r_bins, t_lags, C_rt



def spatiotemporal_crosscorrelation(
    stack1: np.ndarray,
    stack2: np.ndarray,
    dx: float,
    dy: float,
    dt: float,
    radial_bins: int | None = None,
    detrend: bool = True,
    windowXY: str = "hann",
    windowT: str = "hann",
    tukeyAlpha: float = 0.25,
    normalize: str = "density"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute radially averaged spatio-temporal cross-correlation
    C(r, t_lag) between two stacks with shape (nt, ny, nx).

    Returns
    -------
    r_bins : 1D array, radii in µm
    t_lags : 1D array, time lags in hours
    C_rt   : 2D array, shape (N_r, N_t), cross-correlation vs r and t_lag
    """
    if stack1.shape != stack2.shape:
        raise ValueError("stack1 and stack2 must have the same shape (nt, ny, nx).")

    nt, ny, nx = stack1.shape
    if radial_bins is None:
        radial_bins = min(nx, ny) // 2

    data1 = stack1.astype(float)
    data2 = stack2.astype(float)

    # ------- detrend -------
    if detrend:
        mu1_t = np.nanmean(data1, axis=(1, 2), keepdims=True)
        mu2_t = np.nanmean(data2, axis=(1, 2), keepdims=True)
        data1 = data1 - mu1_t
        data2 = data2 - mu2_t

    # ------- windows -------
    wx = _make_1d_window(windowXY, nx, tukeyAlpha)
    wy = _make_1d_window(windowXY, ny, tukeyAlpha)
    wt = _make_1d_window(windowT, nt, tukeyAlpha)

    Wxy = wy[None, :, None] * wx[None, None, :]
    W = wt[:, None, None] * Wxy
    data1_w = data1 * W
    data2_w = data2 * W

    # ------- 3D FFT and cross-spectrum -------
    F1 = np.fft.fftn(data1_w, axes=(0, 1, 2))
    F2 = np.fft.fftn(data2_w, axes=(0, 1, 2))
    Ntot = nt * ny * nx

    normalize = normalize.lower()
    if normalize == "density":
        S = (F1 * np.conjugate(F2)) / Ntot
    elif normalize == "amplitude":
        S = (F1 / np.sqrt(Ntot)) * np.conjugate(F2 / np.sqrt(Ntot))
    elif normalize == "none":
        S = F1 * np.conjugate(F2)
    else:
        raise ValueError(f"Unknown normalize setting: {normalize}")

    # ------- cross-correlation via inverse FFT -------
    C = np.fft.ifftn(S, axes=(0, 1, 2)).real
    C_spatial_shifted = np.fft.fftshift(C, axes=(1, 2))  # shift y,x only

    # positive time lags
    nt_half = nt // 2
    C_pos = C_spatial_shifted[:nt_half + 1, :, :]  # (nt_half+1, ny, nx)

    # ------- radial averaging in (y,x) -------
    y_pix = np.arange(ny) - ny // 2
    x_pix = np.arange(nx) - nx // 2
    X_pix, Y_pix = np.meshgrid(x_pix, y_pix)
    X = X_pix * dx
    Y = Y_pix * dy
    R = np.sqrt(X ** 2 + Y ** 2)  # µm

    r_max = np.max(X)
    r_edges = np.linspace(0.0, r_max, radial_bins + 1)
    r_bins = 0.5 * (r_edges[:-1] + r_edges[1:])

    R_flat = R.ravel()
    C_flat = C_pos.reshape(nt_half + 1, -1)

    bin_idx = np.digitize(R_flat, r_edges) - 1
    bin_idx[bin_idx < 0] = 0
    bin_idx[bin_idx >= radial_bins] = radial_bins - 1

    C_rt = np.zeros((radial_bins, nt_half + 1), dtype=float)
    for b in range(radial_bins):
        mask = (bin_idx == b)
        if np.any(mask):
            C_rt[b, :] = C_flat[:, mask].mean(axis=1)

    t_lags = np.arange(nt_half + 1) * dt
    return r_bins, t_lags, C_rt



# ============================================================
# PIPELINE: RAW + GAUSSIAN-FILTERED (FIXED PARAM)
# ============================================================

def compute_correlations(stack: np.ndarray,
                         w_space=GAUSS_FILTER_95_WIDTH_UM,
                         w_time=GAUSS_FILTER_95_WIDTH_FRAMES) -> tuple:
    """Return (r_raw, t_raw, C_raw, r_filt, t_filt, C_filt).

    - raw  : correlations from the original stack
    - filt : correlations from the stack smoothed with the same
             spatial/temporal Gaussian used elsewhere (smooth_stack),
             here using GAUSS_FILTER_95_WIDTH_UM in space and no
             temporal smoothing.
    """
    # raw
    r_raw, t_raw, C_raw = spatiotemporal_autocorrelation(
        stack,
        dx=DX, dy=DY, dt=DT,
        radial_bins=RADIAL_BINS,
        detrend=DETREND,
        windowXY=WINDOW_XY,
        windowT=WINDOW_T,
        tukeyAlpha=TUKEY_ALPHA,
        normalize=NORMALIZE
    )

    # use the same smoothing function as in the RMS analysis
    if GAUSS_FILTER_95_WIDTH_UM > 0:
        # spatial smoothing only, no temporal smoothing
        stack_filt = smooth_stack(
            stack,
            spatial_width_um=w_space,
            temporal_width_frames=w_time,
        )
    else:
        stack_filt = stack.copy()

    r_filt, t_filt, C_filt = spatiotemporal_autocorrelation(
        stack_filt,
        dx=DX, dy=DY, dt=DT,
        radial_bins=RADIAL_BINS,
        detrend=DETREND,
        windowXY=WINDOW_XY,
        windowT=WINDOW_T,
        tukeyAlpha=TUKEY_ALPHA,
        normalize=NORMALIZE,
    )

    return r_raw, t_raw, C_raw, r_filt, t_filt, C_filt



def compute_crosscorrelations(stack1: np.ndarray, stack2: np.ndarray) -> tuple:
    """Return (r_raw, t_raw, C_raw, r_filt, t_filt, C_filt).

    - raw  : correlations from the original stack
    - filt : correlations from the stack smoothed with the same
             spatial/temporal Gaussian used elsewhere (smooth_stack),
             here using GAUSS_FILTER_95_WIDTH_UM in space and no
             temporal smoothing.
    """
    # raw
    r_raw, t_raw, C_raw = spatiotemporal_crosscorrelation(
        stack1,
        stack2,
        dx=DX, dy=DY, dt=DT,
        radial_bins=RADIAL_BINS,
        detrend=DETREND,
        windowXY=WINDOW_XY,
        windowT=WINDOW_T,
        tukeyAlpha=TUKEY_ALPHA,
        normalize=NORMALIZE,
    )

    return r_raw, t_raw, C_raw


# ============================================================
# SMOOTHING + RMS BETWEEN FLUX AND CHANGE
# ============================================================

def smooth_stack(stack: np.ndarray,
                 spatial_width_um: float,
                 temporal_width_frames: float) -> np.ndarray:
    """
    Apply separable Gaussian smoothing in time and space:

    - spatial_width_um: 
    - temporal_width_frames: 
      Use 0.0 to disable smoothing in that dimension.
    """

    sigma_spatial_phys = spatial_width_um
    sigma_spatial_pix  = sigma_spatial_phys / DX
    sigma_t = temporal_width_frames

    sigma = (sigma_t, sigma_spatial_pix, sigma_spatial_pix)
    if sigma_t == 0.0 and sigma_spatial_pix == 0.0:
        return stack.copy()
    return gaussian_filter(stack, sigma=sigma)


def compute_rms_difference(stack1: np.ndarray,
                           stack2: np.ndarray,
                           spatial_width_um: float,
                           temporal_width_frames: float) -> float:
    """
    Normalized RMS of correlations of smoothed flux and smoothed change:

        NRMS = sqrt(<(C_J - C_dm)^2>) / sqrt(<C_dm^2+C_J^2>),

    where J is smoothed flux, dm is smoothed mass change,
    and <·> denotes average over space and time.
    """

    _, _, _, _, _, C1_filt = compute_correlations(stack1, w_space=spatial_width_um, w_time=temporal_width_frames)
    _, _, _, _, _, C2_filt = compute_correlations(stack2, w_space=spatial_width_um, w_time=temporal_width_frames)

    diff = C1_filt - C2_filt
    num  = np.sqrt(np.mean(diff**2))

    denom = np.sqrt(np.mean(C1_filt**2 + C2_filt**2))
    if denom == 0:
        return np.nan
    return float(num / denom)


def sweep_rms(stack1: np.ndarray, stack2: np.ndarray, w_r, w_t):
    """Compute RMS vs spatial and temporal smoothing parameters."""
    # vary spatial width, no temporal smoothing
    rms_spatial = []
    for w_um in SPATIAL_WIDTHS_UM:
        rms = compute_rms_difference(stack1, stack2,
                                     spatial_width_um=w_um,
                                     temporal_width_frames=w_t)
        rms_spatial.append(rms)


    # vary temporal width, no extra spatial smoothing (beyond DX grid)
    rms_temporal = []
    for w_frames in TEMP_WIDTHS_FRAMES:
        rms = compute_rms_difference(stack1, stack2,
                                     spatial_width_um=w_r,
                                     temporal_width_frames=w_frames)
        rms_temporal.append(rms)


    rms_spatial  = np.array(rms_spatial)
    rms_temporal = np.array(rms_temporal)

    return rms_spatial, rms_temporal


# ============================================================
# PLOTTING
# ============================================================

def plot_two_correlations(
    r_raw: np.ndarray, t_raw: np.ndarray, C_raw: np.ndarray,
    r_filt: np.ndarray, t_filt: np.ndarray, C_filt: np.ndarray,
    title: str
):
    """Plot raw and filtered correlations side-by-side with a shared colorbar."""
    Z_raw  = symlog(C_raw)
    Z_filt = symlog(C_filt)
    # Z_raw  = (C_raw)
    # Z_filt = (C_filt)

    fig = plt.figure(figsize=FIGSIZE)
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.1], wspace=0.15)

    fig.suptitle(title)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    cax = fig.add_subplot(gs[0, 2])

    im1 = ax1.imshow(
        Z_raw,
        extent=(t_raw[0], t_raw[-1], r_raw[0], r_raw[-1]),
        origin="lower",
        aspect="auto",
        cmap=CMAP,
        vmin=VMIN, vmax=VMAX,
    )
    ax1.set_xlabel("t (h)")
    ax1.set_ylabel("r (µm)")
    ax1.set_xticks([0,2,4])


    im2 = ax2.imshow(
        Z_filt,
        extent=(t_filt[0], t_filt[-1], r_filt[0], r_filt[-1]),
        origin="lower",
        aspect="auto",
        cmap=CMAP,
        vmin=VMIN, vmax=VMAX,
    )
    ax2.set_xlabel("t (h)")
    ax2.set_xticks([0,2,4])
    plt.setp(ax2.get_yticklabels(), visible=False)

    fig.colorbar(im1, cax=cax)
    fig.subplots_adjust(left=0.15, right=0.92, bottom=0.2, top=0.87, wspace=0.2)

    return fig


def plot_rms_sweeps(rms_spatial: np.ndarray, rms_temporal: np.ndarray, r_label, t_label):
    """Plot RMS vs spatial and temporal smoothing parameters."""
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    alpha = [0.5, 1, 0.5]

    for rms, label, a in zip(rms_spatial, t_label, alpha):
        axs[0].plot(SPATIAL_WIDTHS_UM, rms, "-", lw=4, ms=10, alpha=a, label=rf"$r_t = {label:0.1f}$ h")
    axs[0].set_xlabel(r"$r_{\mathrm{xy}}$ (µm)")
    axs[0].set_title(r"RMS($C_{\boldsymbol{\nabla}(\mathrm{m}\mathbf{v})} − C_{\partial_t\mathrm{m}})$")
    axs[0].legend()

    for rms, label, a in zip(rms_temporal, r_label, alpha):
        axs[1].plot(TEMP_WIDTHS_FRAMES * DT, rms, "-", lw=4, ms=10, alpha=a, label=rf"$r_{{xy}} = {label:0.0f}$ µm")
    axs[1].set_xlabel(r"$r_{\mathrm{t}}$ (h)")
    # axs[1].set_ylabel("RMS(C − C)")
    axs[1].legend()

    fig.tight_layout()
    return fig


# ============================================================
# MAIN
# ============================================================

font = {'size': FONTSIZE}
mpl.rc('font', **font)

def main():
    # Usage:
    #   python this_script.py /path/to/mass_flux /path/to/mass_change

    flux_dir   = sys.argv[2]
    change_dir = sys.argv[1]

    flux_stack   = load_tiff_stack(flux_dir)
    change_stack = load_tiff_stack(change_dir)

    # 1) correlations with fixed smoothing parameter (original behavior)
    title_flux   = r"$\mathrm{symlog}\,\!\left(C_{\boldsymbol{\nabla}(\mathrm{m}\mathbf{v})}\right)$"
    title_change = r"$\mathrm{symlog}\,\!\left(C_{\partial_t\mathrm{m}}\right)$"

    r1_raw, t1_raw, C1_raw, r1_filt, t1_filt, C1_filt = compute_correlations(flux_stack)
    Path("figs").mkdir(exist_ok=True)
    fig1 = plot_two_correlations(r1_raw, t1_raw, C1_raw, r1_filt, t1_filt, C1_filt, title_flux)
    fig1.savefig("figs/mass_flux_correlation.png", dpi=300)
    plt.close(fig1)

    r2_raw, t2_raw, C2_raw, r2_filt, t2_filt, C2_filt = compute_correlations(change_stack)
    fig2 = plot_two_correlations(r2_raw, t2_raw, C2_raw, r2_filt, t2_filt, C2_filt, title_change)
    fig2.savefig("figs/mass_change_correlation.png", dpi=300)
    plt.close(fig2)

    # 2) RMS sweeps over smoothing parameters
    rlabel = np.array([25,  GAUSS_FILTER_95_WIDTH_UM, 75])
    tlabel = np.array([0.8, GAUSS_FILTER_95_WIDTH_FRAMES, 2.4])
    rms_spatial  = []
    rms_temporal = []
    
    for rl, tl in zip(rlabel, tlabel):
        rms_r, rms_t = sweep_rms(flux_stack, change_stack, rl, tl)

        rms_spatial.append(rms_r)
        rms_temporal.append(rms_t)

    fig3 = plot_rms_sweeps(rms_spatial, rms_temporal, rlabel, tlabel)
    fig3.savefig("figs/rms_vs_smoothing.png", dpi=300)
    plt.close(fig3)



if __name__ == "__main__":
    main()

