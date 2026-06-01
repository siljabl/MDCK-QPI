import numpy as np

def radial_profile(kx_grid, ky_grid, power2d, k_bins):
    """
    Radially average power2d on the (kx_grid, ky_grid) coordinates
    into bins defined by k_bins (edges).
    Returns bin_centers, S_k (average in each radial bin), counts.
    """
    k_mag = np.sqrt(kx_grid**2 + ky_grid**2).ravel()
    p = power2d.ravel()

    inds = np.digitize(k_mag, k_bins) - 1
    nbins = len(k_bins) - 1
    S = np.zeros(nbins, dtype=float)
    counts = np.zeros(nbins, dtype=int)
    for i in range(nbins):
        mask = inds == i
        counts[i] = np.count_nonzero(mask)
        if counts[i] > 0:
            S[i] = p[mask].mean()
        else:
            S[i] = np.nan
    bin_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    return bin_centers, S, counts


def structure_factor_2d(field, dx=1.0, dy=None, window=True, detrend=True,
                        q_bins=None, n_kbins=50):
    """
    Compute radially averaged static structure factor S(k) of a 2D scalar field.

    Inputs:
      - field: 2D array (Ny, Nx) of scalar values (e.g., mass flux per cell).
      - dx: spacing between columns (x direction) in physical units (e.g., µm).
      - dy: spacing between rows (y direction). If None, dy = dx.
      - window: apply 2D Hann window (reduces edge leakage) if True.
      - detrend: subtract mean before FFT if True.
      - k_bins: optional array of radial bin edges (in cycles per length unit).
      - n_kbins: number of radial bins if k_bins is None.

    Returns:
      k_vals: 1D array of radial frequencies (cycles / length unit)
      S_k:    1D array of radially averaged power spectral density (same units as |FFT|^2 normalization)
      counts: number of k modes in each bin
      full_power2d: 2D power spectrum (shifted so k=0 is in center)
      kx_grid, ky_grid: 2D arrays of kx, ky coordinates (same shape as full_power2d)
    """
    if dy is None:
        dy = dx

    f = np.asarray(field, dtype=float)
    Ny, Nx = f.shape

    if detrend:
        f = f - np.nanmean(f)

    # Apply separable Hann window to reduce spectral leakage (optional)
    if window:
        wy = np.hanning(Ny)
        wx = np.hanning(Nx)
        W = np.sqrt(np.outer(wy, wx))  # sqrt so that 2D window energy ~ product
        f = f * W

    # 2D FFT (use fftpack or np.fft) and power spectral density
    Fk = np.fft.fft2(f)
    # Shift zero-frequency to center for convenience
    Fk_shift = np.fft.fftshift(Fk)

    # Normalization: choose convention S = |Fk|^2 / (Nx*Ny)
    # This yields Parseval-consistent power (sum of power ~ sum(f^2))
    norm = (Nx * Ny)
    power2d = (np.abs(Fk_shift)**2) / norm

    # Build kx, ky arrays in units of cycles per length (not radians)
    kx = np.fft.fftfreq(Nx, d=dx)   # array length Nx, cycles per length
    ky = np.fft.fftfreq(Ny, d=dy)
    kx_shift = np.fft.fftshift(kx)
    ky_shift = np.fft.fftshift(ky)
    qx_grid, qy_grid = np.meshgrid(2*np.pi*kx_shift, 2*np.pi*ky_shift)

    # Radial binning
    q_max = 2*np.pi*np.max(kx_shift)
    if q_bins is None:
        q_bins = np.linspace(0.0, q_max, n_kbins+1)

    q_vals, S_k, counts = radial_profile(qx_grid, qy_grid, power2d, q_bins)

    return q_vals, S_k, power2d