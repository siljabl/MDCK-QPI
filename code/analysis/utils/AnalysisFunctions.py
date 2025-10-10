import numpy as np



def average_cell_radius(df, frame, vox_to_um, blur_factor=1.5):
    frame_mask = (df.frame == frame)

    # total volume of all cells in pixels
    A_cells = np.sum(df[frame_mask].A) / (vox_to_um[-2]*vox_to_um[-1])
    N_cells = np.sum(frame_mask)
    A_cell = A_cells / N_cells

    # typical cell area x blur factor
    r_cell = int(blur_factor*np.sqrt(A_cell/np.pi))

    return r_cell


def compute_flatness(stack, dataframe):
    mean    = np.zeros(len(stack))
    rel_err = np.zeros(len(stack))
    density = np.zeros(len(stack))

    i = 0
    for frame in stack:
        mean[i]    = np.mean(frame[frame > 0])
        rel_err[i] = np.std(frame[frame > 0]) / mean[i]

        N_cells = np.sum(dataframe.frame == i)
        A_cells = np.sum(dataframe[dataframe.frame == i].A)
        density[i] = N_cells / A_cells

        i += 1

    return density, mean, rel_err



def compute_flatness_cellwise(dataframe, fmin, fmax):
    mean    = np.zeros(fmax-fmin)
    rel_err = np.zeros(fmax-fmin)
    density = np.zeros(fmax-fmin)

    N_avrg = 0

    i = 0
    for frame in range(fmax-fmin):
        mask = dataframe.frame == frame
        mean[i]    = np.mean(dataframe[mask].h_avrg)
        rel_err[i] = np.std(dataframe[mask].h_avrg) / mean[i]

        N_cells = np.sum(mask)
        A_cells = np.sum(dataframe[mask].A)
        density[i] = N_cells / A_cells

        N_avrg  += N_cells
        i += 1

    print(N_avrg / i)

    return density, mean, rel_err



def compute_temporal_fluctuations(stack, dataframe, dt, steps, stepsize, empty_val=0):
    density = []
    fluctuations = []

    i = 0
    for t in range(0, steps * stepsize, stepsize):
        # mask of empty areas
        empty   = (np.min(stack[:dt], axis=0) <= empty_val)
        mean_dt = np.mean(stack[t:t+dt], axis=0, dtype=np.float32)
        std_dt  = np.std(stack[t:t+dt],  axis=0, dtype=np.float32)
        tmp_df  = dataframe[(dataframe.frame > t) * (dataframe.frame < t+dt)]

        rel_err = np.copy(std_dt)
        rel_err[empty == 0]  = rel_err[empty == 0] / mean_dt[empty == 0]
        rel_err[empty == 1] = 0

        fluctuations.append(rel_err[empty == 0].ravel())

        N_cells = len(tmp_df)
        A_cells = np.sum(tmp_df.A)
        density.append((N_cells / A_cells) * 1e6 * np.ones_like(fluctuations[-1]))

        i += 1

    return np.concatenate(density), np.concatenate(fluctuations)



def compute_temporal_fluctuations_cellwise_entire_stack(dataframe):
    # save averages of cells
    n_t, n_err = [], []
    h_t, h_err = [], []
    for cell in np.unique(dataframe.particle):
        particle_df = dataframe[dataframe.particle == cell]

        h_t.append(np.mean(particle_df.h_avrg))
        h_err.append(np.std(particle_df.h_avrg))

        try:
            n_t.append(np.mean(particle_df.n_avrg))
            n_err.append(np.std(particle_df.n_avrg))
        except:
            None

    h_rel_err = np.array(h_err) / np.array(h_t)

    try:
        n_rel_err = np.array(n_err) / np.array(n_t)
    except:
        None

    N_cells = len(np.unique(dataframe.particle))
        
    return N_cells, np.array(h_rel_err), np.array(n_rel_err)



def compute_temporal_fluctuations_cellwise(dataframe, dt, steps, stepsize):
    density = []
    hfluctuation = []
    nfluctuation = []

    i = 0
    for t in range(0, steps * stepsize, stepsize):
        tmp_df  = dataframe[(dataframe.frame >= t) * (dataframe.frame < t+dt)]

        # pick out particles in entire frame interval
        cells_ti = tmp_df[(tmp_df.frame==t)].particle
        cells_tf = tmp_df[(tmp_df.frame==t+dt-1)].particle
        cells_mask = np.intersect1d(cells_ti, cells_tf)
        
        # save averages of cells
        n_t, n_err = [], []
        h_t, h_err = [], []
        for cell in cells_mask:
            particle_df = tmp_df[tmp_df.particle == cell]

            h_t.append(np.mean(particle_df.h_avrg))
            h_err.append(np.std(particle_df.h_avrg))

            try:
                n_t.append(np.mean(particle_df.n_avrg))
                n_err.append(np.std(particle_df.n_avrg))
            except:
                None

        h_rel_err = np.array(h_err) / np.array(h_t)
        hfluctuation.append(h_rel_err)

        try:
            n_rel_err = np.array(n_err) / np.array(n_t)
            nfluctuation.append(n_rel_err)
        except:
            None

        N_cells = len(tmp_df)
        A_cells = np.sum(tmp_df.A)
        density.append((N_cells / A_cells) * 1e6 * np.ones_like(hfluctuation[-1]))

        i += 1
        
    return np.concatenate(density), np.concatenate(hfluctuation), np.concatenate(nfluctuation)



def spatial_autocorrelation(im):
    # center input around zero (ensure negative correlation)
    # var = np.mean(im**2) - np.mean(im)**2
    # im  = (im - np.mean(im)) / np.sqrt(var)

    # compute FFT of image
    fft_im  = np.fft.fft2(im, norm="ortho")
    ifft_im = np.fft.ifft2(np.abs(fft_im)**2, norm="ortho")
    ifft_im = np.fft.fftshift(ifft_im)

    # normalize output
    ifft_im_norm = np.real(ifft_im) / np.max(np.real(ifft_im))

    return ifft_im_norm



def radial_distribution(im, vox_to_um, binsize=5):
    # raduis matrix
    Lx = int(len(im)/2)
    Ly = int(len(im[0])/2)
    x = np.arange(-Lx,Lx+1,1) * vox_to_um[-2]
    y = np.arange(-Ly,Ly+1,1) * vox_to_um[-1]
    xx, yy = np.meshgrid(y, x)
    r = np.sqrt(xx**2. + yy**2.)

    nbins = int((r.max() / binsize)+1)
    dist = np.zeros(nbins)
    mean = np.zeros(nbins)
    std  = np.zeros(nbins)

    for i in range(nbins):
        mask = (r >= i*binsize) * (r < (i+1)*binsize)
        dist[i] = np.mean(r[mask])
        mean[i] = np.mean(im[mask])
        std[i]  = np.std(im[mask])

    return dist, mean, std