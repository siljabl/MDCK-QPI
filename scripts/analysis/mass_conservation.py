import sys
import json
import imageio
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from skimage.measure   import block_reduce
from scipy.interpolate import griddata
from scipy.ndimage     import gaussian_filter

sys.path.append("scripts/utils/")
from file_operations import *
from Microscopes     import Holomonitor, Tomocube
from SegmentedCells  import SegmentedCells


data_path = "../../../../hdd_data/silja/Monolayers/"


def filter_velocity_fields(X, Y, U, V, speed_limit, sigma=0.5):
    """
    Replace outlier velocities with interpolation of surrounding pixels
    
    PARAMETERS:
    X:           x-positions of velocity vectors
    Y:           y-positions of velocity vectors
    U:           x-components of velocity vectors
    V:           y-components of velocity vectors
    speed_limit: limit that defines outlier velocities. Distance from meed speed
    sigma:       std of gaussian blur

    Returns:
    U_out:  new x-components of velocity vectors
    V_out:  new y-components of velocity vectors
    """

    # define mean velocity
    mask_extreme_outliers = (abs(U) < 100) * (abs(V) < 100) # mask that excludes extreme outliers from affecting mean
    mean_velocity = [np.mean(U[mask_extreme_outliers]), np.mean(V[mask_extreme_outliers])] # µm / h

    # mask velocities that differ from mean velocity by more than speed limit
    distance_from_mean = np.ma.sqrt((U-mean_velocity[0])**2 + (V-mean_velocity[1])**2)
    mask = distance_from_mean > speed_limit

    # define positions of velocities
    keep_points    = np.vstack((X[~mask], Y[~mask])).T  # postions of velocities that are below limit
    replace_points = np.vstack((X[mask],  Y[mask])).T   # postions of velocities that are above limit

    # create output arrays
    U_out = np.copy(U)
    V_out = np.copy(V)

    # replace bad velocities by interpolation in grid
    U_out[mask] = griddata(keep_points, U[~mask], replace_points)
    V_out[mask] = griddata(keep_points, V[~mask], replace_points)

    # remove nans
    U_out = np.nan_to_num(U_out)
    V_out = np.nan_to_num(V_out)

    # smoothen velocity fields
    U_out = gaussian_filter(U_out, sigma)
    V_out = gaussian_filter(V_out, sigma)

    return U_out, V_out



def shift_field_to_faces(field, axis=0):
    """
    Shifts field to the pixel faces along the axis specified

    PARAMETERS:
    field:      2 dimensional array (NxM)

    RETURNS:
    face_field: 2 dimensional array with one additional column (axis=0) or row (axis=1)
    """

    dims = np.array(np.shape(field))    # Shape of input
    dims[axis] += 1                     # Shape of output

    face_field = np.zeros([dims[0], dims[1]])

    # Shift field to faces
    # Faces take the mean of its two adjacent pixel centers as values
    # The boundary is taken as the original field boundary
    if axis == 0:
        face_field[1:-1,:] = 0.5 * (field[:-1,:] + field[1:,:])
        face_field[0,:]  = field[0,:]
        face_field[-1,:] = field[-1,:]

    elif axis == 1:
        face_field[:,1:-1] = 0.5 * (field[:,:-1] + field[:,1:])
        face_field[:,0]  = field[:,0]
        face_field[:,-1] = field[:,-1]

    return face_field



def finite_volume_mass_flux(mass_density, U, V, pix_to_um, piv_step_size=64, velocity_scale=1, smoothing=0):
    """
    Downsamples the mass density to same resolution as the velocity fields and computes mass density flux in and out of every pixel

    PARAMETERS:
    mass_density:   Full 2 dimensional array representing a single frame. In µm
    U:              X-velocity field of same frame. In µm/h
    V:              Y-velocity field of same frame. In µm/h
    pix_to_um:      Conversion factor from pixels to µm
    piv_step_size:  Step size of PIV window, the same as the xy-spacing of the velocity field
    velocity_scale: Factor to multiply velocity field  with. Typically so distribution of PIV speeds and tracked speed get same mean

    RETURNS:
    mass_flux:      Field that represents mass flux of every pixel. Same size as original velocity fields
    """
    
    # 1) Downsample mass denstity
    s = int(piv_step_size / 2)
    mass_density = mass_density[s:-s, s:-s] # removing region outside velocity field
    mass_density = block_reduce(mass_density, block_size=piv_step_size, func=np.mean)

    # 2) Compute density on pixel faces
    mass_density_xface = shift_field_to_faces(mass_density, axis=1)
    mass_density_yface = shift_field_to_faces(mass_density, axis=0)

    # 3) Compute velocity on pixel faces
    U_face = shift_field_to_faces(U, axis=1)
    V_face = shift_field_to_faces(V, axis=0)

    # 4) Compute fluxes through faces
    flux_x = mass_density_xface * U_face
    flux_y = mass_density_yface * V_face

    # 5) Smoothend fields
    flux_x = gaussian_filter(flux_x, smoothing)
    flux_y = gaussian_filter(flux_y, smoothing)

    left_flux   = flux_x[:,:-1]
    right_flux  = flux_x[:,1:]
    bottom_flux = flux_y[:-1,:]
    top_flux    = flux_y[1:,:]

    # 6) Compute net flux
    face_lenght = pix_to_um * piv_step_size  # lenght of the downsampled face in µm
    mass_flux   = (left_flux - right_flux) + (bottom_flux - top_flux)
    mass_flux  *= velocity_scale * face_lenght  # µm³/h

    return mass_flux



def finite_volume_mass_change(mass_densities, pix_to_um, dt=0.25, piv_step_size=64, smoothing=0):
    """
    Downsamples the mass density to same resolution as the velocity fields and computes mass change in every pixel

    PARAMETERS:
    mass_densities: Full 2 dimensional array representing a single frame. In µm
    pix_to_um:      Conversion factor from pixels to µm
    dt:             Conversion factor from frames to h
    piv_step_size:  Step size of PIV window, the same as the xy-spacing of the velocity field

    RETURNS:
    mass_change:    Field that represents mass change in every pixel. Same size as original velocity fields
    """

    # 1) Downsample mass denstity
    s = int(piv_step_size / 2)
    mass_densities = mass_densities[:, s:-s, s:-s] # removing region outside velocity field
    mass_densities = [block_reduce(mass_density, block_size=piv_step_size, func=np.mean) for mass_density in mass_densities]
    mass_densities = [gaussian_filter(mass_density, smoothing) for mass_density in mass_densities]

    # 2) Compute mass change
    new_pixel_area = (piv_step_size * pix_to_um)**2  # area of the downsampled pixel in µm²
    mass_change    = (mass_densities[1] - mass_densities[0])
    mass_change   *= new_pixel_area / dt  # µm³/h

    return mass_change




def compute_divergence(U, V, pix_to_um, piv_step_size=64, velocity_scale=1):
    """
    Compute divergence of velocity field.

    PARAMETERS:
    U:              X-velocity field of same frame. In µm/h
    V:              Y-velocity field of same frame. In µm/h
    pix_to_um:      Conversion factor from pixels to µm
    piv_step_size:  Step size of PIV window, the same as the xy-spacing of the velocity field
    velocity_scale: Factor to multiply velocity field  with. Typically so distribution of PIV speeds and tracked speed get same mean

    RETURNS:
    divergence:     Divergence on pixel centers(?)
    """

    # 1) Compute velocity on pixel faces
    U_face = shift_field_to_faces(U, axis=1)
    V_face = shift_field_to_faces(V, axis=0)

    # 2) Compute divergence
    divergence = (U_face[:, 1:] - U_face[:, :-1]) + (V_face[1:, :] - V_face[:-1, :])
    divergence *= velocity_scale / (piv_step_size * pix_to_um)

    return divergence



def compute_curl(U, V, pix_to_um, piv_step_size=64, velocity_scale=1):
    """
    Compute curl of velocity field.

    PARAMETERS:
    U:              X-velocity field of same frame. In µm/h
    V:              Y-velocity field of same frame. In µm/h
    pix_to_um:      Conversion factor from pixels to µm
    piv_step_size:  Step size of PIV window, the same as the xy-spacing of the velocity field
    velocity_scale: Factor to multiply velocity field  with. Typically so distribution of PIV speeds and tracked speed get same mean

    RETURNS:
    curle:          Curl on pixel centers(?)
    """
    
    # 1) Compute velocity on pixel faces
    U_face = shift_field_to_faces(U, axis=1)
    V_face = shift_field_to_faces(V, axis=0)

    # Compute curl
    dvdx = (V_face[:, 1:] - V_face[:, :-1])  # ∂v/∂x
    dudy = (U_face[1:, :] - U_face[:-1, :])  # ∂u/∂y

    curl = (dvdx[1:-1,:] - dudy[:,1:-1]) 
    curl *= velocity_scale / (piv_step_size* pix_to_um)

    return curl


def downsample_field(field, piv_step_size=64):

    s = int(piv_step_size / 2)
    field = field[s:-s, s:-s] # removing region outside velocity field
    field = block_reduce(field, block_size=piv_step_size, func=np.mean)

    return field


def poly2(p, x):
    return p[0] + p[1] * x + p[2] * x**2


def main():

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("path",     type=str, help="Path to dataset, as config/dataset.json")
    parser.add_argument("PIV_path", type=str, help="Path to PIV, as ../../../../hdd_data/silja/Monolayers/piv/dataset/param/")
    parser.add_argument("-v", "--speed_limit", type=float, help="Limit on distance in velocity space", default=40)
    parser.add_argument("-s", "--sigma", type=float, help="Size of gaussian kernel", default=0)
    parser.add_argument("-a", "--alpha", type=float, help="Size of gaussian kernel", default=0.19)
    args = parser.parse_args()


    dataset = Path(args.path).stem
    config  = json.load(open(f"configs/{dataset}.json"))

    microscope = config["microscope"]
    if config['microscope'] == "holomonitor": microscope = Holomonitor()
    elif config['microscope'] == "tomocube":  microscope = Tomocube()

    # Importing mass field
    if microscope.name == "holomonitor":

        # estimate RI
        cells = SegmentedCells(f"{data_path}cell_features/calibrated/{dataset}_cells.p")
        dn    = poly2(microscope.a, cells.density)

        # import height
        h_field = load_stack(f"{data_path}height_fields/calibrated/{dataset}/", config, data_type="cells")

        # compute mass
        m_field = h_field * dn / args.alpha

    elif microscope.name == "tomocube":

        # import average refractive index and height
        print("loading n_field")
        n_field = load_stack(f"{data_path}ri_fields/calibrated/{dataset}/", config, param="n", data_type="cells")
        print("loading h_field")
        h_field = load_stack(f"{data_path}height_fields/calibrated/{dataset}/", config, data_type="cells")

        m_field = h_field * (n_field - Tomocube().n0) / args.alpha
        print("done loading")

    else:
        m_field = 0
        print("Error: Do not recognize microscope")

    # Not scaling velocity here, but instead leaving it for later analysis
    velocity_scale = 1

    # starting frmo same frame as m_field
    frame = config["cells"]["fmin"]
    for mass_0, mass_1 in tqdm(zip(m_field[:-1], m_field[1:])):


        # Importing velocity field
        vec_position, vec_velocity = import_PIV_frame(args.PIV_path, frame=frame)

        # Compute size of grid 
        nx = len(np.unique(vec_position[0]))
        ny = len(np.unique(vec_position[1]))

        piv_step_size = vec_position[1][1] - vec_position[1][0]

        # Transform PIV input to matrices. Dimensions are given by PIV txt
        U = np.array(vec_velocity[0]).reshape(nx, ny).T * microscope.pix_to_um / microscope.frame_to_h
        V = np.array(vec_velocity[1]).reshape(nx, ny).T * microscope.pix_to_um / microscope.frame_to_h
        X = np.array(vec_position[0]).reshape(nx, ny).T-1
        Y = np.array(vec_position[1]).reshape(nx, ny).T-1

        # remove spurious velocities
        U, V = filter_velocity_fields(X, Y, U, V, speed_limit=args.speed_limit, sigma=0.5)

        mass_flux = finite_volume_mass_flux(mass_0, U, V,
                                            pix_to_um=microscope.pix_to_um, 
                                            piv_step_size=piv_step_size,
                                            velocity_scale=velocity_scale,
                                            smoothing=args.sigma)

        mass_change = finite_volume_mass_change(np.array([mass_0, mass_1]),
                                                pix_to_um=microscope.pix_to_um,
                                                piv_step_size=piv_step_size,
                                                dt=microscope.frame_to_h,
                                                smoothing=args.sigma)


        mass_density = downsample_field(mass_0)
        mass_density = gaussian_filter(mass_density, args.sigma)

        divergence = compute_divergence(U, V, 
                                        pix_to_um=microscope.pix_to_um, 
                                        piv_step_size=piv_step_size,
                                        velocity_scale=velocity_scale)


        # save fields
        Path("results/mass_conservation/").mkdir(parents=True, exist_ok=True)
        Path("results/mass_conservation/frames").mkdir(parents=True, exist_ok=True)
        Path("results/mass_conservation/mass_flux").mkdir(parents=True, exist_ok=True)
        Path("results/mass_conservation/mass_change").mkdir(parents=True, exist_ok=True)
        Path("results/mass_conservation/mass_density").mkdir(parents=True, exist_ok=True)
        Path("results/mass_conservation/velocity_field").mkdir(parents=True, exist_ok=True)
        Path("results/mass_conservation/velocity_divergence").mkdir(parents=True, exist_ok=True)

        imageio.imwrite(f"results/mass_conservation/mass_change/frame_{frame}-{frame+1}.tiff",          np.array(mass_change,  dtype=np.float64))
        imageio.imwrite(f"results/mass_conservation/mass_flux/frame_{frame}-{frame+1}.tiff",            np.array(mass_flux,    dtype=np.float64))
        imageio.imwrite(f"results/mass_conservation/mass_density/frame_{frame}.tiff",                   np.array(mass_density, dtype=np.float64))
        imageio.imwrite(f"results/mass_conservation/velocity_divergence/frame_{frame}-{frame+1}.tiff",  np.array(divergence,   dtype=np.float64))
        imageio.imwrite(f"results/mass_conservation/velocity_field/sigma_{int(args.sigma)}_x_frame_{frame}-{frame+1}.tiff",     np.array(U,            dtype=np.float64))
        imageio.imwrite(f"results/mass_conservation/velocity_field/sigma_{int(args.sigma)}_y_frame_{frame}-{frame+1}.tiff",     np.array(V,            dtype=np.float64))



        if args.sigma == 0:
            vmax = 240
        else:
            vmax = 240 / args.sigma

        fig, ax = plt.subplots(1,3, figsize=(11,3))
        im1 = ax[0].imshow(mass_change, cmap="RdBu_r", origin="lower", extent=[0,567,567,0], vmin=-vmax,     vmax=vmax)
        im0 = ax[1].imshow(mass_flux,   cmap="RdBu_r", origin="lower", extent=[0,567,567,0], vmin=-vmax,     vmax=vmax)
        im2 = ax[2].imshow(mass_change - mass_flux,    origin="lower", extent=[0,567,567,0], vmin=-2*vmax/3, vmax=2*vmax/3)#, cmap="RdBu_r", extent=[0,567,567,0], vmin=-200, vmax=200)

        fig.colorbar(im0, ax=ax[0])
        fig.colorbar(im1, ax=ax[1])
        fig.colorbar(im2, ax=ax[2])

        ax[0].set(title=r"Mass change $\left(\frac{\text{µg}}{\text{h}\cdot\text{mm}^2}\right)$", xlabel=r"x (µm)", ylabel=r"y (µm)");
        ax[1].set(title=r"Mass flux $\left(\frac{\text{µg}}{\text{h}\cdot\text{mm}^2}\right)$", xlabel=r"x (µm)", ylabel=r"y (µm)");
        ax[2].set(title=r"Difference $\left(\frac{\text{µg}}{\text{h}\cdot\text{mm}^2}\right)$", xlabel=r"x (µm)", ylabel=r"y (µm)");

        fig.tight_layout()
        fig.savefig(f"figs/frames/mass_conservation/sigma_{int(args.sigma)}_frame_{frame}-{frame+1}.png")
        plt.close(fig)



        frame += 1



if __name__ == "__main__":
    main()

