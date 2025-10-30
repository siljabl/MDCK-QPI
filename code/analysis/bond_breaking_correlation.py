"""
Saves and plots bond breaking correlation
"""

import sys
import json
import numpy as np
import pandas as pd
import trackpy as tp
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("Agg")

from tqdm import tqdm
from pathlib import Path
from scipy.spatial import Delaunay
from skimage.measure import regionprops

sys.path.append("code/preprocessing/utils/")
from file_handling import import_stack
from segment2D import get_pixel_size


# experimental parameters
frames_to_hours = 1 / 12
pix_to_um = get_pixel_size()[1]

# tracking
search_range = 20
memory = 5
threshold = 5


# load dataset
path = sys.argv[1]
dataset = Path(path).stem
config  = json.load(open(f"data/experimental/configs/{dataset}.json"))
im_areas   = np.load(f"data/experimental/processed/{dataset}/im_cell_areas.npy")
im_heights = import_stack(f"data/experimental/raw/{dataset}/", config)


# get region properties
cellprops = [regionprops(im_areas[i], im_heights[i]) for i in range(len(im_areas))]
densities = [10**6 * len(cells) / (np.sum(areas > 0) * pix_to_um**2) for cells, areas in zip(cellprops, im_areas)]


# prepare data frame
positions = np.concatenate([[cell.centroid_weighted for cell in cellprop] for cellprop in cellprops])
frames    = np.concatenate([[frame for cell in cellprops[frame]] for frame in range(len(cellprops))])
labels    = np.concatenate([[cell.label for cell in cellprop] for cellprop in cellprops])

cell_df = pd.DataFrame({'y': positions.T[0],
                        'x': positions.T[1],
                        'label': labels,
                        'frame': frames})


# tracks cells
tracks = tp.link(cell_df, search_range=search_range, memory=memory);
tracks = tp.filter_stubs(tracks, threshold=threshold);


# get arrays for neighbour computation
particle_id = tracks.particle.values
positions   = np.array([tracks.y.values, tracks.x.values]).T
frames      = tracks.frame.values


# perpare matrix
Nframes = np.max(frames)
Ncells  = np.max(tracks.particle) + 1
neighbors_matrix = np.zeros([Nframes, Ncells, Ncells])


# fill matrix
for frame in tqdm(range(len(neighbors_matrix))):
    
    # triangualtion
    tri = Delaunay(positions[frames == frame])

    # loop through cells
    for cell_idx in range(len(positions[frames == frame])): 

        label = particle_id[frames == frame][cell_idx]

        # Find neighbors from triangulation
        neighbors_idx = set()
        for simplex in tri.simplices:

            if cell_idx in simplex:
                neighbors_matrix[frame, label, particle_id[frames == frame][simplex]] = 1
                
                neighbors_idx.update(simplex)

# remove diagonal
for i in range(len(neighbors_matrix[0])):
    neighbors_matrix[:, i,i] = 0



# compute bond_breaking correlations
bond_breaking = np.zeros([Nframes, Nframes])

for i in tqdm(range(Nframes)):
    j = Nframes - i
    bond_breaking[:j,i] = np.sum((neighbors_matrix * np.roll(neighbors_matrix, shift=-i, axis=0))[:j], axis=(1,2))

for i in range(Nframes):
    bond_breaking[i] /= bond_breaking[i,0]


# compute distribution of # neighbours
Nneighbors = np.sum(neighbors_matrix, axis=2)
nmax = int(np.max(Nneighbors))
neighbors_dist = np.zeros([Nframes, nmax])

for f in range(Nframes):
    neighbors_dist[f] = np.histogram(Nneighbors[f], density=True, range=(1, nmax), bins=nmax)[0]


# prepare to plot
frames = np.arange(Nframes) * frames_to_hours

# define density colormap
densities  = np.array(densities)
Ndensities = len(densities)

cmap   = mpl.colormaps['plasma']
colors = cmap(np.linspace(0.1, 0.9, Ndensities))
sm     = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=densities.min(), vmax=densities.max()))


fig, ax = plt.subplots(1,2, figsize=(9,3.5))

ax[0].axhline(0, ls='dashed', color="gray")
im = ax[1].imshow(bond_breaking, aspect="auto", extent=[0, Nframes*frames_to_hours, densities.min(), densities.max()])

for i in range(Nframes):
    ax[0].plot(frames[bond_breaking[i] > 0], bond_breaking[i][bond_breaking[i] > 0], color=colors[i])


ax[0].set(xlabel="$t ~(h)$", ylabel=r"$C_b(t)$")
ax[1].set(xlabel="$t ~(h)$", ylabel=r"$\rho_{cells}$")

fig.colorbar(sm, ax=ax[0], label=r"$\rho_{cells}$")
fig.colorbar(im, ax=ax[1], label=r"$C_b(t)$")
fig.tight_layout()
fig.savefig(f"results/{dataset}/bond_breaking_correlation.png", dpi=300)


# save output
np.save(f"data/experimental/processed/{dataset}/bond_breaking_correlation.npy", bond_breaking)
np.save(f"data/experimental/processed/{dataset}/neighbours_distribution.npy", neighbors_dist)
np.save(f"data/experimental/processed/{dataset}/densities.npy", densities)