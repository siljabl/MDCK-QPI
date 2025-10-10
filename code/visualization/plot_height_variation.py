
import os
import sys
import glob
import argparse
import numpy as np
import seaborn as sns
from pathlib import Path

# Avoid localhost error on my machine
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')

sys.path.append("code/analysis")
from utils.data_class import VariationData



def initialize_figure(type):
    """ Create figure """

    plt.figure(figsize=(6, 4), dpi=300)
    plt.xlabel(r'$\rho_{cell}~(mm^{-2})$')

    if type == 'r':
        plt.ylabel(r'$\hat{\sigma}_{h,xy}$')

    else:
        plt.ylabel(r'$\hat{\sigma}_{h,t}$')



def save_plot(figure, out_path):
    """ Save the generated plot to a specific directory. """
    figure.savefig(out_path)
    print(f"Plot saved to {out_path}")



def main():
    fig_dir = "results/"

    parser = argparse.ArgumentParser(description="Plot all defined autocorrelations")
    #parser.add_argument('filepattern',   type=str, help="Path to files to plot. Typically 'data/simulated/obj/file'. Filename is on form <'path/to/file'>*.autocorr")
    parser.add_argument('pattern',       type=str, help="Correlation variable (t or r)")
    parser.add_argument('var',           type=str, help="Correlation variable (t or r)")
    parser.add_argument('--plot_unbinned', action="store_true")
    args = parser.parse_args()

    # Assert temporal or spatial correlation
    assert args.var in ['r', 't'], "Wrong correlation variable. Must be r or t"


    # Bin data

    # Define line colors
    # cmap   = mpl.colormaps[args.cmap]
    # colors = cmap(np.linspace(0.1, 0.9, len(files_list)))


    # Plot each data set
    initialize_figure(args.var)

    color = ['r', 'b', 'g', 'm']

    i = 0
    for path in glob.glob(f"{args.pattern}*/height_variations.p"):
        
        # skipping technical and biological replicates
        if "2024" in path:
           continue

        # Load data
        dataset = Path(path).parent.stem
        data = VariationData(path)


        if args.plot_unbinned:
            plt.plot(data.density, data.var_pixel, '.', c=color[i], label=dataset)
            plt.plot(data.density, data.var_disk, 'o',  c=color[i], label=dataset)
            plt.plot(data.density, data.var_cell, '*',  c=color[i], label=dataset)

        else:
            if data.datatype == "binned":
                # Plot
                if args.var == "r":
                    # plot pixel
                    plt.errorbar(data.density, 
                                 data.var_pixel, 
                                 yerr=data.svar_pixel, 
                                 fmt='.', 
                                #  lw=2,
                                #  ms=5,
                                #  capsize=2, 
                                #  capthick=2,
                                 label=dataset)
                    # plot disk
                    plt.errorbar(data.density, 
                                 data.var_disk, 
                                 yerr=data.svar_disk, 
                                 fmt='.', 
                    #              lw=2,
                    #              ms=5,
                    #              capsize=2, 
                    #              capthick=2,
                                 label=dataset)
                    # plot cell
                    plt.errorbar(data.density, 
                                 data.var_cell, 
                                 yerr=data.svar_cell, 
                                 fmt='.', 
                    #              lw=2,
                    #              ms=5,
                    #              capsize=2, 
                    #              capthick=2,
                                 label=dataset)
            else:
                binned_pixel = data.bin_data(data.var_pixel)
                binned_disk  = data.bin_data(data.var_disk)
                binned_cell  = data.bin_data(data.var_cell)
                density = (binned_pixel["density_bins"][1:] + binned_pixel["density_bins"][:-1]) / 2

                # pixel
                plt.errorbar(density, 
                             binned_pixel["mean"], 
                             yerr=binned_pixel["std"], 
                             fmt='.', 
                            #  lw=2,
                            #  ms=5,
                            #  capsize=2, 
                            #  capthick=2,
                             label=dataset)
                
                # disk
                plt.errorbar(density, 
                             binned_disk["mean"], 
                             yerr=binned_disk["std"], 
                             fmt='.', 
                            #  lw=2,
                            #  ms=5,
                            #  capsize=2, 
                            #  capthick=2,
                             label=dataset)

                # cell
                plt.errorbar(density, 
                             binned_cell["mean"], 
                             yerr=binned_cell["std"], 
                             fmt='.', 
                            #  lw=2,
                            #  ms=5,
                            #  capsize=2, 
                            #  capthick=2,
                             label=dataset)

        i += 1

    out_path = "results/spatial_variation.png"
    plt.legend()
    plt.ylim(0.10, 0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")



if __name__ == "__main__":
    main()

