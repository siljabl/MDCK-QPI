import os
import sys
import glob
import platform
import argparse
import numpy as np
from pathlib import Path

# Append the path of relative_parent directories
sys.path.append("code/analysis/utils")
from data_class import AutocorrelationData

# Avoid localhost error on my machine
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')



def initialize_figure(varname, type):
    """ Create figure """

    fig, ax = plt.subplots(1,1, figsize=(6, 4), dpi=300)

    if type == 'r':
        plt.title(rf'$C_{{{varname}}}(r)$')
        plt.xlabel(r'$r~(µm)$')
        plt.axhline(0, 0, 1, linestyle="dashed", color="gray")

    else:
        plt.title(rf'$C_{{{varname}}}(t)$')
        plt.xlabel(r'$t ~(h)$')
        plt.axhline(0, 0, 1, linestyle="dashed", color="gray")

    return fig, ax



def save_plot(figure, out_path):
    """ Save the generated plot to a specific directory. """
    figure.savefig(out_path)
    print(f"Plot saved to {out_path}")


def plot_correlation(autocorr, density_idx, color, args, frame_to_hour=1/12):

    # Plot
    if args.var == "r":
        x = autocorr.r_array[args.param]
        y = autocorr.spatial[args.param][density_idx]
        plt.plot(x[(x <= args.xlim) * (y <= args.ylim)], y[(x <= args.xlim) * (y <= args.ylim)],
                    args.fmt,
                    color=color)
    
    else:
        # Get persistence time
        x = autocorr.t_array[args.param] * frame_to_hour
        y = autocorr.temporal[args.param][density_idx]
        plt.plot(x[(x <= args.xlim) * (y <= args.ylim)], y[(x <= args.xlim) * (y <= args.ylim)],
                    args.fmt,
                    color=color)
        



def main():
    fig_dir = "results/"

    parser = argparse.ArgumentParser(description="Plot all defined autocorrelations")
    parser.add_argument('dir',          type=str,   help="Path to file to plot. Typically 'data/experimental/processed/<dataset>")
    parser.add_argument('param',         type=str,   help="Parameter to plot correlation of (varvar)")
    parser.add_argument('var',           type=str,   help="Correlation variable (t or r)")
    parser.add_argument('-c','--cmap',   type=str,   help="Specify matplotlib colormap (str)",        default='plasma')
    parser.add_argument('-o','--outdir', type=str,   help="Output directory",                         default="results/")
    parser.add_argument('-x', '--xlim',  type=float, help="Upper limit on x-axis", default=9999)
    parser.add_argument('-y', '--ylim',  type=float, help="Upper limit on y-axis", default=1.1)
    parser.add_argument('-b','--binned', action="store_true", help="Bin data by density")
    parser.add_argument('--xlog',        action="store_true")
    parser.add_argument('--ylog',        action="store_true")
    parser.add_argument('--log',         action="store_true")
    parser.add_argument('--cell',        action="store_true")
    parser.add_argument('--field',       action="store_true")
    parser.add_argument('--fmt',         default="-")
    parser.add_argument('-r', '--return_plot', action="store_true")
    args = parser.parse_args()

    # Assert temporal or spatial correlation
    assert args.var in ['r', 't'], "Wrong correlation variable. Must be r or t"
    assert args.cell or args.field, "Must specify cell or field"

    dataset = Path(args.dir).stem
    if args.cell:
        file = f"{args.dir}cell_autocorr.p"

    if args.field:
        file = f"{args.dir}field_autocorr.p"


    # Import data
    autocorr  = AutocorrelationData(file)
    densities = np.array(autocorr.density)

    # Define line colors
    cmap   = mpl.colormaps[args.cmap]


    # Plot each data set
    fig, ax = initialize_figure(args.param, args.var)
    if args.binned:
        autocorr_binned = autocorr.bin_data(args.var, args.param, update_obj=True, bin_size=100)
        Nbins = len(autocorr_binned['mean'])
        densities = (autocorr_binned['density_bins'][1:] + autocorr_binned['density_bins'][:-1]) / 2
        
        colors = cmap(np.linspace(0.1, 0.9, len(densities)))
   
        for i in range(Nbins):
            plot_correlation(autocorr, i, colors[i], args)

    else:
        colors = cmap(np.linspace(0.1, 0.9, len(densities)))
        
        try:
            Nframes = len(autocorr.temporal[args.param])
        except:
            Nframes = len(autocorr.spatial[args.param])
        
        for i in range(Nframes):
            plot_correlation(autocorr, i, colors[i], args)

            
        
    sm = plt.cm.ScalarMappable(cmap=args.cmap, norm=plt.Normalize(vmin=min(densities), vmax=max(densities)))
    fig.colorbar(sm, ax=ax, label=r"$\rho_{cell}$")


    # Ouput path
    if args.var == "r":
        out_path = f"{fig_dir}{dataset}/spatial_autocorrelation_{args.param}.png"
    else:
        out_path = f"{fig_dir}{dataset}/temporal_autocorrelation_{args.param}.png"

    # log-scale
    if args.xlog or args.log:
        plt.xscale("log")
    if args.ylog or args.log:
        plt.yscale("log")

    if args.return_plot:
        return fig

    else:
        fig.tight_layout()
        fig.savefig(out_path)
        print(f"Plot saved to {out_path}")



if __name__ == "__main__":
    main()

