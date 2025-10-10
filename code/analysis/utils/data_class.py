import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.append("code/preprocessing/utils")
sys.path.append("code/analysis/")
import utils.masked_correlation_functions as compute
from utils.variation_functions import global_density


class SegmentationData:
    def __init__(self, path):
        self.date = datetime.today().strftime('%Y/%m/%d_%H:%M')
        self.path = path

        if os.path.isfile(self.path):
            self.load(path)



    def transform_df_to_ma(self, df, xy_to_um):

        x = df.pivot(index='frame', columns='particle', values='x').to_numpy()# * xy_to_um
        y = df.pivot(index='frame', columns='particle', values='y').to_numpy()# * xy_to_um
        A = df.pivot(index='frame', columns='particle', values='A').to_numpy()# * xy_to_um**2
        h = df.pivot(index='frame', columns='particle', values='h_avrg').to_numpy()

        label  = df.pivot(index='frame', columns='particle', values='label').to_numpy()
        aminor = df.pivot(index='frame', columns='particle', values='a_min').to_numpy()# * xy_to_um
        amajor = df.pivot(index='frame', columns='particle', values='a_max').to_numpy()# * xy_to_um

        self.x = np.ma.masked_where(np.isnan(x), x)
        self.y = np.ma.masked_where(np.isnan(y), y)
        self.A = np.ma.masked_where(np.isnan(A), A)
        self.h = np.ma.masked_where(np.isnan(h), h)

        self.dx = np.ma.diff(self.x, axis=0)
        self.dy = np.ma.diff(self.y, axis=0)

        self.label = np.ma.masked_where(np.isnan(label), label)
        self.aminor = np.ma.masked_where(np.isnan(aminor), aminor)
        self.amajor = np.ma.masked_where(np.isnan(amajor), amajor)

        self.density = global_density(self.A)

        try:
            n = df.pivot(index='frame', columns='particle', values='n_avrg').to_numpy()
            self.n = np.ma.masked_where(np.isnan(n), n)
        except:
            pass


    def load(self, path):
        """
        Loads the state from a pickle file.

        Parameters:
        - path: path to pickle to load.
        """
        
        # Load pickle
        with open(f"{path}", 'rb') as f:
            state = pickle.load(f)
        
        # Update object
        self.x = state.get('x', {})
        self.y = state.get('y', {})
        self.h = state.get('h', {})
        self.A = state.get('A', {})

        self.dx = state.get('dx', {})
        self.dy = state.get('dy', {})

        self.label  = state.get('label', {})
        self.aminor = state.get('aminor', {})
        self.amajor = state.get('amajor', {})

        self.density = state.get('density', {})

        try:
            self.n = state.get('n', {})
        except:
            pass

        print(f"State loaded from {path}.")


    def save(self, path):
        """ Saves object as pickle"""

        # Prepare state dictionary to save
        state = {
            'x': self.x,
            'y': self.y,
            'h': self.h,
            'A': self.A,
            'dx': self.dx,
            'dy': self.dy,
            'label': self.x,
            'aminor': self.aminor,
            'amajor': self.amajor, 
            'density': self.density
        }

        try:
            state['n'] = self.n
        except:
            pass
        
        # Save
        with open(f"{path}", 'wb') as f:
            pickle.dump(state, f)

        print(f"State saved to {path}")


    def add(self, param, value):
        if param == "density":
            self.density = value


class VariationData:
    def __init__(self, path):

        self.datatype = "empty"
        self.path = path

        if os.path.isfile(self.path):
            self.load()



    def add_unbinned_data(self, density, height, variation, measure):
        self.datatype = "unbinned"
        self.density  = density

        if measure == "pixel":
            self.var_pixel = variation
            self.h_pixel   = height

        elif measure == "disk":
            self.var_disk = variation
            self.h_disk   = height
        
        elif measure == "cell":
            self.var_cell = variation
            self.h_cell   = height



    def add_binned_data(self, density, variation, svariation, measure):
        self.datatype = "binned"
        self.density  = density

        if measure == "pixel":
            self.var_pixel  = variation
            self.svar_pixel = svariation
            # self.h_pixel  = height
            # self.sh_pixel = sheight

        elif measure == "disk":
            self.var_disk  = variation
            self.svar_disk = svariation
            # self.h_disk  = height
            # self.sh_disk = sheight
        
        elif measure == "cell":
            self.var_cell  = variation
            self.svar_cell = svariation
            # self.h_cell  = height
            # self.sh_cell = sheight



    def load(self):
        # Load pickle
        with open(f"{self.path}", 'rb') as f:
            state = pickle.load(f)
        
        # Update object
        self.density  = state.get('density', {})
        self.datatype = state.get('datatype', {})

        self.var_pixel = state.get('var_pixel', {})
        self.var_disk  = state.get('var_disk', {})
        self.var_cell  = state.get('var_cell', {})

        if self.datatype == "binned":
            self.svar_pixel = state.get('svar_pixel', {})
            self.svar_disk  = state.get('svar_disk', {})
            self.svar_cell  = state.get('svar_cell', {})
            self.sh_pixel = state.get('sh_pixel', {})
            self.sh_disk  = state.get('sh_disk', {})
            self.sh_cell  = state.get('sh_cell', {})

        if self.datatype == "unbinned":
            self.h_pixel = state.get('h_pixel', {})
            self.h_disk  = state.get('h_disk', {})
            self.h_cell  = state.get('h_cell', {})

        print(f"State loaded from {self.path}.")



    def save(self):
        """ Saves object as pickle"""

        # Prepare state dictionary to save
        state = {
            'density':   self.density,
            'datatype':  self.datatype,
            'var_pixel': self.var_pixel,
            'var_disk':  self.var_disk,
            'var_cell':  self.var_cell,
            }
        
        if self.datatype == "binned":
            state['svar_pixel'] = self.svar_pixel
            state['svar_disk']  = self.svar_disk
            state['svar_cell']  = self.svar_cell
            # state['sh_pixel'] = self.sh_pixel
            # state['sh_disk']  = self.sh_disk
            # state['sh_cell']  = self.sh_cell
        
        if self.datatype == "unbinned":
            state['h_pixel'] = self.h_pixel
            state['h_disk']  = self.h_disk
            state['h_cell']  = self.h_cell

        # Save
        with open(f"{self.path}", 'wb') as f:
            pickle.dump(state, f)

        print(f"State saved to {self.path}")



    def bin_data(self, variable, bin_size=100):

        min_bin = int(np.min(self.density) / bin_size) * bin_size
        max_bin = int(np.max(self.density) / bin_size) * bin_size
        bins = np.arange(min_bin, max_bin + bin_size, bin_size)

        mean_variable = np.zeros(len(bins) - 1)
        std_variable  = np.zeros(len(bins) - 1)
        counts        = np.zeros(len(bins) - 1)

        density_idx = np.digitize(self.density, bins)

        for i in range(1, len(bins)):
            idx_in_bin = np.where(density_idx == i)[0]
            counts[i-1] = len(idx_in_bin)

            if counts[i-1] == 0:
                #mean_variable[i-1] = np.nan
                #std_variable[i-1]  = np.nan
                continue

            mean_variable[i-1] = np.mean(variable[idx_in_bin])
            if counts[i-1] > 1:
                std_variable[i-1]  = np.std(variable[idx_in_bin], ddof=1)



        output = {
            "density_bins": bins,
            "mean": mean_variable,
            "std": std_variable
            #"N_in_bin": counts
        }

        return output
            



class AutocorrelationData:
    def __init__(self, path):

        self.path = path

        if os.path.isfile(self.path):
            self.load()

        else:
            self.temporal_cell = {}
            self.temporal = {}
            self.spatial  = {}
            self.t_array  = {}
            self.r_array  = {}
            self.density  = {}
            self.log = {'t_cell': {},
                        't': {},
                        'r': {}}



    def load(self):
        """
        Loads the state from a pickle file.

        Parameters:
        - path: path to pickle to load.
        """
        
        # Load pickle
        with open(f"{self.path}", 'rb') as f:
            state = pickle.load(f)
        
        # Update object
        self.temporal_cell = state.get('temporal_cell', {})
        self.temporal = state.get('temporal', {})
        self.spatial  = state.get('spatial', {})
        self.t_array  = state.get('t_array', {})
        self.r_array  = state.get('r_array', {})
        self.density  = state.get('density', {})
        self.log      = state.get('log', {})

        print(f"State loaded from {self.path}.")



    def save(self):
        """ Saves object as pickle"""

        # Prepare state dictionary to save
        state = {
            'temporal_cell': self.temporal_cell,
            'temporal': self.temporal,
            'spatial':  self.spatial,
            't_array':  self.t_array,
            'r_array':  self.r_array,
            'density':  self.density,
            'log':      self.log
        }
        
        # Save
        with open(f"{self.path}", 'wb') as f:
            pickle.dump(state, f)

        print(f"State saved to {self.path}")



    def add_density(self, masked_areas):
        self.density = global_density(masked_areas)
        self.save()



    def compute_spatial(self, positions, variable, variable_name, dr, r_max, t_avrg=False, overwrite=False):
        """ Computes spatial autocorrelation """

        # Check if correlation exists
        if not overwrite:
            if variable_name in self.spatial.keys():
                print(f"Spatial autocorrelation of {variable_name} already exists.")
                return

        # Compute autocorrelation
        fmax = variable.shape[-2]

        Cr = compute.general_spatial_correlation(positions[0][:fmax], positions[1][:fmax], variable,
                                                 dr=dr, r_max=r_max, t_avrg=t_avrg)

        # Update object
        self.spatial[variable_name]  = Cr['C_norm']#.compressed()
        self.r_array[variable_name]  = Cr['r_bin_centers']#.compressed()
        self.log['r'][variable_name] = datetime.today().strftime('%Y/%m/%d_%H:%M')



    def compute_temporal(self, variable, variable_name, t_max, mean_var='r', t_avrg=False, overwrite=False):
        """ Computes temporal autocorrelation """

        assert mean_var == 'r' or mean_var == 'cell', "Fluctuations must be computed w.r.t. space average ('r') or cell average ('cell')"

        # Check if correlation exists
        if not overwrite:
            if mean_var == 'r':
                if variable_name in self.temporal.keys():
                    print(f"Temporal autocorrelation of {variable_name} already exists.")
                    return

            else:
                if variable_name in self.temporal_cell.keys():
                    print(f"Temporal autocorrelation of {variable_name} already exists.")
                    return


        # Compute autocorrelation    
        Ct = compute.general_temporal_correlation(variable, t_max=t_max, t_avrg=t_avrg)

        # Update object
        self.t_array[variable_name]  = np.arange(t_max)

        if mean_var == 'r':
            self.temporal[variable_name] = Ct['C_norm']
            self.log['t'][variable_name] = datetime.today().strftime('%Y/%m/%d_%H:%M')

        else:
            # add t_cell to log if not added earlier
            if 't_cell' not in self.log.keys():
                self.log['t_cell'] = {}

            self.temporal_cell[variable_name] = Ct['C_norm']
            self.log['t_cell'][variable_name] = datetime.today().strftime('%Y/%m/%d_%H:%M')


    
    def bin_data(self, variable, parameter, bin_size=200):
        
        # Assertions to ensure correct data exists
        assert variable == "r" or variable == "t", "Variable must be either 'r' or 't'."
        assert parameter == "hh" or parameter == "vv" or parameter == "AA" or parameter == "VV", "Parameter must be 'hh', 'AA', 'VV', or 'vv'."

        if len(self.density) == 0:
            print("Must add density before binning!")

        else:

            # Spatial correlation
            if variable == "r":
                correlation = self.spatial[parameter]

            # Temporal correlation
            elif variable == "t":
                correlation = self.temporal[parameter]

            elif variable == "t_cell":
                correlation = self.temporal[parameter]


            # Ensure same length on density array as correlation array
            if len(self.density) > len(correlation[0]):
                print(f"Density array is longer than correlation array! Forcing equal length {len(correlation)}")
                self.density = self.density[:len(correlation)]

            # Create density bins
            min_bin = int(np.min(self.density) / bin_size) * bin_size
            max_bin = int(np.max(self.density) / bin_size) * bin_size
            bins = np.arange(min_bin, max_bin + bin_size, bin_size)

            # Empty arrays for computation
            mean_correlation = np.zeros([len(bins) - 1, len(correlation[0])])
            std_correlation  = np.zeros([len(bins) - 1, len(correlation[0])])
            counts           = np.zeros(len(bins) - 1)

            # Sort densities by bins
            density_idx = np.digitize(self.density, bins)

            for i in range(1, len(bins)):

                # Count densities in bin
                idx_in_bin = np.where(density_idx == i)[0]
                counts[i-1] = len(idx_in_bin)

                if counts[i-1] == 0:
                    #mean_variable[i-1] = np.nan
                    #std_variable[i-1]  = np.nan
                    continue

                # Compute mean
                mean_correlation[i-1] = np.mean(correlation[idx_in_bin], axis=0)

                # Compute standard deviation if more than one density in bin
                if counts[i-1] > 1:
                    std_correlation[i-1]  = np.std(correlation[idx_in_bin], axis=0, ddof=1)



            output = {
                "density_bins": bins,
                "mean": np.ma.array(mean_correlation, mask=mean_correlation==0),
                "std": np.ma.array(std_correlation, mask=mean_correlation==0),
                "N_in_bin": counts
            }

            return output
        


