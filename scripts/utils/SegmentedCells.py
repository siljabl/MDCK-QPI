import os
import pickle
import numpy as np
from datetime import datetime



class SegmentedCells:
    def __init__(self, path):
        self.date = datetime.today().strftime('%Y/%m/%d_%H:%M')
        self.path = path

        if os.path.isfile(self.path):
            self.load(path)



    def transform_df_to_ma(self, df, xy_to_um):

        x = df.pivot(index='frame', columns='particle', values='x').to_numpy()    * xy_to_um
        y = df.pivot(index='frame', columns='particle', values='y').to_numpy()    * xy_to_um
        A = df.pivot(index='frame', columns='particle', values='area').to_numpy() * xy_to_um**2
        h = df.pivot(index='frame', columns='particle', values='hmean').to_numpy()  # already in µm

        label  = df.pivot(index='frame', columns='particle', values='label').to_numpy()
        aminor = df.pivot(index='frame', columns='particle', values='a_min').to_numpy() * xy_to_um
        amajor = df.pivot(index='frame', columns='particle', values='a_max').to_numpy() * xy_to_um
        theta  = df.pivot(index='frame', columns='particle', values='theta').to_numpy()
        ecc    = df.pivot(index='frame', columns='particle', values='ecc').to_numpy()

        self.x = np.ma.masked_where(np.isnan(x), x)
        self.y = np.ma.masked_where(np.isnan(y), y)
        self.A = np.ma.masked_where(np.isnan(A), A)
        self.h = np.ma.masked_where(np.isnan(h), h)

        self.dx = np.ma.diff(self.x, axis=0)
        self.dy = np.ma.diff(self.y, axis=0)

        self.label  = np.ma.masked_where(np.isnan(label),  label)
        self.aminor = np.ma.masked_where(np.isnan(aminor), aminor)
        self.amajor = np.ma.masked_where(np.isnan(amajor), amajor)
        self.theta  = np.ma.masked_where(np.isnan(amajor), theta)
        self.ecc    = np.ma.masked_where(np.isnan(amajor), ecc)

        self.density = 10 ** 6 / np.ma.mean(self.A, axis=1)

        try:
            n = df.pivot(index='frame', columns='particle', values='nmean').to_numpy()
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
        self.theta  = state.get('theta', {})
        self.ecc    = state.get('ecc', {})

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
            'theta':  self.theta, 
            'ecc': self.ecc, 
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


    def remove_cells(self, mask):
        self.x.data[~mask] = 0
        self.y.data[~mask] = 0
        self.h.data[~mask] = 0
        self.A.data[~mask] = 0

        self.x.mask[~mask] = True
        self.y.mask[~mask] = True
        self.h.mask[~mask] = True
        self.A.mask[~mask] = True

        self.label.data[~mask]  = 0
        self.aminor.data[~mask] = 0
        self.amajor.data[~mask] = 0
        self.theta.data[~mask]  = 0
        self.ecc.data[~mask]    = 0

        self.label.mask[~mask]  = True
        self.aminor.mask[~mask] = True
        self.amajor.mask[~mask] = True
        self.theta.mask[~mask]  = True
        self.ecc.mask[~mask]    = True

        self.dx = np.ma.diff(self.x, axis=0)
        self.dy = np.ma.diff(self.y, axis=0)
        self.density = 10 ** 6 / np.ma.mean(self.A, axis=1)
