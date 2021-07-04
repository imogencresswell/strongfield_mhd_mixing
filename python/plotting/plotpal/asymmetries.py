import os
import logging
from collections import OrderedDict
from sys import stdout
from sys import path

import numpy as np
import h5py
from mpi4py import MPI
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 9})

from dedalus.tools.parallel import Sync

from plotpal.file_reader import SingleFiletypePlotter
from plotpal.plot_grid import PlotGrid

logger = logging.getLogger(__name__.split('.')[-1])


class AsymmetryPlotter(SingleFiletypePlotter):
    """
    A class for plotting 1D profiles that help understand asymmetries.

    1D profiles are made from 2D slices.
    For each slice, 2 profiles are made: one where a provided "mask" is positive,
    the other where the provided mask is negative.

    Example --  Produces a profile of Temperature vs. height, 
    but creates one profile where velocity is positive, and another where it is negative.

    # Public Methods
    - __init__()
    - calculate_profiles()
    - plot_profiles()

    # Attributes
        profs   (OrderedDict) :
        A dictionary of the produced asymmetric profiles and the masks used to make them.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the PDF plotter.

        # Arguments
            *args, **kwargs : Additional keyword arguments for super().__init__() (see file_reader.py)
        """
        super(AsymmetryPlotter, self).__init__(*args, distribution='even', **kwargs)
        self.profs = OrderedDict()
        self.basis_k = None

    def calculate_profiles(self, prof_list, mask_list, avg_axis=0, basis='z', **kwargs):
        """
        Calculate asymmetry profiles for specified task.

        Given a 2D slice to make a profile out of (p), as well as a 2D slice for masking that profile (m), create two profiles:
            1. p where m >= 0
            2. p where m <  0
        Then average that 2D slice over the basis that isn't specified

        # Arguments
            prof_list (list) :
                The names of the tasks to create profiles of
            mask_list (list) :
                The names of tasks to use as masks
            avg_axis (int) :
                The axis over which to take the average to go from 2D -> 1D profile
            basis (string) :
                The Dedalus basis which the profile varies against
            **kwargs (dict) :
                additional keyword arguments for the self._get_interpolated_slices() function.
        """
        with self.my_sync:
            if self.idle : return
           
            for i, keys in enumerate(zip(prof_list, mask_list)):
                k, m = keys
                N = 0
                local_pos_prof_sum = None
                local_neg_prof_sum = None
                while self.files_remain(bases, tasks):
                    bs, tsk, writenum, times = self.read_next_file()
                    if local_pos_prof_sum is None:
                        local_pos_prof_sum = np.zeros(len(bs[basis]))
                        local_neg_prof_sum = np.zeros(len(bs[basis]))
                    for j in range(len(times)):
                        masked_pos = np.ma.masked_where(tsk[m][j,:] <  0, tsk[k][j,:])
                        masked_neg = np.ma.masked_where(tsk[m][j,:] >= 0, tsk[k][j,:])
                        local_pos_prof_sum += masked_pos.mean(axis=avg_axis)
                        local_neg_prof_sum += masked_neg.mean(axis=avg_axis)
                        N += 1

                total_writes_global  = np.zeros(1, dtype=np.float64)
                global_pos_prof_sum  = np.zeros(len(local_pos_prof_sum), dtype=np.float64)
                global_neg_prof_sum  = np.zeros(len(local_neg_prof_sum), dtype=np.float64)
                self.dist_comm.Allreduce(np.array([N,], dtype=np.float64), total_writes_global, op=MPI.SUM)
                self.dist_comm.Allreduce(np.array(local_pos_prof_sum, dtype=np.float64), global_pos_prof_sum, op=MPI.SUM)
                self.dist_comm.Allreduce(np.array(local_neg_prof_sum, dtype=np.float64), global_neg_prof_sum, op=MPI.SUM)

                self.profs['{:s}-{:s}'.format(k, m)] = [global_pos_prof_sum/total_writes_global,
                                                        global_neg_prof_sum/total_writes_global]
                if basis not in self.profs.keys():
                    self.profs[basis] = bs[basis]
                    self.basis_k = basis


    def plot_profiles(self, dpi=150, **kwargs):
        """
        Plot the asymmetric profiles and save to file

        # Arguments
            dpi (int, optional) :
                Pixel density of output image.
            **kwargs (dict) :
                additional keyword arguments for PlotGrid()
        """
        with self.my_sync:
            if self.reader.comm.rank != 0: return

            grid = PlotGrid(1,1, **kwargs)
            ax = grid.axes['ax_0-0']
            
            for k, data in self.profs.items():
                if k == self.basis_k: continue
                field, m = k.split('-')
                ax.plot(self.profs[self.basis_k], data[0], lw=2, c='k', label='positive {}'.format(m))
                ax.plot(self.profs[self.basis_k], data[1], lw=2, c='k', label='negative {}'.format(m), ls='-.')
                ax.set_xlim(self.profs[self.basis_k].min(), self.profs[self.basis_k].max())
                ax.set_xlabel(self.basis_k)
                ax.set_ylabel('{:s}'.format(field))
                ax.legend(loc='best')

                grid.fig.savefig('{:s}/{:s}.png'.format(self.out_dir, k), dpi=dpi, bbox_inches='tight')
                ax.clear()

            self._save_profs()

    def _save_profs(self):
        """ 
        Save profiles to file. 
        
        For each PDF, e.g., 'entropy' and 'w', the file will have a dataset with:
            1. pos_mask (the mask for the positive profile)
            2. neg_mask (the mask for the negative profile)
            3. the basis (e.g., 'z')
        """
        if self.reader.comm.rank == 0:
            with h5py.File('{:s}/asymmetry_data.h5'.format(self.out_dir), 'w') as f:
                for k, data in self.profs.items():
                    if k == self.basis_k:
                        f[k] = data
                    else:
                        pos_mask, neg_mask = data
                        this_group = f.create_group(k)
                        for d, n in ((pos_mask, 'pos_mask'), (neg_mask, 'neg_mask')):
                            dset = this_group.create_dataset(name=n, shape=d.shape, dtype=np.float64)
                            f['{:s}/{:s}'.format(k, n)][:] = d

