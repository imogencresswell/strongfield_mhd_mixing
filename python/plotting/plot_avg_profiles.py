"""
Script for plotting colormaps of the evolution of 1D profiles of a dedalus simulation.  
This script plots time evolution of the fields specified in 'fig_type'

Usage:
    plot_avg_profiles.py <root_dir> [options]

Options:
    --fig_name=<fig_name>               Name of figure output directory & base name of saved figures [default: avg_profs]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]
    --avg_writes=<n_writes>             Number of output writes to average over [default: 200]

    --col_inch=<in>                     Number of inches / column [default: 6]
    --row_inch=<in>                     Number of inches / row [default: 3]

    --fig_type=<fig_type>               Type of figure to plot
                                            1 - T, enth_flux, kappa_flux, tot_flux
                                        [default: 1]
"""
from docopt import docopt
args = docopt(__doc__)
from plot_logic.profiles import ProfilePlotter
import logging
logger = logging.getLogger(__name__)


n_files     = args['--n_files']
if n_files is not None: n_files = int(n_files)
start_file  = int(args['--start_file'])
avg_writes = int(args['--avg_writes'])

root_dir    = args['<root_dir>']
if root_dir is None:
    logger.error('No dedalus output dir specified, exiting')
    import sys
    sys.exit()
fig_name   = args['--fig_name']

plotter = ProfilePlotter(root_dir, file_dir='profiles', fig_name=fig_name, start_file=start_file, n_files=n_files)

if int(args['--fig_type']) == 1:
    plotter.add_profile('u', avg_writes)
    plotter.add_profile('re_stress', avg_writes)
    plotter.add_profile('max_stress', avg_writes)

plotter_kwargs = { 'col_in' : int(args['--col_inch']), 'row_in' : int(args['--row_inch']) }
plotter.plot_avg_profiles(dpi=int(args['--dpi']), **plotter_kwargs)
