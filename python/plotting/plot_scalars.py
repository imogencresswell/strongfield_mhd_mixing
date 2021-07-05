"""
Script for plotting traces of evaluated scalar quantities vs. time.

Usage:
    plot_scalars.py <root_dir> [options]

Options:
    --fig_name=<fig_name>               Output directory for figures [default: traces]
    --start_file=<start_file>           Dedalus output file to start at [default: 1]
    --n_files=<num_files>               Number of files to plot [default: 100000]
    --dpi=<dpi>                         Image pixel density [default: 150]
"""
import logging
logger = logging.getLogger(__name__)
from docopt import docopt
args = docopt(__doc__)
from plot_logic.scalars import ScalarFigure, ScalarPlotter

root_dir = args['<root_dir>']
fig_name  = args['--fig_name']
start_file = int(args['--start_file'])
n_files     = args['--n_files']
if n_files is not None: n_files = int(n_files)


# Nu vs time
fig1 = ScalarFigure(3, 1, col_in=6, row_in=2.5, fig_name='vel_trace')
fig1.add_field(0, 'u')
fig1.add_field(1, 'w')
fig1.add_field(2, 'u')
fig1.add_field(2, 'w')

# Re vs. time
fig2 = ScalarFigure(4, 1, col_in=6, row_in=2.5, fig_name='mag_trace')
fig2.add_field(0, 'Bx')
fig2.add_field(1, 'Bz')
fig2.add_field(2, 'b_mag')
fig2.add_field(3, 'Bx')
fig2.add_field(3, 'Bz')
fig2.add_field(3, 'b_mag')

# Energies
fig3 = ScalarFigure(3, 1, col_in=6, row_in=2.5, fig_name='energies')
fig3.add_field(0, 'KE')
fig3.add_field(1, 'ME_tot')
fig3.add_field(2, 'KE')
fig3.add_field(2, 'ME_tot')


# Energies
fig4 = ScalarFigure(1, 1, col_in=6, fig_name='vorticity')
fig4.add_field(0, 'zeta')

# Fluxes
fig5 = ScalarFigure(1, 1, col_in=6, fig_name='current density')
fig5.add_field(0, 'J')

# KE sources
fig6 = ScalarFigure(1, 1, col_in=6,  fig_name='flux function')
fig6.add_field(0, 'psi')


# KE sources
fig7 = ScalarFigure(1, 1, col_in=6,  fig_name='streamfunction')
fig7.add_field(0, 'phi')



# Load in figures and make plots
plotter = ScalarPlotter(root_dir, file_dir='scalar', fig_name=fig_name, start_file=start_file, n_files=n_files)
figs = [fig1, fig2, fig3, fig4, fig5, fig6, fig7]
plotter.load_figures(figs)
plotter.plot_figures(dpi=int(args['--dpi']))
plotter.plot_convergence_figures(dpi=int(args['--dpi']))

