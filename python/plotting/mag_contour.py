"""
A shell of a script for making nice DFD talk slices

Usage:
    plot_talk_slices.py <root_dir> [options]

Options:
    --fig_name=<fig_name>               Name of figure output directory & base name of saved figures [default: mag_bxbz]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]

    --col_inch=<in>                     Number of inches / column [default: 6]
    --row_inch=<in>                     Number of inches / row [default: 3]
"""
import numpy as np
import matplotlib.pyplot as plt
from docopt import docopt
args = docopt(__doc__)
from plotpal.file_reader import SingleFiletypePlotter as SFP
from plotpal.plot_grid   import ColorbarPlotGrid as CPG
import matplotlib.colors as colors
# Read in master output directory
root_dir    = args['<root_dir>']
data_dir    = 'slices'
if root_dir is None:
    print('No dedalus output dir specified, exiting')
    import sys
    sys.exit()

start_file  = int(args['--start_file'])
#avg_writes  = int(args['--avg_writes'])
fig_name    = args['--fig_name']
n_files     = args['--n_files']
if n_files is not None: 
    n_files = int(n_files)

plotter = SFP(root_dir, file_dir=data_dir, fig_name=fig_name, start_file=start_file, n_files=n_files, distribution='even')

field_keys = ['u','b_mag_tot', 'psi', 'Bx','Bz', 'w']
bases_keys  = ['x', 'z']

plot_grid = CPG(1, 1, col_in=float(args['--col_inch']), row_in=float(args['--row_inch'])) 
axs  = plot_grid.axes
caxs = plot_grid.cbar_axes
ax1  = axs['ax_0-0']
cax1 = caxs['ax_0-0']
#ax2 = axs['ax_0-1']
#cax2 = axs['ax_0-1']

with plotter.my_sync:
    if not plotter.idle:
        while plotter.files_remain(bases_keys, field_keys):
            bases, tasks, write_num, sim_time = plotter.read_next_file()

            x = bases['x']
            z = bases['z']

            zz, xx = np.meshgrid(z.flatten(), x.flatten())

            for i, num in enumerate(write_num):
                print('writing {}/{}'.format(num, len(write_num)))
                b_mag = tasks['b_mag_tot'][i,:]
                u = tasks['u'][i,:]
                w = tasks['w'][i,:]
                psi = tasks['psi'][i,:]
                Bx = tasks['Bx'][i,:]
                Bz = tasks['Bz'][i,:]
                skip=(slice(None,None,15),slice(None,None,15))
                vel_rms=np.sqrt(u**2 + w**2)
                tot_Bx=np.array(Bx) + 1.0
                #p = ax1.pcolormesh(xx, zz, psi, cmap='viridis')
                #plt.colorbar(p, cax1, orientation='horizontal')
                #f=ax1.quiver(xx[skip],zz[skip],u[skip],w[skip], color='white',  units='xy', scale = 70)
                #cax1.text(0.5, 0.5, 'Streamfunction', ha='center', va='center', transform=cax1.transAxes)

                g = ax1.pcolormesh(xx, zz, b_mag, cmap='inferno')
                plt.colorbar(g, cax1, orientation='horizontal')
                h=ax1.quiver(xx[skip],zz[skip],tot_Bx[skip],Bz[skip], color='white',  units='xy', scale = 10)
                cax1.text(0.5, 0.5, 'Magnetic Field', ha='center', va='center', transform=cax1.transAxes)

                plt.savefig('{:s}/{:s}_{:04d}.png'.format(plotter.out_dir, fig_name, num), dpi=float(args['--dpi']))
                for ax in [ax1, cax1,]:
                    ax.cla()
