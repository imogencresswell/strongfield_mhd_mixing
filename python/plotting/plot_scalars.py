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
fig1 = ScalarFigure(1, 1, col_in=6, fig_name='nu_trace')
fig1.add_field(0, 'Nu')

# Re vs. time
fig2 = ScalarFigure(1, 1, col_in=6, fig_name='pe_trace')
fig2.add_field(0, 'Pe')

# dT 
fig3 = ScalarFigure(1, 1, col_in=6, fig_name='delta_T')
fig3.add_field(0, 'delta_T')

# Energies
fig4 = ScalarFigure(4, 1, col_in=6, row_in=2.5, fig_name='energies')
fig4.add_field(0, 'KE')
fig4.add_field(1, 'KE')
fig4.add_field(0, 'IE')
fig4.add_field(2, 'IE')
fig4.add_field(0, 'TE')
fig4.add_field(3, 'TE')

# Energies
fig5 = ScalarFigure(2, 1, col_in=6, row_in=2.5, fig_name='boundary_temps')
fig5.add_field(0, 'left_T')
fig5.add_field(1, 'right_T')

# Fluxes
fig6 = ScalarFigure(2, 1, col_in=6, row_in=2.5, fig_name='boundary_fluxes')
fig6.add_field(0, 'left_flux')
fig6.add_field(1, 'right_flux')

# KE sources
fig7 = ScalarFigure(3, 1, col_in=6, row_in=2.5, fig_name='ke_sources')
fig7.add_field(0, 'KE')
fig7.add_field(1, 'enstrophy')
fig7.add_field(2, 'wT')

# KE sources
fig8 = ScalarFigure(3, 1, col_in=6, row_in=2.5, fig_name='ke_sources2')
fig8.add_field(0, 'KE')
fig8.add_field(1, 'visc_KE_source')
fig8.add_field(2, 'buoy_KE_source')


# boundary_currents 
fig9 = ScalarFigure(2, 1, col_in=6, row_in=2.5, fig_name='boundary_currents')
fig9.add_field(0, 'left_Jx')
fig9.add_field(0, 'right_Jx')
fig9.add_field(1, 'left_Jy')
fig9.add_field(1, 'right_Jy')

# boundary horizontal B 
fig10 = ScalarFigure(2, 1, col_in=6, row_in=2.5, fig_name='boundary_b_horiz')
fig10.add_field(0, 'left_Bx')
fig10.add_field(0, 'right_Bx')
fig10.add_field(1, 'left_By')
fig10.add_field(1, 'right_By')

# Re vs. time
if 'rotating' in root_dir:
    figRo = ScalarFigure(1, 1, col_in=6, fig_name='ro_trace')
    figRo.add_field(0, 'Ro')
    figRo.add_field(0, 'true_Ro')

if 'mhd' in root_dir:
    figb_mag = ScalarFigure(1, 1, col_in=6, fig_name='b_mag_trace')
    figb_mag.add_field(0, 'b_mag')
    figb_perp = ScalarFigure(1, 1, col_in=6, fig_name='b_perp_trace')
    figb_perp.add_field(0, 'b_perp')
    figb_comp = ScalarFigure(4, 1, col_in=6, fig_name='b_comp_trace')
    figb_comp.add_field(0, 'Bx_rms')
    figb_comp.add_field(0, 'By_rms')
    figb_comp.add_field(0, 'Bz_rms')
    figb_comp.add_field(1, 'Bx_rms')
    figb_comp.add_field(2, 'By_rms')
    figb_comp.add_field(3, 'Bz_rms')
    fig_forces = ScalarFigure(6, 1, col_in=6, fig_name='forces_trace')
    fig_forces.add_field(0, 'f_v_mag')
    fig_forces.add_field(0, 'f_i_mag')
    fig_forces.add_field(0, 'f_mn_mag')
    fig_forces.add_field(0, 'f_ml_mag')
    fig_forces.add_field(0, 'f_b_mag')
    fig_forces.add_field(1, 'f_v_mag')
    fig_forces.add_field(2, 'f_i_mag')
    fig_forces.add_field(3, 'f_ml_mag')
    fig_forces.add_field(4, 'f_mn_mag')
    fig_forces.add_field(5, 'f_b_mag')

if 'bootstrap' in root_dir:
    fig_forces2 = ScalarFigure(6, 1, col_in=6, fig_name='sol_forces_trace')
    fig_forces2.add_field(0, 's_v_mag')
    fig_forces2.add_field(0, 's_i_mag')
    fig_forces2.add_field(0, 's_mn_mag')
    fig_forces2.add_field(0, 's_ml_mag')
    fig_forces2.add_field(0, 's_b_mag')
    fig_forces2.add_field(1, 's_v_mag')
    fig_forces2.add_field(2, 's_i_mag')
    fig_forces2.add_field(3, 's_ml_mag')
    fig_forces2.add_field(4, 's_mn_mag')
    fig_forces2.add_field(5, 's_b_mag')

    fig_pressures = ScalarFigure(6, 1, col_in=6, fig_name='pressures_trace')
    fig_pressures.add_field(0, 'p_v')
    fig_pressures.add_field(0, 'p_i')
    fig_pressures.add_field(0, 'p_mn')
    fig_pressures.add_field(0, 'p_ml')
    fig_pressures.add_field(0, 'p_b')
    fig_pressures.add_field(1, 'p_v')
    fig_pressures.add_field(2, 'p_i')
    fig_pressures.add_field(3, 'p_ml')
    fig_pressures.add_field(4, 'p_mn')
    fig_pressures.add_field(5, 'p_b')

    fig_p = ScalarFigure(1, 1, col_in=6, fig_name='p_goodness')
    fig_p.add_field(0, 'p_goodness')

    fig_inputs = ScalarFigure(2, 1, col_in=6, fig_name='inputs_trace')
    fig_inputs.add_field(0, 'Ra')
    fig_inputs.add_field(0, 'Q')



# Load in figures and make plots
plotter = ScalarPlotter(root_dir, file_dir='scalar', fig_name=fig_name, start_file=start_file, n_files=n_files)
figs = [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10]
if 'rotating' in root_dir: figs.append(figRo)
if 'mhd' in root_dir: figs+=[figb_mag , figb_perp, figb_comp, fig_forces]
if 'bootstrap' in root_dir: figs += [fig_p, fig_forces2, fig_pressures, fig_inputs]
plotter.load_figures(figs)
plotter.plot_figures(dpi=int(args['--dpi']))
plotter.plot_convergence_figures(dpi=int(args['--dpi']))

