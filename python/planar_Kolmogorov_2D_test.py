"""
Dedalus script for 2D strongfield MHD Kolmogorov flow

This script usesa Fourier bases in both the vertical and horozontal directions with fully
periodic boundary condtions. 

Usage:
    planar_Kolmogorov_2D_test.py [options]
    planar_Kolmogorov_2D_test.py <config> [options]



Options:
    --Lx_factor=<Lx>                Box size in x direction [default: 2.0]
    --Lz_factor=<Lz>                Box size in z direction [default: 4.0]
    --Nx=<Nx>                       Horizontal resolution [default: 128]
    --Nz=<Nz>                       Vertical resolution [default: 256]

    --HB=<HB>                       Magnetic field strength [default: 0.4]
    --Pm=<MagneticPrandtl>          Magnetic prandtl number [default: 0.1]
    --Re=<Reynolds>                 Reynolds number [default: 20.0]

    --root_dir=<dir>                Root directory for output [default: ./]
    --label=<label>                 Add label to directory name





Need to add stop time options



"""

import logging
import os
import sys
import time
from configparser import ConfigParser
from pathlib import Path

import numpy as np
from docopt import docopt
from mpi4py import MPI

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools  import post
from dedalus.tools.config import config


from logic.output import initialize_output #need to change output to what I want to output
from logic.checkpointing import Checkpoint
from logic.extras import filter_field
from logic.parsing import  construct_out_dir

logger = logging.getLogger(__name__)

args   = docopt(__doc__)
if args['<config>'] is not None:
    config_file = Path(args['<config>'])
    config = ConfigParser()
    config.read(str(config_file))
    for n, v in config.items('parameters'):
        for k in args.keys():
            if k.split('--')[-1].lower() == n:
                if v == 'true': v = True
                args[k] = v
#Read in command line args, set up data directory

resolution_flags=['Nx','Nz']
data_dir = construct_out_dir(args, base_flags=['HB','Pm','Re','Lx_factor','Lz_factor'],label_flags=[], resolution_flags=resolution_flags, parent_dir_flag='root_dir')
logger.info("saving run in: {}".format(data_dir))


# Simulation Parameters
Lx=float(args['--Lx_factor'])*np.pi
Lz=float(args['--Lz_factor'])*np.pi
Nx=int(args['--Nx'])
Nz=int(args['--Nz'])
HB_star=float(args['--HB'])
Pm=float(args['--Pm'])
Re=float(args['--Re'])
Re_m = Re*Pm
aspect=Lx
init_dt = 0.01 * Lx / (Nx)  # I have not thought about what init_dt should be, this was a guess


logger.info("HB = {:2g}, Pm = {:2g}, Re = {:2g} , boxsize={}x{}, resolution = {}x{}".format(HB_star, Pm, Re, Lx, Lz,  Nx, Nz))

# simulation stop conditions
stop_sim_time = np.inf  # stop time in simulation time units
stop_wall_time = np.inf  # stop time in terms of wall clock
stop_iteration = 1000000  # stop time in terms of iteration count


#def filter_field(field, frac=0.5):
#    """
#    Taken from Dedalus example notebook on Taylor-Couette flow. This is meant to filter out small-scale noise in
#    the initial condition, which can cause problems.
#    """
#    dom = field.domain
#    local_slice = dom.dist.coeff_layout.slices(scales=dom.dealias)
#    coeff = []
#    for i in range(dom.dim)[::-1]:
#        coeff.append(np.linspace(0, 1, dom.global_coeff_shape[i], endpoint=False))
#    cc = np.meshgrid(*coeff)
#
#    field_filter = np.zeros(dom.local_coeff_shape, dtype='bool')
#    for i in range(dom.dim):
#        field_filter = field_filter | (cc[i][local_slice] > frac)
#    field['c'][field_filter] = 0j


# Create bases and domain
start_init_time = time.time()  # start a timer to see how long things take
x_basis = de.Fourier('x', Nx, interval=(0, Lx), dealias=3/2)
z_basis = de.Fourier('z', Nz, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)# , mesh=(5, 8))

problem = de.IVP(domain, variables=['phi', 'psi'], time='t')
# Phi is streamfunction, psi is flux function

# problem.parameters['MA2inv'] = 1.0/(MA**2.0)
problem.parameters['Lz'] = Lz
problem.parameters['Lx'] = Lx
problem.parameters['HB_star'] = HB_star
problem.parameters['Reinv'] = 1.0/Re
problem.parameters['Rminv'] = 1.0/Re_m
problem.parameters['MA'] = 1.0/np.sqrt(HB_star)
problem.parameters['aspect']=aspect

problem.substitutions['zeta'] = "dx(dx(phi)) + dz(dz(phi))"  # vorticity
problem.substitutions['J'] = "dx(dx(psi)) + dz(dz(psi))"  # current density
problem.substitutions['u'] = "dz(phi)"
problem.substitutions['w'] = "-dx(phi)"
problem.substitutions['Bx'] = "dz(psi)"
problem.substitutions['Bx_tot'] = "1 + Bx"
problem.substitutions['Bz'] = "-dx(psi)"
problem.substitutions['vel_rms']     = 'sqrt(u**2 +  w**2)'
problem.substitutions['b_mag']        = 'sqrt(Bx**2  + Bz**2)'
problem.substitutions['b_mag_tot']        = 'sqrt(Bx_tot**2  + Bz**2)'
problem.substitutions['plane_avg(A)'] = 'integ(A, "x")/Lx'
problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Lz'
problem.substitutions['output_Re'] = "vel_rms*Lx"
problem.substitutions['output_MA'] = "1.0/b_mag"#is this right?
problem.substitutions['re_stress_tot']="dx(phi)*dz(phi)"
problem.substitutions['max_stress_tot']="dx(psi)*dz(psi)"
# I don't understand why there's a 1/Re coefficient on the forcing term in the uncommented line. Need to double-check.
# Presumably I made a silly mistake.
# problem.add_equation("dt(zeta) - Reinv*(dx(dx(zeta)) + dz(dz(zeta))) = -dx(zeta) * dz(phi) + dx(phi) * dz(zeta) + HB_star * (dx(J) * dz(psi) - dx(psi) * dz(J)) - cos(x)", condition="(nx!=0) or (nz!=0)")
problem.add_equation("dt(zeta) - Reinv * (dx(dx(zeta)) + dz(dz(zeta))) = -dx(zeta) * dz(phi) + dx(phi) * dz(zeta) + HB_star * (dx(J) * (dz(psi) + 1) - dx(psi) * dz(J)) +  cos(z)", condition="(nx!=0) or (nz!=0)")
problem.add_equation("phi = 0", condition="(nx==0) and (nz==0)")
problem.add_equation("dt(psi) - Rminv*J = dx(phi)*(dz(psi) + 1) - dx(psi)*dz(phi)")

# Build solver
# solver = problem.build_solver(de.timesteppers.SBDF3)  # same timestepper as PADDIM
solver = problem.build_solver(de.timesteppers.RK443)
logger.info('Solver built')

# Initial conditions
x = domain.grid(0)
z = domain.grid(1)
phi = solver.state['phi']
psi = solver.state['psi']
# for f in [phi, psi]:
    # f.set_scales(domain.dealias, keep_data=False)
# w['g'] = np.sin(x)

# Noise initial conditions
# Random perturbations, initialized globally for same results in parallel
pert = 1e-5
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=42)
noise = rand.standard_normal(gshape)[slices]
phi['g'] = pert * noise
filter_field(phi)
phi['g'] += np.cos(z)

# Integration parameters
solver.stop_sim_time = stop_sim_time
solver.stop_wall_time = stop_wall_time
solver.stop_iteration = stop_iteration

# Analysis
snap = solver.evaluator.add_file_handler('snapshots', iter=500, max_writes=100000000)
snap.add_system(solver.state)
analysis_tasks = initialize_output(solver, data_dir, Lx, Lz, plot_boundaries=False, threeD=False, mode="overwrite", slice_output_dt=0.25, output_dt=0.11, out_iter=100) #need to change this when i make output file

# CFL
# Not totally sure what the best settings here are. Maybe need to experiment a bit.
CFL = flow_tools.CFL(solver, initial_dt=init_dt, cadence=1, safety=0.4,
                     max_change=1.5, max_dt=2e-1, threshold=0.1)
CFL.add_velocities(('u', 'w'))
CFL2 = flow_tools.CFL(solver, initial_dt=init_dt, cadence=1, safety=0.4,
                     max_change=1.5, min_change=2e-1, max_dt=2e-1, threshold=0.1)#maybe need to add max dt and safety as
                                                                                #input variables if timestepping is an issue
CFL2.add_velocities(('Bx/MA', 'Bz/MA'))

#Set up flow tracking for terminal output
flow = flow_tools.GlobalFlowProperty(solver, cadence=1)

flow.add_property("dx(Bx) + dz(Bz)", name='divB')
flow.add_property("b_mag/MA", name="b_mag")
flow.add_property("output_Re", name="output_Re")


# Main loop
end_init_time = time.time()
logger.info('Initialization time: %f' %(end_init_time-start_init_time))
try:
    Re_avg=0
    logger.info('Starting loop')
    start_iter = solver.iteration
    start_run_time = time.time()
    init_time = last_time = solver.sim_time
    while (solver.ok and np.isfinite(Re_avg)):
        dt1 = CFL.compute_dt()
        dt2 = CFL2.compute_dt()

        dt=np.min([dt1, dt2])
        solver.step(dt)
        effective_iter = solver.iteration - start_iter
        effective_sim_time = solver.sim_time - init_time
        if effective_iter % 10 == 0:
            Re_avg = flow.grid_average('output_Re')
            log_string =  'Iteration: {:5d}, '.format(solver.iteration)
            log_string +=  'Time: {:8.3e}, '.format(effective_sim_time)
            log_string +=  'dt: {:8.3e}, '.format(dt)
            log_string += 'divB: {:8.3e}, '.format(flow.grid_average('divB'))
            log_string += 'dt_ratio: {:8.3e}, '.format(dt/dt2)
            log_string += 'B_rms: {:8.3e}, '.format(flow.volume_average('b_mag'))
            log_string += 'Re: {:8.3e}/{:8.3e}, '.format(Re_avg, flow.max('output_Re'))
            logger.info(log_string)

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_run_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_run_time-start_run_time))
    logger.info('Run time: %f cpu-hr' %((end_run_time-start_run_time)/60/60*domain.dist.comm_cart.size))

