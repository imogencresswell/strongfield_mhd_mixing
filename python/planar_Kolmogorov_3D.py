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
    --Ly_factor=<Lz>                Box size in z direction [default: 4.0]
    --Nx=<Nx>                       Horizontal x resolution [default: 128]
    --Nz=<Nz>                       Vertical resolution [default: 256]
    --Ny=<Ny>                       Horizontal y resolution [default: 256]
    --HB=<HB>                       Magnetic field strength [default: 0.4]
    --Pm=<MagneticPrandtl>          Magnetic prandtl number [default: 0.1]
    --Re=<Reynolds>                 Reynolds number [default: 20.0]
    --mesh=<mesh>                   Processor mesh if distributing 3D run in 2D 
    --root_dir=<dir>                Root directory for output [default: ./]
    --label=<label>                 Add label to directory name
    --safety=<s>               CFL safety factor [default: 0.7]
    --mag_safety=<ms>          Magnetic CFL safety factor [default: 0.7]
    --SBDF2              Uses SBDF2 timestepper
    --SBDF4              Uses SBDF4 timestepper
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

resolution_flags=['Nx','Ny','Nz']
data_dir = construct_out_dir(args, base_flags=['HB','Pm','Re','Lx_factor','Ly_factor','Lz_factor'],label_flags=[], resolution_flags=resolution_flags, parent_dir_flag='root_dir')
logger.info("saving run in: {}".format(data_dir))


# Simulation Parameters
Lx=float(args['--Lx_factor'])*np.pi
Ly=float(args['--Ly_factor'])*np.pi
Lz=float(args['--Lz_factor'])*np.pi
Nx=int(args['--Nx'])
Ny=int(args['--Ny'])
Nz=int(args['--Nz'])
HB_star=float(args['--HB'])
Pm=float(args['--Pm'])
Re=float(args['--Re'])
Re_m = Re*Pm
aspect=Lx
init_dt = 0.01 * Lx / (Nx)  # I have not thought about what init_dt should be, this was a guess
mesh = args['--mesh']
ncpu = MPI.COMM_WORLD.size
if mesh is not None:
    mesh = mesh.split(',')
    mesh = [int(mesh[0]), int(mesh[1])]
    else:
        log2 = np.log2(ncpu)
        if log2 == int(log2):
            mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
        logger.info("running on processor mesh={}".format(mesh))


logger.info("HB = {:2g}, Pm = {:2g}, Re = {:2g} , boxsize={}x{}x{}, resolution = {}x{}x{}".format(HB_star, Pm, Re, Lx, Ly, Lz,  Nx, Ny, Nz))
# simulation stop conditions
stop_sim_time = np.inf  # stop time in simulation time units
stop_wall_time = 2.5*60.*60.  # stop time in terms of wall clock
stop_iteration = 32000  # stop time in terms of iteration count



# Create bases and domain
start_init_time = time.time()  # start a timer to see how long things take
x_basis = de.Fourier('x', Nx, interval=(0, Lx), dealias=3/2)
y_basis = de.Fourier('y', Ny, interval=(0, Ly), dealias=3/2)
z_basis = de.Fourier('z', Nz, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64, mesh=(mesh1, mesh2))

# 3D MHD in terms of magnetic vector potential in Coulomb gauge (div(A) = 0)
# Drawing heavily from advice from Ben Brown/examples he and the other Dedalus developers shared on github,
# particularly https://github.com/DedalusProject/dedalus_scaling/blob/master/RB_mhd_3d.py
problem = de.IVP(domain, variables=['u', 'v', 'w', 'p', 'Ax', 'Ay', 'Az', 'phi'], time='t')
problem.parameters['MA2inv'] = HB_star  # 99% sure that this is just H_B^*
problem.parameters['MA'] = np.sqrt(1/HB_star)
problem.parameters['Reinv'] = 1.0/Reynolds
problem.parameters['Rminv'] = 1.0/mReynolds
problem.parameters['Lx'] = Lx
problem.parameters['Ly'] = Ly
problem.parameters['Lz'] = Lz

problem.substitutions['Bx'] = "(dy(Az) - dz(Ay))"
problem.substitutions['By'] = "(dz(Ax) - dx(Az))"
# problem.substitutions['Bz'] = "(1.0 + dx(Ay) - dy(Ax))"
problem.substitutions['Bz'] = "(dx(Ay) - dy(Ax))"
problem.substitutions['Bz_tot'] = "(1.0 + Bz)"
problem.substitutions['Jx'] = "(dy(Bz) - dz(By))"
problem.substitutions['Jy'] = "(dz(Bx) - dx(Bz))"
problem.substitutions['Jz'] = "(dx(By) - dy(Bx))"
# Here, Ox, Oy, Oz are the x, y, z components of vorticity Omega = curl(u)
problem.substitutions['Ox'] = "(dy(w) - dz(v))"
problem.substitutions['Oy'] = "(dz(u) - dx(w))"
problem.substitutions['Oz'] = "(dx(v) - dy(u))"
problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Ly/Lz'
problem.substitutions["Kx"] = "dy(Oz)-dz(Oy)"
problem.substitutions["Ky"] = "dz(Ox)-dx(Oz)"
problem.substitutions["Kz"] = "dx(Oy)-dy(Ox)"



# Note the pressure term in this formulation is really p + u^2/2
problem.add_equation("dt(u) + Reinv*Kx + dx(p)  = v*Oz - w*Oy + MA2inv*(Jy*Bz - Jz*By) + sin(z) ")
problem.add_equation("dt(v) + Reinv*Ky + dy(p) + MA2inv*Jz = w*Ox - u*Oz + MA2inv*(Jz*Bx - Jx*Bz)")
problem.add_equation("dt(w) + Reinv*Kz + dz(p) - MA2inv*Jy = u*Oy - v*Ox + MA2inv*(Jx*By - Jy*Bx) ")



#problem.add_equation("dt(u) + Reinv*Kx + dx(p) - MA2inv*Jy = v*Oz - w*Oy + MA2inv*(Jy*Bz - Jz*By)")
#problem.add_equation("dt(v) + Reinv*Ky + dy(p) + MA2inv*Jx = w*Ox - u*Oz + MA2inv*(Jz*Bx - Jx*Bz)")
#problem.add_equation("dt(w) + Reinv*Kz + dz(p) = u*Oy - v*Ox + MA2inv*(Jx*By - Jy*Bx) + sin(x)")
# What's commented out here: old code where the momentum advection term was v dot grad v, as opposed to what's above
#problem.add_equation("dt(u) - Reinv*(dx(dx(u)) + dy(dy(u)) + dz(dz(u))) + dx(p) - MA2inv*Jy = - u*dx(u) - v*dy(u) - w*dz(u) + MA2inv*(Jy*Bz - Jz*By)")
#problem.add_equation("dt(v) - Reinv*(dx(dx(v)) + dy(dy(v)) + dz(dz(v))) + dy(p) + MA2inv*Jx = - u*dx(v) - v*dy(v) - w*dz(u) + MA2inv*(Jz*Bx - Jx*Bz)")
#problem.add_equation("dt(w) - Reinv*(dx(dx(w)) + dy(dy(w)) + dz(dz(w))) + dz(p) = - u*dx(w) - v*dy(w) - w*dz(w) + MA2inv*(Jx*By - Jy*Bx) + sin(x)")

# Induction equations but for A. Note that if div(A) = 0 then curl(curl(A)) = -Laplacian(A)
# problem.add_equation("dt(Ax) - Rminv*(dx(dx(Ax)) + dy(dy(Ax)) + dz(dz(Ax))) + dx(phi) - v = v*Bz - w*By")
# problem.add_equation("dt(Ay) - Rminv*(dx(dx(Ay)) + dy(dy(Ay)) + dz(dz(Ay))) + dy(phi) + u = w*Bx - u*Bz")
# problem.add_equation("dt(Az) - Rminv*(dx(dx(Az)) + dy(dy(Az)) + dz(dz(Az))) + dz(phi) = u*By - v*Bx")
problem.add_equation("dt(Ax) + Rminv*Jx + dx(phi) - v = v*Bz - w*By")
problem.add_equation("dt(Ay) + Rminv*Jy + dy(phi) + u = w*Bx - u*Bz")
problem.add_equation("dt(Az) + Rminv*Jz + dz(phi) = u*By - v*Bx")

# div(u) = 0 and div(A) = 0 reduce to 0 = 0 for the kx=ky=kz Fourier mode. Thus, closing the system at that k requires
# specifying p and phi instead
problem.add_equation("dx(u) + dy(v) + dz(w) = 0", condition="(nx!=0) or (ny!=0) or (nz!=0)")
problem.add_equation("p=0", condition="(nx==0) and (ny==0) and (nz==0)")
problem.add_equation("dx(Ax) + dy(Ay) + dz(Az) = 0", condition="(nx!=0) or (ny!=0) or (nz!=0)")
problem.add_equation("phi=0", condition="(nx==0) and (ny==0) and (nz==0)")

# Build solver
if args['--SBDF2']:
    ts = de.timesteppers.SBDF2
if args['--SBDF4']:
    ts = de.timesteppers.SBDF4
else:
    ts = de.timesteppers.RK443
solver = problem.build_solver(ts)
logger.info('Solver built')

# Initial conditions
x = domain.grid(0)
y = domain.grid(1)
z = domain.grid(2)
psi = domain.new_field(name='psi')
u = solver.state['u']
v = solver.state['v']
w = solver.state['w']
#for f in [psi, u, v, w]:
    #f.set_scales(domain.dealias, keep_data=False)


# Noise initial conditions
# Random perturbations, initialized globally for same results in parallel
pert = 1e-4
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=42)
noise = rand.standard_normal(gshape)[slices]
psi['g'] = pert * noise * np.ones_like(z)
filter_field(psi)
psi.set_scales(1/16, keep_data=True)
psi['c']
psi['g']
psi.set_scales(1, keep_data=True)
psi.differentiate('z', out=u)
psi['g'] = -1.0*np.copy(psi['g'])
psi.differentiate('x', out=w)
w.set_scales(1, keep_data=True)
w['g'] = w['g'] + np.sin(x)

# Integration parameters
solver.stop_sim_time = stop_sim_time
solver.stop_wall_time = stop_wall_time
solver.stop_iteration = stop_iteration

# Analysis
dumps = solver.evaluator.add_file_handler('dumps', iter=5000, max_writes=20)
dumps.add_system(solver.state)

snap = solver.evaluator.add_file_handler('snapshots', iter=100, max_writes=1000)
scalar = solver.evaluator.add_file_handler('scalar', iter=10, max_writes=10000)
for task_name in ["u", "v", "w", "Bx", "By", "Bz_tot"]:
    snap.add_task("interp(" + task_name + ", x=0)", scales=1, name=task_name + " side")
    snap.add_task("interp(" + task_name + ", y=0)", scales=1, name=task_name + " front")
    snap.add_task("interp(" + task_name + ", z=0)", scales=1, name=task_name + " bottom")
    snap.add_task("integ(" + task_name + ", 'y', 'z')", scales=1, name=task_name + "2Davg")
    snap.add_task("integ(" + task_name + ", 'z')", scales=1, name=task_name + "1Davg")

    scalar.add_task("vol_avg(" + task_name + "**2)", name=task_name + " squared")
    

    
# CFL
cfl_safety = float(args['--safety'])
mag_cfl_safety = float(args['--mag_safety'])


CFL = flow_tools.CFL(solver, initial_dt=init_dt, cadence=1, safety=cfl_safety, )
                     max_change=1.5, min_change=0.5, max_dt=dt_max, threshold=0.1)
CFL.add_velocities(('u', 'v', 'w'))
CFL2 = flow_tools.CFL(solver, initial_dt=init_dt, cadence=1, mag_cfl_safety,)
                     max_change=1.5, min_change=0.5, max_dt=dt_max, threshold=0.1)
                                                                                
CFL2.add_velocities(('Bx/MA', 'By/MA', 'Bz/MA'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("sqrt(u*u + v*v + w*w)", name='u_abs')

# Main loop
end_init_time = time.time()
logger.info('Initialization time: %f' %(end_init_time-start_init_time))
try:
    logger.info('Starting loop')
    start_run_time = time.time()
    while solver.ok:
        dt1 = CFL.compute_dt()
        dt2 = CFL2.compute_dt()

        dt = np.min([dt1, dt2])
        solver.step(dt)
        if (solver.iteration-1) % 10 == 0: #was 10
            logger.info('Iteration: %i, sim_time: %e, dt: %e, wall_time: %.2f sec' %(solver.iteration, solver.sim_time, dt, time.time()-start_run_time))
            logger.info('dt/dt2 = %f' %(dt/dt2))
            logger.info('Max u_abs = %f' %flow.max('u_abs'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_run_time = time.time()
    logger.info('Iterations: %i' % solver.iteration)
    logger.info('Sim end time: %f' % solver.sim_time)
    logger.info('Run time: %.2f sec' % (end_run_time-start_run_time))
    logger.info('Run time: %f cpu-hr' % ((end_run_time-start_run_time)/60/60*domain.dist.comm_cart.size))

