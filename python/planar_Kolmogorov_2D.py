import numpy as np
import numpy.ma as ma
from mpi4py import MPI
import time
from dedalus import public as de
from dedalus.extras import flow_tools
import logging
import traceback
logger = logging.getLogger(__name__)

# Parameters
Lx, Lz = (2.0*np.pi, 20.0*np.pi)  # was lz=8pi
Nx, Nz = (128, 1280)  # was nz=512
HB_star = 1.0  # /(40.0**2.0)  # 1.0
# MA = 40.0
Pm = 0.1
Reynolds = 20.0
mReynolds = Reynolds*Pm
init_dt = 0.01 * Lx / (Nx)

# simulation stop conditions
stop_sim_time = np.inf  # stop time in simulation time units
stop_wall_time = np.inf  # stop time in terms of wall clock
stop_iteration = 100000  # stop time in terms of iteration count


def filter_field(field, frac=0.5):
    """
    Taken from Dedalus example notebook on Taylor-Couette flow. This is meant to filter out small-scale noise in
    the initial condition, which can cause problems.
    :param field:
    :param frac:
    :return:
    """
    dom = field.domain
    local_slice = dom.dist.coeff_layout.slices(scales=dom.dealias)
    coeff = []
    for i in range(dom.dim)[::-1]:
        coeff.append(np.linspace(0, 1, dom.global_coeff_shape[i], endpoint=False))
    cc = np.meshgrid(*coeff)

    field_filter = np.zeros(dom.local_coeff_shape, dtype='bool')
    for i in range(dom.dim):
        field_filter = field_filter | (cc[i][local_slice] > frac)
    field['c'][field_filter] = 0j


# Create bases and domain
start_init_time = time.time()  # start a timer to see how long things take
x_basis = de.Fourier('x', Nx, interval=(0, Lx), dealias=3/2)
# y_basis = de.Fourier('y', Ny, interval=(0, Ly), dealias=3/2)
z_basis = de.Fourier('z', Nz, interval=(0, Lz), dealias=3/2)
# domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64, mesh=(10, 12))
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)  # , mesh=(5, 8))

# problem = de.IVP(domain, variables=['u', 'v', 'w', 'p', 'Ax', 'Ay', 'Az', 'phi'], time='t')
problem = de.IVP(domain, variables=['phi', 'psi'], time='t')
# problem.parameters['MA2inv'] = 1.0/(MA**2.0)  # 99% sure that this is just H_B^*
problem.parameters['HB_star'] = HB_star
problem.parameters['Reinv'] = 1.0/Reynolds
problem.parameters['Rminv'] = 1.0/mReynolds
problem.parameters['MA'] = 1.0/np.sqrt(HB_star)

# problem.substitutions['Bx'] = "(dy(Az) - dz(Ay))"
# problem.substitutions['By'] = "(dz(Ax) - dx(Az))"
# problem.substitutions['Bz'] = "(1.0 + dx(Ay) - dy(Ax))"
# problem.substitutions['Jx'] = "(dy(Bz) - dz(By))"
# problem.substitutions['Jy'] = "(dz(Bx) - dx(Bz))"
# problem.substitutions['Jz'] = "(dx(By) - dy(Bx))"
problem.substitutions['zeta'] = "dx(dx(phi)) + dz(dz(phi))"
problem.substitutions['J'] = "dx(dx(psi)) + dz(dz(psi))"
problem.substitutions['u'] = "dz(phi)"
problem.substitutions['w'] = "-dx(phi)"
problem.substitutions['Bx'] = "dz(psi)"
problem.substitutions['Bz'] = "-dx(psi)"

# problem.add_equation("dt(zeta) - Reinv*(dx(dx(zeta)) + dz(dz(zeta))) = -dx(zeta) * dz(phi) + dx(phi) * dz(zeta) + HB_star * (dx(J) * dz(psi) - dx(psi) * dz(J)) - cos(x)", condition="(nx!=0) or (nz!=0)")
problem.add_equation("dt(zeta) - Reinv * (dx(dx(zeta)) + dz(dz(zeta))) = -dx(zeta) * dz(phi) + dx(phi) * dz(zeta) + HB_star * (dx(J) * dz(psi) - (dx(psi) - 1) * dz(J)) - Reinv * cos(x)", condition="(nx!=0) or (nz!=0)")
problem.add_equation("phi = 0", condition="(nx==0) and (nz==0)")
problem.add_equation("dt(psi) - Rminv*J = dx(phi)*dz(psi) - (dx(psi) - 1)*dz(phi)")

# Build solver
# solver = problem.build_solver(de.timesteppers.SBDF3)  # same timestepper as PADDIM
solver = problem.build_solver(de.timesteppers.SBDF2)
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
filter_field(psi)
phi['g'] += np.cos(x)

# Integration parameters
solver.stop_sim_time = stop_sim_time
solver.stop_wall_time = stop_wall_time
solver.stop_iteration = stop_iteration

# Analysis
snap = solver.evaluator.add_file_handler('snapshots', iter=500, max_writes=100000000) #was iter=250
snap.add_system(solver.state)

# CFL
# The following is meant to mimick what I've been doing with PADDIM
CFL = flow_tools.CFL(solver, initial_dt=init_dt, cadence=1, safety=0.6,
                     max_change=1.5, max_dt=2e-1, threshold=0.1)

# Unfortunately, I don't think PADDIM's CFL condition can be
# implemented exactly in Dedalus.
CFL.add_velocities(('u', 'w'))
CFL.add_velocities(('Bx/MA', 'Bz/MA'))

# Flow properties
# flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
# flow.add_property("sqrt(u*u + v*v + w*w) / Re", name='Re')

# Main loop
end_init_time = time.time()
logger.info('Initialization time: %f' %(end_init_time-start_init_time))
try:
    logger.info('Starting loop')
    start_run_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        solver.step(dt)
        if (solver.iteration-1) % 100 == 0: #was 100
            logger.info('Iteration: %i, sim_time: %e, dt: %e, wall_time: %.2f sec' %(solver.iteration, solver.sim_time, dt, time.time()-start_run_time))
            # logger.info('Max Re = %f' %flow.max('Re'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_run_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_run_time-start_run_time))
    logger.info('Run time: %f cpu-hr' %((end_run_time-start_run_time)/60/60*domain.dist.comm_cart.size))

