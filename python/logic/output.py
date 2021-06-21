"""
    This file is a partial driving script for boussinesq dynamics.  Here,
    formulations of the boussinesq equations are handled in a clean way using
    classes.
"""
import numpy as np
import logging
from collections import OrderedDict
logger = logging.getLogger(__name__.split('.')[-1])

def initialize_output(solver, data_dir, Lx,Lz, threeD=False, volumes=False,
                      max_writes=20, output_dt=0.1, slice_output_dt=1, vol_output_dt=10, out_iter=np.inf,
                      mode="overwrite", **kwargs):
    """
    Sets up output from runs.
    """

    Lx = Lx
    Lz = Lz
    

    # Analysis
    analysis_tasks = OrderedDict()
    profiles = solver.evaluator.add_file_handler(data_dir+'profiles', sim_dt=output_dt, max_writes=max_writes*10, mode=mode, iter=out_iter)
    profiles.add_task("plane_avg(u)", name="u")
    profiles.add_task("plane_avg(w)", name="w")
    profiles.add_task("plane_avg(Bx)", name="Bx")
    profiles.add_task("plane_avg(Bz)", name="Bz")
    profiles.add_task("plane_avg(b_mag)", name="b_mag")
    profiles.add_task("plane_avg(zeta)", name="zeta")
    profiles.add_task("plane_avg(J)", name="J")
    profiles.add_task("plane_avg(phi)", name="phi")
    profiles.add_task("plane_avg(psi)", name="psi")

    analysis_tasks['profiles'] = profiles

    scalar = solver.evaluator.add_file_handler(data_dir+'scalar', sim_dt=output_dt, max_writes=max_writes*100, mode=mode, iter=out_iter)
    scalar.add_task("vol_avg(0.5*vel_rms**2)", name="KE")
    scalar.add_task("vol_avg(0.5*output_MA**2)", name="ME")
    scalar.add_task("vol_avg(u)",  name="u")
    scalar.add_task("vol_avg(w)",  name="w")
    scalar.add_task("vol_avg(Bx)",  name="Bx")
    scalar.add_task("vol_avg(Bz)",  name="Bz")
    scalar.add_task("vol_avg(b_mag)",  name="b_mag")
    scalar.add_task("vol_avg(zeta)",  name="zeta")
    scalar.add_task("vol_avg(J)",  name="J")
    scalar.add_task("vol_avg(phi)",  name="phi")
    scalar.add_task("vol_avg(psi)",  name="psi")
    analysis_tasks['scalar'] = scalar

        # Analysis
    slices = solver.evaluator.add_file_handler(data_dir+'slices', sim_dt=slice_output_dt, max_writes=max_writes, mode=mode, iter=out_iter*(slice_output_dt/output_dt))
    slices.add_task("u")
    slices.add_task("w")
    slices.add_task("Bx")
    slices.add_task("Bz")
    slices.add_task("zeta")
    slices.add_task("J")
    slices.add_task("phi")
    slices.add_task("psi")
    analysis_tasks['slices'] = slices


    return analysis_tasks



