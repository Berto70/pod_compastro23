"""
====================================================================================================================
Collection of functions to estimate the timestep of the Nbody integrations (:mod:`fireworks.nbodylib.timesteps`)
====================================================================================================================

This module contains functions and utilities to estimate the timestep for the Nbody integrations.
There are no strict requirements for these functions. Obviously  it is important that they return a timestep.
It could be also useful to have as inputs a minimum and maximum timestep


"""
from typing import Optional, Tuple, Callable, Union, Dict
import numpy as np
import numpy.typing as npt
import gc
from ..particles import Particles

def adaptive_timestep_simple(particles: Particles, tmin: Optional[float] = None, tmax: Optional[float] = None) -> float:
    """
    Very simple adaptive timestep based on the ratio between the position and the velocity of the particles

    :param particles: that is an instance of the class :class:`~fireworks.particles.Particles`
    :return: estimated timestep
    """

    # Simple idea, use the R/V of the particles to have an estimate of the required timestep
    # Take the minimum among all the particles

    ts = np.nanmin(particles.radius()/particles.vel_mod())

    # Check tmin, tmax
    if tmin is not None: ts = max(ts, tmin)
    if tmax is not None: ts = min(ts, tmax)

    return ts

def adaptive_timestep_vel(particles: Particles, eta: float, acc = npt.NDArray[np.float64], tmin: Optional[float] = None, tmax: Optional[float] = None) -> float:

    # acc_mod = np.sqrt(np.sum(acc*acc, axis=1))[:,np.newaxis]

    ts = eta*np.nanmin(np.linalg.norm(particles.vel_mod(), axis=1)/np.linalg.norm(acc, axis=1))

    if tmin is not None: ts = max(ts, tmin)
    if tmax is not None: ts = min(ts, tmax)

    return ts

def adaptive_timestep_jerk(acc: npt.NDArray[np.float64], jerk:npt.NDArray[np.float64], eta: float, tmin: Optional[float] = None, tmax: Optional[float] = None) -> float:
    """
    Very simple adaptive timestep based on the ratio between the position and the velocity of the particles

    :param particles: that is an instance of the class :class:`~fireworks.particles.Particles`
    :return: estimated timestep
    """

    # Simple idea, use the R/V of the particles to have an estimate of the required timestep
    # Take the minimum among all the particles

    # ts = eta*np.nanmin(acc/jerk)
    ts = eta*np.nanmin(np.linalg.norm(acc, axis=1)/np.linalg.norm(jerk, axis=1))

    # Check tmin, tmax
    if tmin is not None: ts = max(ts, tmin)
    if tmax is not None: ts = min(ts, tmax)

    return ts

def adaptive_timestep(integrator: Callable,
                       int_args: Dict[str, float],
                       int_rank: int,
                       predictor: Callable,
                       pred_args: Dict[str, float],
                       pred_rank: int,
                       epsilon: float,
                       tmin: Optional[float] = None,
                       tmax: Optional[float] = None):
    
    '''
    Return the adaptive timestep by computing the differences between the prediction made by an integrator and its lower order version.

    :param particles: that is an instance of the class :class:`~fireworks.particles.Particles`.
    :param epsilon: Arbitrary threshold.
    :param integrator: Callable function of the integrator for which we want to compute the adaptive timestep.
    :param int_args: Dictionary containing the arguments of the integrator function.
    :param int_rank: Integrator rank.
    :param predictor: Callable function of the lower rank integrator.
    :param pred_args: Dictionary containing the arguments of the lower rank integrator function.
    :param pred_rank: Integrator rank.
    :param dt: Timestep computed at the previous step.
    :param tmin: Minimum possible output.
    :param tmax: Maximum possible output.
    :return: Estimated timestep.
    '''    

    n_min = int_rank #- pred_rank
    
    particles_int, dt, _, _, _ = integrator(**int_args)
    particles_pred, _, _, _, _ = predictor(**pred_args)

    del _ 
    gc.collect()


    # r_int = np.sqrt(np.sum(particles_int.pos*particles_int.pos, axis=1))
    # r_pred = np.sqrt(np.sum(particles_pred.pos*particles_pred.pos, axis=1))
    
    eps_r = np.linalg.norm(particles_int.pos - particles_pred.pos, axis=1)

    # v_int = np.sqrt(np.sum(particles_int.vel*particles_int.vel, axis=1))
    # v_pred = np.sqrt(np.sum(particles_pred.vel*particles_pred.vel, axis=1))
    eps_v = np.linalg.norm(particles_int.vel - particles_pred.vel, axis=1)


    ts = dt* np.min([np.power(np.nanmin(epsilon/(eps_r+0.000001)), 1/n_min), np.power(np.nanmin(epsilon/(eps_v+0.000001)), 1/n_min)])
    # ts = dt* np.power(np.min([np.nanmin(epsilon/(eps_r+0.000001)), np.nanmin(epsilon/(eps_v+0.000001))]) , 1/n_min)


    if tmin is not None: ts = max(ts, tmin)
    if tmax is not None: ts = min(ts, tmax)

    return ts
