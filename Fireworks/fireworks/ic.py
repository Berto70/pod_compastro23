"""
==============================================================
Initial conditions utilities , (:mod:`fireworks.ic`)
==============================================================

This module contains functions and utilities to generate
initial conditions for the Nbody simulations.
The basic idea is that each function/class should returns
an instance of the class :class:`~fireworks.particles.Particles`

"""

import numpy as np
from .particles import Particles

def ic_random_normal(N: int, mass: float=1) -> Particles:
    """
    Generate random initial condition drawing from a normal distribution
    (centred in 0 and with dispersion 1) for the position and velocity.
    The mass is instead the same for all the particles.

    :param N: number of particles to generate
    :param mass: mass of the particles
    :return: An instance of the class :class:`~fireworks.particles.Particles` containing the generated particles
    """

    pos  = np.random.normal(size=3*N).reshape(N,3) # Generate 3xN 1D array and then reshape as a Nx3 array
    vel  = np.random.normal(size=3*N).reshape(N,3) # Generate 3xN 1D array and then reshape as a Nx3 array
    mass = np.ones(N)*mass

    return Particles(position=pos, velocity=vel, mass=mass)
