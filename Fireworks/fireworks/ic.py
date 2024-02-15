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

def ic_random_uniform(N: int, mass: list[float, float], pos: list[float, float], vel: list[float, float]) -> Particles:
    """
    Generate random initial condition drawing from a uniform distribution
    (between upper and lower boundary) for the mass, position and velocity.

    :param N: number of particles to generate
    :type N: int
    :param mass: list of lower and upper boundary for mass particle distribution
    :type mass: list of float
    :param pos: list of lower and upper boundary for position particle distribution
    :type pos: list of float
    :param vel: list of lower and upper boundary for velocity particle distribution
    :type vel: list of float
    :return: An instance of the class :class:`~fireworks.particles.Particles` containing the generated particles, characterized by pos, vel and mass.
    """

    mass = np.random.uniform(low=mass[0], high=mass[1], size=N) # Generate 1D array of N elements
    pos = np.random.uniform(low=pos[0], high=pos[1], size=3*N).reshape(N,3) # Generate 3xN 1D array and then reshape as a Nx3 array
    vel = np.random.uniform(low=vel[0], high=vel[1], size=3*N).reshape(N,3) # Generate 3xN 1D array and then reshape as a Nx3 array

    return Particles(position=pos, velocity=vel, mass=mass)

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


def ic_two_body(mass1: float, mass2: float, rp: float, e: float):
    """
    Create initial conditions for a two-body system.
    By default the two bodies will placed along the x-axis at the
    closest distance rp.
    Depending on the input eccentricity the two bodies can be in a
    circular (e<1), parabolic (e=1) or hyperbolic orbit (e>1).

    :param mass1:  mass of the first body [nbody units]
    :param mass2:  mass of the second body [nbody units]
    :param rp: closest orbital distance [nbody units]
    :param e: eccentricity
    :return: An instance of the class :class:`~fireworks.particles.Particles` containing the generated particles
    """

    Mtot=mass1+mass2

    if e==1.:
        vrel=np.sqrt(2*Mtot/rp)
    else:
        a=rp/(1-e)
        vrel=np.sqrt(Mtot*(2./rp-1./a))

    # To take the component velocities
    # V1 = Vcom - m2/M Vrel
    # V2 = Vcom + m1/M Vrel
    # we assume Vcom=0.
    v1 = -mass2/Mtot * vrel
    v2 = mass1/Mtot * vrel

    pos  = np.array([[0.,0.,0.],[rp,0.,0.]])
    vel  = np.array([[0.,v1,0.],[0.,v2,0.]])
    mass = np.array([mass1,mass2])

    return Particles(position=pos, velocity=vel, mass=mass)