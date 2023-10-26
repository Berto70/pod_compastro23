"""
====================================================================================================================
Collection of functions to estimate the Gravitational forces and accelerations (:mod:`fireworks.nbodylib.dynamics`)
====================================================================================================================

This module contains a collection of functions to estimate acceleration due to
gravitational  forces.

Each method implemented in this module should follow the input-output structure show for the
template function  :func:`~acceleration_estimate_template`:

Every function needs to have two input parameters:

    - particles, that is an instance of the class :class:`~fireworks.particles.Particles`
    - softening, it is the gravitational softening. The parameters need to be included even
        if the function is not using it. Use a default value of 0.

The function needs to return a tuple containing three elements:

    - acc, the acceleration estimated for each particle, it needs to be a Nx3 numpy array,
        this element is mandatory it cannot be 0.
    - jerk, time derivative of the acceleration, it is an optional value, if the method
        does not estimate it, just set this element to None. If it is not None, it has
        to be a Nx3 numpy array.
    - pot, gravitational potential at the position of each particle. it is an optional value, if the method
        does not estimate it, just set this element to None. If it is not None, it has
        to be a Nx1 numpy array.


"""
from typing import Optional, Tuple
import numpy as np
import numpy.typing as npt
from ..particles import Particles

try:
    import pyfalcon
    pyfalcon_load=True
except:
    pyfalcon_load=False

def acceleration_estimate_template(particles: Particles, softening: float =0.) \
        -> Tuple[npt.NDArray[np.float64],Optional[npt.NDArray[np.float64]],Optional[npt.NDArray[np.float64]]]:
    """
    This an empty functions that can be used as a basic template for
    implementing the other functions to estimate the gravitational acceleration.
    Every function of this kind needs to have two input parameters:

        - particles, that is an instance of the class :class:`~fireworks.particles.Particles`
        - softening, it is the gravitational softening. The parameters need to be included even
          if the function is not using it. Use a default value of 0.

    The function needs to return a tuple containing three elements:

        - acc, the acceleration estimated for each particle, it needs to be a Nx3 numpy array,
            this element is mandatory it cannot be 0.
        - jerk, time derivative of the acceleration, it is an optional value, if the method
            does not estimate it, just set this element to None. If it is not None, it has
            to be a Nx3 numpy array.
        - pot, gravitational potential at the position of each particle. it is an optional value, if the method
            does not estimate it, just set this element to None. If it is not None, it has
            to be a Nx1 numpy array.

    :param particles: An instance of the class Particles
    :param softening: Softening parameter
    :return: A tuple with 3 elements:

        - acc, Nx3 numpy array storing the acceleration for each particle
        - jerk, Nx3 numpy array storing the time derivative of the acceleration, can be set to None
        - pot, Nx1 numpy array storing the potential at each particle position, can be set to None
    """

    acc  = np.zeros(len(particles))
    jerk = None
    pot = None

    return (acc,jerk,pot)


def acceleration_pyfalcon(particles: Particles, softening: float =0.) \
        -> Tuple[npt.NDArray[np.float64],Optional[npt.NDArray[np.float64]],Optional[npt.NDArray[np.float64]]]:
    """
    Estimate the acceleration following the fast-multipole gravity Dehnen2002 solver (https://arxiv.org/pdf/astro-ph/0202512.pdf)
    as implementd in pyfalcon (https://github.com/GalacticDynamics-Oxford/pyfalcon)

    :param particles: An instance of the class Particles
    :param softening: Softening parameter
    :return: A tuple with 3 elements:

        - Acceleration: a NX3 numpy array containing the acceleration for each particle
        - Jerk: None, the jerk is not estimated
        - Pot: a Nx1 numpy array containing the gravitational potential at each particle position
    """

    if not pyfalcon_load: return ImportError("Pyfalcon is not available")

    acc, pot = pyfalcon.gravity(particles.pos,particles.mass,softening)
    jerk = None

    return acc, jerk, pot

