"""
====================================================================================================================
Collection of functions to estimate the Gravitational forces and accelerations (:mod:`fireworks.nbodylib.dynamics`)
====================================================================================================================

This module contains a collection of functions to estimate acceleration due to
gravitational forces.

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
import fireworks.ic as fic

try:
    import pyfalcon
    pyfalcon_load=True
except:
    pyfalcon_load=False


# def acc_2body(particles: Particles):

#     # acc_2body = np.zeros(shape=(len(particles), 3))

#     for i in range(len(particles)):
#         for j in range(len(particles)):
#             if i != j:

#                 # dx = particles.pos[i,0] - particles.pos[j,0]
#                 # dy = particles.pos[i,1] - particles.pos[j,1]
#                 # dz = particles.pos[i,2] - particles.pos[j,2]
#                 # r = np.sqrt(dx**2 + dy**2 + dz**2)
#                 # acc_2body[i,0] = -particles.mass[j]*dx/r**3
#                 # acc_2body[i,1] = -particles.mass[j]*dy/r**3
#                 # acc_2body[i,2] = -particles.mass[j]*dz/r**3

#                 dx = particles.pos[i,0] - particles.pos[j,0]
#                 dy = particles.pos[i,1] - particles.pos[j,1]
#                 dz = particles.pos[i,2] - particles.pos[j,2]
#                 r = np.sqrt(dx**2 + dy**2 + dz**2)
#                 acc_2body0 = -particles.mass[j]*dx/r**3
#                 acc_2body1 = -particles.mass[j]*dy/r**3
#                 acc_2body2 = -particles.mass[j]*dz/r**3

#     return acc_2body0, acc_2body1, acc_2body2

    # return acc_2body

def acceleration_direct(particles: Particles, softening: float = 0.) \
    -> Tuple[npt.NDArray[np.float64], Optional[npt.NDArray[np.float64]], Optional[npt.NDArray[np.float64]]]:

    """
    Function used to estimate the gravitational acceleration using a direct summation method.

    :param particles: An instance of the class :class:`~fireworks.particles.Particles`
    :param softening: Gravitational softening parameter. Default value is 0.
    :type softening: float
    :return: A tuple with 3 elements:

        - Acceleration: a Nx3 numpy array containing the acceleration for each particle
        - Jerk: Time derivative of the acceleration. If not None, it has to be a Nx3 numpy array
        - Pot: Gravitational potential at the position of each particle. If not None, it has to be a Nx1 numpy array

    """

    acc = np.zeros(shape=(len(particles), 3))
    jerk = None
    pot = None

    # if ic == 'ic_2_body':
    #     particles = fic.ic_two_body()

    # for i in range(0, len(particles)-1):
    #     for j in range(i+1, len(particles)):
    #         acc_ij0, acc_ij1, acc_ij2 = acc_2body(particles)
    #         acc[i, 0] += acc_ij0
    #         acc[i, 1] += acc_ij1
    #         acc[i, 2] += acc_ij2
    #         acc[j, 0] -= particles.mass[i]/particles.mass[j]*acc_ij0
    #         acc[j, 1] -= particles.mass[i]/particles.mass[j]*acc_ij1
    #         acc[j, 2] -= particles.mass[i]/particles.mass[j]*acc_ij2

    # return (acc, jerk, pot)   

    # Get the number of particles in the system.
    #n_particles = particles.shape[0]

    # Initialize the acceleration array.
    acc = np.zeros(shape=(len(particles), 3))

    # Compute the acceleration of each particle due to the force from all other particles.
    for i in range(len(particles)):
        for j in range(i + 1, len(particles)):
            # Compute the distance between particles i and j.
            r = np.linalg.norm(particles.pos[i, :3] - particles.pos[j, :3])

            # Compute the force between particles i and j.
            #F = -particles[i, 3] * particles[j, 3] / (r**3) * (particles[i, :3] - particles[j, :3])

            # Add the force from particle j to the acceleration of particle i.
            acc[i, :] += particles.mass[j] * (particles.pos[i, :3] - particles.pos[j, :3]) / (r**3)

    return (acc, jerk, pot)


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