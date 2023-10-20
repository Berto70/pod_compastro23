"""
==================================================================================================================
Collection of classes to estimate the Gravitational forces from potentials (:mod:`fireworks.nbodylib.potentials`)
==================================================================================================================

This module contains a collection of classes and functions to estimate acceleration due to
gravitational  forces of a fixed potential

In Fireworks that functions to estimate the acceleration depends only by the particles and the softening.
However, the potential can depend on a number of additional parameters.
Therefore instead of creating directly the function each Potential is created through a calls derived
from the basic class :class:`~.Potential_Base`, this class returns the acceleration through the method
:func:`~.Potential_Base.acceleration`. This is method is just a wrapper of the specific method
:func:`~.Potential_Base._acceleration` that needs to be implemented for each potential.
So a Potential needs to add an __init__ method that takes into account all the parameters necessary
to define the potential and the method _acceleration to properly estimate and return the acceleration.
Then this method can be used inside the fireworks integrators (:mod:`fireworks.nbodylib.integrators`)

Example

>>> from fireworks.potentials import Example_Potential
>>> pot = Example_Potential(par1=1,par2=2) # Instantiate the potential
>>> accfunc = pot.acceleration # This method can be used with the fireworks integrators

"""
from typing import Optional, Tuple, Callable, Union, List
import numpy as np
import numpy.typing as npt
from ..particles import Particles

class Potential_Base:
    """
    Basic class to be used as a parent class to instantiate new potentials
    """

    def __init__(self):
        pass

    def acceleration(self, particles: Particles, softening: float = 0.) \
        -> Tuple[npt.NDArray[np.float64],Optional[npt.NDArray[np.float64]],Optional[npt.NDArray[np.float64]]]:
        """
        Estimate the acceleration (and optionally the jerk and the potential)
        :param particles: An instance of the class Particles
        :param softening: Softening parameter
        :return: A tuple with 3 elements:

            - acc, Nx3 numpy array storing the acceleration for each particle
            - jerk, Nx3 numpy array storing the time derivative of the acceleration, can be set to None
            - pot, Nx1 numpy array storing the potential at each particle position, can be set to None

        """

        return self._acceleration(particles)

    def _acceleration(self, particles: Particles, softening: float = 0.) \
            -> Tuple[npt.NDArray[np.float64],Optional[npt.NDArray[np.float64]],Optional[npt.NDArray[np.float64]]]:
        """
        Estimate the acceleration (and optionally the jerk and the potential)
        :param particles: An instance of the class Particles
        :param softening: Softening parameter
        :return: A tuple with 3 elements:

            - acc, Nx3 numpy array storing the acceleration for each particle
            - jerk, Nx3 numpy array storing the time derivative of the acceleration, can be set to None
            - pot, Nx1 numpy array storing the potential at each particle position, can be set to None

        """

        raise NotImplementedError(f"Method _acceleration for the {self.__name__} is not implemented yet")

class Point_Mass(Potential_Base):
    """
    Simple point mass potential.
    It assumes the presence of a point of mass M fixed at centre of the frame of reference.
    The potential is:
        Phi = - Mass/sqrt(r^2 + epsilons^2)
    where epsilon is the softening.

    The potential is spherically symmetric and considering the Nbody units, the circular velocity
    at r=1 is v=sqrt(Mass)

    """

    def __init__(self, Mass: float):
        """
        Initialise the potential setting the mass of the point
        :param Mass: Mass of the point [nbody units]
        """
        self.Mass = Mass

    def _acceleration(self, particles: Particles, softening: float = 0.) \
            -> Tuple[npt.NDArray[np.float64],Optional[npt.NDArray[np.float64]],Optional[npt.NDArray[np.float64]]]:
        """
        Estimate the acceleration for a point mass potential, vec(F)= - M/|r^2+epsilon^2| *(vec(r)/r)

        :param particles: An instance of the class Particles
        :param softening: Softening parameter
        :return: A tuple with 3 elements:

            - acc, Nx3 numpy array storing the acceleration for each particle
            - jerk, set to None
            - pot, set to None

        """

        r  = particles.radius()
        reff2 = r*r + softening*softening
        acc = -self.Mass/reff2 * (particles.pos/r) # p.pos/r means x/r, y/r, z/r, these are the three components of the acceleration

        return acc, None, None # not include jerk and potential atm



