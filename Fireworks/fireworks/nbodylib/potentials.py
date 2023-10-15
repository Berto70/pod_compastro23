"""
==================================================================================================================
Collection of classes to estimate the Gravitational forces from potentials (:mod:`fireworks.nbodylib.potentials`)
==================================================================================================================

This module contains a collection of classes and functions to estimate acceleration due to
gravitational  forces of a fixed potential

"""
import numpy as np
import numpy.typing as npt
from ..particles import Particles

class Potential_Base:

    def __init__(self):
        pass

    def acceleration(self, particles: Particles, softening: float = 0.) \
        -> Tuple[npt.NDArray[np.float64],Optional[npt.NDArray[np.float64]],Optional[npt.NDArray[np.float64]]]:

        return self._acceleration(particles)

    def _acceleration(self, particles: Particles, softening: float = 0.) \
            -> Tuple[npt.NDArray[np.float64],Optional[npt.NDArray[np.float64]],Optional[npt.NDArray[np.float64]]]:

        raise NotImplementedError(f"Method _acceleration for the {self.__name__} is not implemented yet")

class Point_Mass(Potential_Base):

    def __init__(self, Mass: float):
        self.Mass = Mass

    def _acceleration(self, particles: Particles, softening: float = 0.) \
            -> Tuple[npt.NDArray[np.float64],Optional[npt.NDArray[np.float64]],Optional[npt.NDArray[np.float64]]]:

        r  = particles.radius()
        reff2 = r*r + softening*softening
        acc = np.zeros_like()




