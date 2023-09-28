"""
==============================================================
Particles data structure , (:mod:`fireworks.particles`)
==============================================================

This module contains the class used to store the Nbody particles data


"""

__all__ = ['Particles']

import numpy as np

class Particles:
    """
    Simple class to store the properties position, velocity, mass of the particles

    """
    def __init__(self, position, velocity, mass):

        self.pos = np.atleast_2d(position)
        if self.pos.shape[1] != 3: print(f"Input position should contain a Nx3 array, current shape is {self.pos.shape}")

        self.vel = np.atleast_2d(velocity)
        if self.vel.shape[1] != 3: print(f"Input velocity should contain a Nx3 array, current shape is {self.pos.shape}")
        if len(self.vel) != len(self.pos): print(f"Position and velocity in input have not the same number of elemnts")

        self.mass = np.atleast_1d(mass)
        if len(self.mass) != len(self.pos): print(f"Position and mass in input have not the same number of elemnts")

        self.ID=np.arange(len(self.mass))

