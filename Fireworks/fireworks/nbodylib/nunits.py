"""
=====================================================================
Utilities to deal with Nbody units (:mod:`fireworks.nbodylib.nunits`)
=====================================================================

This module contains functions and utilities to deal with N-body units
and units conversion

"""
from __future__ import annotations
from typing import Optional, Tuple, Callable, Union
import numpy as np
from ..particles import Particles

class Nbody_units:
    """
    This class is used to handle transformation from and to Nbody Units.
    It assumes that the physical units are:

        - length: pc
        - mass: Msun
        - velocity: km/s
        - time: Myr

    However, such scales can be changed at the class initialisation
    """

    Gcgs = 6.67430e-8 #Gravitational constants in cgs
    Lpc_cgs = 3.086e+18 #From parsec to cm
    Lkpc_cgs = 3.086e21 #From kpc to cm
    Msun_cgs = 1.9891e33 #from Msun to cm
    Lsun_cgs = 6.96e10 #from Rsun to cm
    Lau_cgs = 1.496e13 #from AU to cm
    cms_to_kms = 1e-5 #cm/s to km/s
    s_to_Myr = 3.1709791983765E-14 #s to Myr

    def __init__(self, M: float = 1., L: float = 1., V: float = 1., T: float = 1.):
        """
        Intialiser for the class. It is used to set the scales.
        The units are assumed in Msun (for the Mass), in pc (for the length),
        in km/s (for the velocity), annd in Myr (for the time).


        :param M:  Set the Mass scale in units of Msun
        :param L:  Set the length scale in units of pc
        :param V:  Set the velocity scale in units of km/s
        :param T:  Set the time scale in units of Myr
        """

        cms_to_kms = 1e-5
        s_to_Myr = 3.1709791983765E-14

        Gscale_cgs = Nbody_units.Gcgs
        self.Lscale = L  # cgs
        self.Mscale = M  # cgs
        Mscale_cgs = self.Mscale * Nbody_units.Msun_cgs  # Msun
        Lscale_cgs = self.Lscale * Nbody_units.Lpc_cgs  # pc
        self.Vscale = V * cms_to_kms * np.sqrt(Gscale_cgs * Mscale_cgs / Lscale_cgs)  # km/s
        self.Tscale = T * s_to_Myr * np.sqrt(Lscale_cgs ** 3 / (Gscale_cgs * Mscale_cgs))  # Myr

    @classmethod
    def Lsun(cls, M: float = 1.) -> Nbody_units:
        """
        Class method to initialise the class using the solar radii as scale.
        All the other scale are set to default
        Usage:

        >>> from fireworks.nbodylib.nunits import Nbody_units
        >>> nu = Nbody_units.Lsun()

        :param M: Set the Mass scale in units of Msun
        :return: An Instance of the class
        """

        return cls(M=M, L=Nbody_units.Lsun_cgs / Nbody_units.Lpc_cgs)

    @classmethod
    def Lkpc(cls, M=1) -> Nbody_units:
        """
        Class method to initialise the class using 1 kpc as scale.
        All the other scale are set to default
        Usage:

        >>> from fireworks.nbodylib.nunits import Nbody_units
        >>> nu = Nbody_units.Lkpc()

        :param M: Set the Mass scale in units of Msun
        :return: An Instance of the class
        """

        return cls(M=M, L=1e3)

    @classmethod
    def LAU(cls, M: float=1) -> Nbody_units:
        """
        Class method to initialise the class using 1 kpc as scale.
        All the other scale are set to default
        Usage:

        >>> from fireworks.nbodylib.nunits import Nbody_units
        >>> nu = Nbody_units.LAU()

        :param M: Set the Mass scale in units of Msun
        :return: An Instance of the class
        """
        return cls(M=M, L=Nbody_units.Lau_cgs / Nbody_units.Lpc_cgs)

    @classmethod
    def Henon(self,particles: Particles, L: float = 1, V: float = 1., T: float = 1., Q: float = 0.5) -> Nbody_units:
        """
        Use the Henon units, so that the mass scale is the total mass of the Nbody particles.
        Then, the scale radius is set so that the potential energy is Epot=-0.5.
        These units also assume that the system is in virial equilibrium, therefore, the velocity
        scale is set in the way that Ekin=0.25, so Etot=-0.25.
        This of couse is not true if all the velocities are 0. In this case Etot=Epot=-0.5.
        The assumption on what is the length scale of the particles is controlled by L, that by default
        is 1 pc. The particles class in input is not modified.
        Example:

        >>> # Assume we have a code that generate a Plummer sphere with the position in kpc and the velocity in kms
        >>> # To instansiated the Nbody units to scale the sytem we can use
        >>> nu = Nbody_units.Henon(particles, L=1000) # L in kpc, V already in km/s
        >>> # so since the mass scale is the total mass  to transform to Nbody units
        >>> posnbody = nu.pos_to_Nbody(particles.pos,L=1000) # L=1000 because this function assume the input in pc, while we are using kpc
        >>> velnbody = nu.pos_to_Nbody(particles.vel) # input scale in km/s -> ok
        >>> massnbody = nu.pos_to_Nbody(particles.mass) # input scale in Msun -> ok

        :param particles: that is an instance of the class :class:`~fireworks.particles.Particles`
        :param L:  Set the length scale in units of pc
        :param V:  Set the velocity scale in units of km/s
        :param T:  Set the time scale in units of Myr
        :param Q:  Virial ratio, (0.5 virial equilibrium, >0.5 super-virial, <0.5 sub-virial)
        :return: An Instance of the class
        """

        pcopy = particles.copy()
        Mtot = np.sum(particles.mass)
        pcopy.mass = pcopy.mass/Mtot

        Epot = pcopy.Epot(softening=0.)
        Rscale = 4*Epot*(Q-1)

        Ekin = pcopy.Ekin()
        Vscale = 0.5*np.sqrt( ((1-Q)/Q)/Ekin )

        return cls(M=Mtot, L=L/Rscale, V=V/Vscale, T=T)

    def pos_to_Nbody(self, pos: Union[npt.NDArray[np.float64],float], L: float=1.) -> npt.NDArray[np.float64]:
        """
        Transform positions from physics (in pc) to Nbody units

        :param pos: A Nx3 numpy array containing the positions of the N particles
        :param L: The input scale in pc (i.e. if the input is in kpc, use 1000)
        :return: A Nx3 numpy array containing the positions of the N particles scaled to Nbody units
        """

        return pos*L / self.Lscale

    def pos_to_physical(self, pos: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Transform positions from Nbody units to physics (in pc)

        :param pos: A Nx3 numpy array containing the positions of the N particles in Nbody units
        :return: A Nx3 numpy array containing the positions of the N particles scaled in physical units (pc)
        """
        return pos * self.Lscale

    def vel_to_Nbody(self, vel: npt.NDArray[np.float64], V: float =1.) -> npt.NDArray[np.float64]:
        """
        Transform velocities from physics to Nbody units

        :param vel: A Nx3 numpy array containing the velocity of the N particles
        :param V: The input scale in km/s (i.e. if the input is in units of 10 km/s, use 10)
        :return: A Nx3 numpy array containing the velocities of the N particles scaled to Nbody units
        """

        return vel*V / self.Vscale

    def vel_to_physical(self, vel: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Transform velocities from Nbody units to physics (in km/s)

        :param vel: A Nx3 numpy array containing the velocities of the N particles in Nbody units
        :return: A Nx3 numpy array containing the velocities of the N particles scaled in physical units (km/s)
        """
        return vel * self.Vscale

    def m_to_Nbody(self, mass: Union[npt.NDArray[np.float64],float] , M: float=1.) -> Union[npt.NDArray[np.float64],float] :
        """
        Transform masses from physics (in Msun) to Nbody units

        :param mass: A Nx1 numpy array (or a number) containing the mass of the  particles
        :param M: The input scale in Msun (i.e. if the input is in 10^5 Msun, use 10^5)
        :return: A Nx1 numpy array (or a number) containing the masses of the N particles scaled to Nbody units
        """

        return mass*M / self.Mscale

    def m_to_physical(self, mass: Union[npt.NDArray[np.float64],float] ) -> Union[npt.NDArray[np.float64],float] :
        """
        Transform masses from Nbody units to physics (in Msun)

        :param mass: A Nx1 numpy array (or a number) containing the masses of the N particles in Nbody units
        :return: A Nx1 numpy array (or a nunmber) containing the masses of the N particles scaled in physical units (Msun)
        """
        return mass * self.Mscale

    def t_to_Nbody(self, t: Union[npt.NDArray[np.float64],float], T: float =1.) -> Union[npt.NDArray[np.float64],float] :
        """
        Transform times from physics (in Myr) to Nbody units

        :param t: A Nx1 numpy array (or a number) containing time(s)
        :param T: The input scale in Myr (i.e. if the input is in Gyr, use 10^3)
        :return: A Nx1 numpy array (or a number) containing the time(s)  scaled to Nbody units
        """

        return t / self.Tscale

    def t_to_physical(self, t):
        """
        Transform time(s) from Nbody units to physics (in Myr)

        :param t: A Nx1 numpy array (or a number) containing the time(s)  scaled to Nbody units
        :return: A Nx1 numpy array (or a number) containing time(s) in physical units (Myr)
        """
        return t * self.Tscale