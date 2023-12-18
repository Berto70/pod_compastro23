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
    Internally it works considering the physical units in

        - mass: Msun
        - length: pc
        - velocity: km/s
        - time: Myr

    However, such scales can be changed at the class initialisation thorugh
    the intput value M, L, V, T. They indicate the scale of the input physical units
    with respect to the standard one, e.g. L=1000 if your input data are in kpc.
    Some examples:

    >>> nu = Nbody_units() # the physical inputs are assumed in  Msun,  pc, km/s, Myr
    >>>  nu = Nbody_units(M=1e3,L=1000,T=1000) # the physical inputs are assumed in units of 10^3 Msun, kpc, km/s, Gyr

    Once initialised, the class can be used to convert to physical untis to Nbody units
    and to Nbody units to the standard physical units (pc, Msun, km/s, Myr) or to the
    physical units used in input.
    For example, assume we have

    >>> x = 3  # kpc
    >>> vx = 10 # km/s
    >>> m=5  # Msun
    >>> t = 1 # Gyr
    >>> # let's set a Nbody_units instance
    >>> nu = Nbody_units(M=1,L=1000,V=1,T=1000)  # Msun, kpc, km/s Gyr
    >>> # transform to Nbody units
    >>> xn = nu.pos_to_Nbody(x) # x in kpc
    >>> vxn = nu.vel_to_Nbody(vx) # vx in km/s
    >>> mn = nu.m_to_Nbody(m) # m in Msun
    >>> tn = nu.m_to_Nbody(t) # t in Gyr
    >>> # Now transform to physical
    >>> xp  = nu.pos_to_physical(xn) # xn nbody, output in kpc (because of how we initialise nu)
    >>> xp2 = nu.pos_to_pc(xn) # xn nbody, output in pc (standard Nbody_units length scale)
    >>> vxp = nu.vel_to_physical(vxn) # vxn nbody, ouput in km/s (because of how we initialise nu)
    >>> tp = nu.t_to_physical(tn) # tn nbody, output in Gyr (because of how we initialise nu)
    >>> tp2 = nu.t_to_Myr(tn) # tn nbody, output in Myr (standard Nbody_units length scale)

    """

    Gcgs = 6.67430e-8 #Gravitational constants in cgs
    Lpc_cgs = 3.086e+18 #From parsec to cm
    Lkpc_cgs = 3.086e21 #From kpc to cm
    Msun_cgs = 1.9891e33 #from Msun to gr
    Lsun_cgs = 6.96e10 #from Rsun to cm
    Lau_cgs = 1.496e13 #from AU to cm
    cms_to_kms = 1e-5 #cm/s to km/s
    s_to_Myr = 3.1709791983765E-14 #s to Myr


    def __init__(self, M: float = 1., L: float = 1., V: float = 1., T: float = 1.):
        """
        Class initialised.
        The standard units are assumed in Msun (for the Mass), in pc (for the length),
        in km/s (for the velocity), and in Myr (for the time).
        These standards can be changes through the input parameter:

        :param M:  Set the physical input Mass scale in units of Msun
        :param L:  Set the physical input length scale in units of pc
        :param V:  Set the physical input velocity scale in units of km/s
        :param T:  Set the physical input time scale in units of Myr
        """

        cms_to_kms = 1e-5
        s_to_Myr = 3.1709791983765E-14

        # This are the units that are used as in input in units of
        self.Lunits_scale = L  # pc
        self.Munits_scale = M  # Msun
        self.Vunits_scale = V  # km/s
        self.Tunits_scale = T  # Myr

        Gscale_cgs = Nbody_units.Gcgs
        self.Lscale = L  # pc
        self.Mscale = M  # Msun
        Mscale_cgs = self.Munits_scale * Nbody_units.Msun_cgs  # Msun
        Lscale_cgs = self.Lunits_scale * Nbody_units.Lpc_cgs  # pc
        self.Vscale = cms_to_kms * np.sqrt(Gscale_cgs * Mscale_cgs / Lscale_cgs)  # km/s
        self.Tscale = s_to_Myr * np.sqrt(Lscale_cgs ** 3 / (Gscale_cgs * Mscale_cgs))  # Myr


    @classmethod
    def Lsun(cls, M: float = 1., V: float = 1., T: float = 1.) -> Nbody_units:
        """
        Class method to initialise the class using the solar radii as scale.
        Usage:

        >>> from fireworks.nbodylib.nunits import Nbody_units
        >>> nu = Nbody_units.Lsun()

        that is equivalent to
        >>> from fireworks.nbodylib.nunits import Nbody_units
        >>> nu = Nbody_units(L=Rsun_in_pc)

        :param M:  Set the physical input Mass scale in units of Msun
        :param V:  Set the physical input velocity scale in units of km/s
        :param T:  Set the physical input time scale in units of Myr
        :return: An Instance of the class
        """

        return cls(M=M, L=Nbody_units.Lsun_cgs / Nbody_units.Lpc_cgs, V=V, T=T)

    @classmethod
    def Lkpc(cls, M: float = 1., V: float = 1., T: float = 1.) -> Nbody_units:
        """
        Class method to initialise the class using 1 kpc as scale.
        Usage:

        >>> from fireworks.nbodylib.nunits import Nbody_units
        >>> nu = Nbody_units.Lkpc()

        that is equivalent to
        >>> from fireworks.nbodylib.nunits import Nbody_units
        >>> nu = Nbody_units(L=1000)

        :param M:  Set the physical input Mass scale in units of Msun
        :param V:  Set the physical input velocity scale in units of km/s
        :param T:  Set the physical input time scale in units of Myr
        :return: An Instance of the class
        """

        return cls(M=M, L=1e3)

    @classmethod
    def LAU(cls, M: float = 1., V: float = 1., T: float = 1) -> Nbody_units:
        """
        Class method to initialise the class using 1 AU as scale.
        Usage:

        >>> from fireworks.nbodylib.nunits import Nbody_units
        >>> nu = Nbody_units.LAU()

        :param M:  Set the physical input Mass scale in units of Msun
        :param V:  Set the physical input velocity scale in units of km/s
        :param T:  Set the physical input time scale in units of Myr
        :return: An Instance of the class
        """
        return cls(M=M, L=Nbody_units.Lau_cgs / Nbody_units.Lpc_cgs)

    @classmethod
    def Henon(cls,particles: Particles, L: float = 1, V: float = 1., T: float = 1., Q: float = 0.5) -> Nbody_units:
        """
        Special intialiser to use the Henon units.
        In these units, the mass scale is the total mass of the Nbody particles.
        In these units the total energy is always set to Etot=-1./4., so use these units
        only for bound Nbody systems.
        The scale radius is set so that it is related to the potential energy,
        while the scale veloisty is related to the kinetic enery of the sytem.
        The parameter Q set the virial ration Q=-Ekin/Epot, if Q=0.5 the system is virialised
        and Epot=-1/2 and Ekin=1/4.
        In case all the velocities are 0 Etot=Epot=-0.25.

        :param particles: that is an instance of the class :class:`~fireworks.particles.Particles`
        :param L:  Set the physical input length scale in units of pc
        :param V:  Set the physical input velocity scale in units of km/s
        :param T:  Set the physical input time scale in units of Myr
        :param Q:  Virial ratio, (0.5 virial equilibrium, >0.5 super-virial, <0.5 sub-virial)
        :return: An Instance of the class
        """

        pcopy = particles.copy()
        Mtot = np.sum(particles.mass)
        pcopy.mass = pcopy.mass/Mtot


        Epot = pcopy.Epot(softening=0.)
        Rscale = 4*Epot*(Q-1)

        Ekin = pcopy.Ekin()


        # Since V is automatically rescale based on Mtot and Lscale, lets set it to 1, take
        # the scale to remove it from the actual scaling for Henon units
        nu_test = Nbody_units(M=Mtot,L=L/Rscale,V=1)
        Vscale_original = nu_test.Vscale

        Vscale = 0.5*np.sqrt( (Q/(1-Q))/Ekin ) * Vscale_original

        return cls(M=Mtot, L=L/Rscale, V=V/Vscale, T=T)

    def pos_to_Nbody(self, pos: Union[npt.NDArray[np.float64],float]) -> npt.NDArray[np.float64]:
        """
        Transform positions from physics to Nbody units.

        :param pos: A Nx3 numpy array containing the positions of the N particles, the physics units
                    should be consistent with the scale used at the class initialisation (e.g. L=1 units are in pc,
                    L=1000 units are in kpc).
        :return: A Nx3 numpy array containing the positions of the N particles scaled to Nbody units
        """

        # First transform input to pc  ( pos*self.Lunits_scale)
        # Then rescale by Lscale

        return pos*self.Lunits_scale / self.Lscale

    def pos_to_pc(self, pos: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Transform positions from  Nbody units to pc

        :param pos:A Nx3 numpy array containing the positions of the N particles in Nbody units
        :return: A Nx3 numpy array containing the positions of the N particles in pc
        """

        return pos * self.Lscale
    def pos_to_physical(self, pos: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Transform positions  Nbody units to  physical units.
        the physics units depends on the scale used at the class initialisation (e.g. L=1 units are in pc,
        L=1000 units are in kpc).

        :param pos:A Nx3 numpy array containing the positions of the N particles in Nbody units
        :return: A Nx3 numpy array containing the positions of the N particles in physical units
        """

        # First transform Nbody input  to pc  ( pos*self.Lscae)
        # Then rescale by input units

        return self.pos_to_pc(pos) / self.Lunits_scale

    def vel_to_Nbody(self, vel: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Transform velocities from physics to Nbody units

        :param vel: A Nx3 numpy array containing the velocity of the N particles. the physics units
                    should be consistent with the scale used at the class initialisation (e.g. V=1 units are in km/s,
                    V=1e-3 units are in m/s).
        :return: A Nx3 numpy array containing the velocities of the N particles scaled to Nbody units
        """

        # First transform input to km/s  ( vel*self.Vunits_scale)
        # Then rescale by Vscale to get Nbody units

        return vel*self.Vunits_scale / self.Vscale #*V / self.Vscale

    def vel_to_kms(self, vel: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Transform velocities from  Nbody units to standard physics units km/s

        :param vel: A Nx3 numpy array containing the velocities of the N particles in Nbody units
        :return: A Nx3 numpy array containing the velocities of the N particles standard physics units (km/s)
        """

        return vel * self.Vscale

    def vel_to_physical(self, vel: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Transform velocities from Nbody  to physics units  (in km/s)
        The physics units should be consistent with the scale used at the class initialisation (e.g. V=1 units are in km/s,
        V=1e-3 units are in m/s).

        :param vel: A Nx3 numpy array containing the velocities of the N particles in Nbody units
        :return: A Nx3 numpy array containing the velocities of the N particles scaled in physics units
        """

        # First transform input to km/s  ( vel/self.Vscale)
        # Then transform to input units

        return self.vel_to_kms(vel)/self.Vunits_scale

    def m_to_Nbody(self, mass: Union[npt.NDArray[np.float64],float]) -> Union[npt.NDArray[np.float64],float] :
        """
        Transform masses from physics to Nbody units

        :param mass: A Nx1 numpy array (or a number) containing the mass of the  particles. the physics units
                    should be consistent with the scale used at the class initialisation (e.g. M=1 units are in Msun,
                    M=1e10 units are 10^{10} Msun).
        :return: A Nx1 numpy array (or a number) containing the masses of the N particles scaled to Nbody units
        """

        return mass*self.Munits_scale / self.Mscale

    def m_to_Msun(self, mass: Union[npt.NDArray[np.float64],float] ) -> Union[npt.NDArray[np.float64],float] :
        """
        Transform ,asses from  Nbody units to standard physics units Msun

        :param mass: A Nx1 numpy array (or a number) containing the mass of the  particles in Nbody units
        :return: A Nx1 numpy array (or a number) containing the masses of the N particles in standard physics units (Msun)
        """

        return mass * self.Mscale

    def m_to_physical(self, mass: Union[npt.NDArray[np.float64],float] ) -> Union[npt.NDArray[np.float64],float] :
        """
        Transform masses from Nbody units to physics

        :param mass: A Nx1 numpy array (or a number) containing the masses of the N particles in Nbody units.
        :return: A Nx1 numpy array (or a nunmber) containing the masses of the N particles scaled in physical units. The physics units
                    should be consistent with the scale used at the class initialisation (e.g. M=1 units are in Msun,
                    M=1e10 units are 10^{10} Msun).
        """
        return self.m_to_Msun(mass) / self.Munits_scale

    def t_to_Nbody(self, t: Union[npt.NDArray[np.float64],float]) -> Union[npt.NDArray[np.float64],float] :
        """
        Transform times from physics to Nbody units

        :param t: A Nx1 numpy array (or a number) containing time(s) in physics units. the physics units
                    should be consistent with the scale used at the class initialisation (e.g. T=1 units are in Myr,
                    T=1e3 units are Gyr).
        :return: A Nx1 numpy array (or a number) containing the time(s)  scaled to Nbody units
        """

        return t *self.Tunits_scale / self.Tscale

    def t_to_Myr(self, t):
        """
        Transform times from Nbody units to standard physics units (Myr)

        :param t: A Nx1 numpy array (or a number) containing time(s) in Nbody units.
        :return: A Nx1 numpy array (or a nunmber) containing the time(s) scaled in  standard physics units (Myr).
        """

        return t * self.Tscale

    def t_to_physical(self, t):
        """
        Transform time(s) from Nbody units to physics (in Myr)

        :param t: A Nx1 numpy array (or a number) containing the time(s)  scaled to Nbody units
        :return: A Nx1 numpy array (or a number) containing time(s) in physics units. the physics units
                    should be consistent with the scale used at the class initialisation (e.g. T=1 units are in Myr,
                    T=1e3 units are Gyr).
        """

        return self.t_to_Myr(t) / self.Tunits_scale