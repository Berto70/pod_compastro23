"""
==============================================================
Particles data structure , (:mod:`fireworks.particles`)
==============================================================

This module contains the class used to store the Nbody particles data


"""
from __future__ import annotations
import numpy as np
import numpy.typing as npt
__all__ = ['Particles']


class Particles:
    """
    Simple class to store the properties position, velocity, mass of the particles.
    Example:

    >>> from fireworks.particles import Particles
    >>> position=np.array([[1.,1.,1.],[2.,2.,2.],[3.,3.,3.]])
    >>> velocity=np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
    >>> mass=np.array([1.,1.,1.])
    >>> P=Particles(position,velocity,mass)
    >>> P.pos # particles'positions
    >>> P.vel # particles'velocities
    >>> P.mass # particles'masses
    >>> P.ID # particles'unique IDs

    The class contains also methods to estimate the radius of all the particles (:func:`~Particles.radius`),
    the module of the velociy of all the particles (:func:`~Particles.vel_mod`), and the module the positition and
    velocity of the centre of mass (:func:`~Particles.com_pos` and :func:`~Particles.com_vel`)

    >>> P.radius() # return a Nx1 array with the particle's radius
    >>> P.vel_mod() # return a Nx1 array with the module of the particle's velocity
    >>> P.com() # array with the centre of mass position (xcom,ycom,zcom)
    >>> P.com() # array with the centre of mass velocity (vxcom,vycom,vzcom)

    It is also possibile to set an acceleration for each particle, using the method set_acc
    Example:

    >>> acc= some_method_to_estimate_acc(P.position)
    >>> P.set_acc(acc)
    >>> P.acc # particles's accelerations

    Notice, if never initialised, P.acc is equal to None

    The class can be used also to estimate the total, kinetic and potential energy of the particles
    using the methods :func:`~Particles.Etot`, :func:`~Particles.Ekin`, :func:`~Particles.Epot`
    **NOTICE:** these methods need to be implemented by you!!!

    The method :func:`~Particles.copy` can be used to be obtaining a safe copy of the current
    Particles instances. Safe means that changing the members of the copied version will not
    affect the members or the original instance
    Example

    >>> P=Particles(position,velocity,mass)
    >>> P2=P.copy()
    >>> P2.pos[0] = np.array([10,10,10]) # P.pos[0] will not be modified!

    """
    def __init__(self, position: npt.NDArray[np.float64], velocity: npt.NDArray[np.float64], mass: npt.NDArray[np.float64]):
        """
        Class initialiser.
        It assigns the values to the class member pos, vel, mass and ID.
        ID is just a sequential integer number associated to each particle.

        :param position: A Nx3 numpy array containing the positions of the N particles
        :param velocity: A Nx3 numpy array containing the velocity of the N particles
        :param mass: A Nx1 numpy array containing the mass of the N particles
        """

        self.pos = np.array(np.atleast_2d(position), dtype=float)
        if self.pos.shape[1] != 3: print(f"Input position should contain a Nx3 array, current shape is {self.pos.shape}")

        self.vel = np.array(np.atleast_2d(velocity), dtype=float)
        if self.vel.shape[1] != 3: print(f"Input velocity should contain a Nx3 array, current shape is {self.pos.shape}")
        if len(self.vel) != len(self.pos): print(f"Position and velocity in input have not the same number of elements")

        self.mass = np.array(np.atleast_1d(mass), dtype=float)
        if len(self.mass) != len(self.pos): print(f"Position and mass in input have not the same number of elements")

        self.ID=np.arange(len(self.mass), dtype=int)

        self.acc=None

        self.flag=None

    def set_flag(self, flag: npt.NDArray[np.int64]):
        """
        Set the flag, 1 or 0, to check for system membership or non-membership, respectively. 

        :param flag: A Nx1 numpy array containing the flag of the N particles
        """

        flag = np.atleast_1d(flag)
        if flag.shape[0] != self.pos.shape[0]: print(f"Flag and position in input have not the same number of elements, current shape is {flag.shape}")

        self.flag=flag

    def set_acc(self, acceleration: npt.NDArray[np.float64]):
        """
        Set the particle's acceleration

        :param acceleration: A Nx3 numpy array containing the acceleration of the N particles
        """

        acc = np.atleast_2d(acceleration)
        if acceleration.shape[1] != 3: print(f"Input acceleration should contain a Nx3 array, current shape is {acc.shape}")

        self.acc=acc

    def radius(self) -> npt.NDArray[np.float64]:
        """
        Estimate the particles distance from the origin of the frame of reference.

        :return:  a Nx1 array containing the particles' distance from the origin of the frame of reference.
        """

        return np.sqrt(np.sum(self.pos*self.pos, axis=1))[:,np.newaxis]

    def vel_mod(self) -> npt.NDArray[np.float64]:
        """
        Estimate the module of the velocity of the particles

        :return: a Nx1 array containing the module of the particles's velocity
        """

        return np.sqrt(np.sum(self.vel*self.vel, axis=1))[:,np.newaxis]

    def com_pos(self) -> npt.NDArray[np.float64]:
        """
        Estimate the position of the centre of mass

        :return: a numpy  array with three elements corresponding to the centre of mass position
        """

        return np.sum(self.mass*self.pos.T,axis=1)/np.sum(self.mass)

    def com_vel(self) -> npt.NDArray[np.float64]:
        """
        Estimate the velocity of the centre of mass

        :return: a numpy  array with three elements corresponding to centre of mass velocity
        """

        return np.sum(self.mass*self.vel.T,axis=1)/np.sum(self.mass)

    def Ekin(self) -> float:
        """
        Estimate the total potential energy of the particles:
        Ekin=0.5 sum_i mi vi*vi

        :return: total kinetic energy
        """
        # #TOU HAVE TO IMPLEMENT IT
        # # Use the class member, e.g. vel=self.vel, mass=self.mass
        # raise NotImplementedError("Ekin method still not implemented")

        # Ekin = 0.5 * np.sum(self.mass[:, np.newaxis] * self.vel**2)

        Ekin = 0.5 * np.sum(self.mass * (np.sum(self.vel**2, axis=1)))

        # vel_squared = np.square(self.vel)
        # Ekin = 0.5 * np.sum(self.mass * vel_squared) # this is prob more efficient for large arrays

        return Ekin

    def Epot(self, softening: float = 0.) -> float:
        """
        Estimate the total potential energy of the particles:
        Epot=-0.5 sumi sumj mi*mj / sqrt(rij^2 + eps^2)
        where eps is the softening parameter

        :param softening: Softening parameter
        :return: The total potential energy of the particles
        """
        # #TOU HAVE TO IMPLEMENT IT
        # # Use the class member, e.g. vel=self.vel, mass=self.mass
        # raise NotImplementedError("Ekin method still not implemented")

            
        # rij = np.sqrt(np.sum((self.pos[:, np.newaxis] - self.pos) ** 2, axis=2))
        # Epot = - np.sum(self.mass[:, np.newaxis] * self.mass / np.sqrt(rij ** 2 + softening ** 2))

        # Calculate all pairwise distances between bodies
        rij = np.linalg.norm(self.pos[:, np.newaxis, :] - self.pos, axis=2)
        
        # Exclude self-distances (diagonal elements) to avoid division by zero
        np.fill_diagonal(rij, 1.0)
        
        # Calculate potential energy using vectorized operations
        Epot_mat = - np.outer(self.mass, self.mass) / np.power((rij**2 + softening**2), 3/2)
        
        # Sum over all unique pairs
        Epot = np.sum(np.triu(Epot_mat, k=1))
        
        return Epot

    # def Ekin(self) -> float:
    #     """
    #     Estimate the total potential energy of the particles:
    #     Ekin=0.5 sum_i mi vi*vi

    #     :return: total kinetic energy
    #     """
    #     mass = self.mass
    #     mod_vel = np.sqrt(np.sum(self.vel**2, axis=1).astype(float))

    #     Ekin = 0.5*np.sum(mass * mod_vel*mod_vel)

    #     return Ekin
    #     #TOU HAVE TO IMPLEMENT IT
    #     # Use the class member, e.g. vel=self.vel, mass=self.mass
    #     # raise NotImplementedError("Ekin method still not implemented")

    # def Epot(self,softening: float = 0.) -> float:
    #     """
    #     Estimate the total potential energy of the particles:
    #     Epot=-0.5 sumi sumj mi*mj / sqrt(rij^2 + eps^2)
    #     where eps is the softening parameter

    #     :param softening: Softening parameter
    #     :return: The total potential energy of the particles
    #     """

    #     N_particles =  len(self.pos)
    #     pos_x = self.pos[:, 0] - self.pos[:, 0].reshape(N_particles, 1)  #broadcasting of (N,) on (N,1) array, obtain distance along x in an (N,N) matrix
    #     pos_y = self.pos[:, 1] - self.pos[:, 1].reshape(N_particles, 1) 
    #     pos_z = self.pos[:, 2] - self.pos[:, 2].reshape(N_particles, 1)

    #     r_ij = np.sqrt((pos_x**2 + pos_y**2 + pos_z**2).astype(float))

    #     mass_ij = self.mass * self.mass.reshape((N_particles, 1))
    #     mass_ij[r_ij==0]=0.0    #in this way the m_i*m_i component are removed
    #     r_ij[r_ij==0]=1.0       #in thi way we remove the division by zero for the r_ii component 
        
    #     Epot = -0.5*np.sum(mass_ij/(np.abs(r_ij) + softening*softening))

    #     return Epot
        
    #     #TOU HAVE TO IMPLEMENT IT
    #     # Use the class member, e.g. vel=self.vel, mass=self.mass
    #     # raise NotImplementedError("Ekin method still not implemented")


    def Etot(self,softening: float = 0.) -> tuple[float,float,float]:
        """
        Estimate the total  energy of the particles: Etot=Ekintot + Epottot

        :param softening: Softening parameter
        :return: a tuple with

            - Total energy
            - Total kinetic energy
            - Total potential energy
        """

        Ekin = self.Ekin()
        Epot = self.Epot(softening=softening)
        Etot = Ekin + Epot

        return Etot, Ekin, Epot

    def copy(self) -> Particles:
        """
        Return a copy of this Particle class

        :return: a copy of the Particle class
        """

        par=Particles(np.copy(self.pos),np.copy(self.vel),np.copy(self.mass))
        if self.acc is not None: par.acc=np.copy(self.acc)

        return Particles(np.copy(self.pos),np.copy(self.vel),np.copy(self.mass))

    def __len__(self) -> int:
        """
        Special method to be called when  this class is used as argument
        of the Python built-in function len()
        :return: Return the number of particles
        """

        return len(self.mass)

    def __str__(self) -> str:
        """
        Special method to be called when  this class is used as argument
        of the Python built-in function print()
        :return: short info message
        """

        return f"Instance of the class Particles\nNumber of particles: {self.__len__()}"

    def __repr__(self) -> str:

        return self.__str__()
