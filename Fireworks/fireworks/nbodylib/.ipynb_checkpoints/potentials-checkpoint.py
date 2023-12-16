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
import scipy.special as scisp
from ..particles import Particles

class Potential_Base:
    """
    Basic class to be used as a parent class to instantiate new potentials.
    Each new class needs to define:

        - an __init__ method in which to set the potential parameters
        - the method _acceleration. This method needs to accept as parameters
          an object of class particles and the softening as a float with default value =0 (even if it practically not used).
          it has to return a Nx3 array containing the accelerations, and optionally a Nx3 array containing the jerk, and a Nx1 containing the potential
          If the jerk and/or the potential are not estimated they have to be set to 0.

    Then, to initialise and use the potential

    >>> from fireworks.potentials import Example_Potential
    >>> pot = Example_Potential(par1=1,par2=2) # Instantiate the potential
    >>> accfunc = pot.acceleration # This method can be used with the fireworks integrators

    See the implementation  of the potential :class:`~.Point_Mass` to have an idea on how to create a new potential
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

    def evaluate(self,R,z=0, softening=0):

        R=np.atleast_1d(R)
        z=np.atleast_1d(z)

        pos=np.zeros(shape=(len(R),3))
        if len(R)==1:
            pos[:,0]=R[0]
        else:
            pos[:,0]=R

        if len(z)==1:
            pos[:,2]=z[0]
        else:
            pos[:,2]=z

        vel = np.zeros_like(pos)
        mass = np.zeros(len(R))

        part=Particles(position=pos, velocity=vel, mass=mass)

        return self.acceleration(part,softening)



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

class MultiPotential(Potential_Base):
    """
    This class can be used to instantiate an objected dertive from the :class:`~.Potential_Base`
    combining multiple objects from that class. It can be used for example to estimate the potential
    of a Galaxy composed by multiple mass components (for example a disc, a bulge and a halo).
    Internally, the function :func:`~.MultiPotential._acceleration` estimate the total acceleration
    summing the contribution of all the components set during the initialisation.

    Example:

    >>> from firworks.nbodylib.potentials import Pot1,Pot2,Pot3,MultiPotential
    >>> # Example a Galaxy composed by a halo, bulge and disc
    >>> halo = Pot1(param1=100) # Use and instantiate a Pot1 object to describe a Galaxy halo
    >>> bulge = Pot2(param1=3,param2=5,param3=4) # Use and instantiate a Pot2 object to describe a Galaxy bulge
    >>> disc = Pot3(param1=20, param2=0.2) # Use and instantiate a Pot3 object to describe a Galaxy disc
    >>> galaxy_pot = MultiPotential([Pot1,Pot2,Pot3]) # Instantiate a Multipotential object
    >>> galaxy_pot.acceleration  # This method can be used with the fireworks integrators


    """
    def __init__(self, potential_list=[]):
        """
        Initialise the object

        :param potential_list: list of instances (at least one) of classes derived from the class :class:`~.Potential_Base`
        """

        self._potentials = potential_list

    def _acceleration(self, particles: Particles, softening: float = 0.) \
            -> Tuple[npt.NDArray[np.float64],Optional[npt.NDArray[np.float64]],Optional[npt.NDArray[np.float64]]]:
        """
        Iteratively estimate the total acceleration (and jerk and potential) obtained
        summing the contributions of all the potentials objects used to initialise the class.
        The acceleration is always a Nx3 numpy array (where N is the number of particles).
        For the Jerk and Potential, if at least one of the potential returns None, they are set to None


        :param particles: An instance of the class Particles
        :param softening: Softening parameter
        :return: A tuple with 3 elements:

            - acc, Nx3 numpy array storing the acceleration for each particle
            - jerk, Nx3 numpy array storing the time derivative of the acceleration, can be set to None
            - pot, Nx1 numpy array storing the potential at each particle position, can be set to None

        """

        acc  = np.zeros_like(particles.pos)
        jerk = None
        pot_tot  =  0

        for pot in self._potentials:

            this_acc, _, this_pot = pot.acceleration(particles=particles,softening=softening)
            acc+=this_acc
            if this_pot is not None and pot_tot is not None: pot_tot+=this_pot
            else: pot_tot=None

        return acc,jerk,pot_tot



class Point_Mass(Potential_Base):
    """
    Simple point mass potential.
    It assumes the presence of a point of mass M fixed at centre of the frame of reference.
    The potential is:

    .. math::

        \Phi = - \\frac{M}{\sqrt{r^2 + \epsilon^2}},

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