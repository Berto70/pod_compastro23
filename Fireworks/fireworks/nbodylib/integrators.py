"""
=========================================================
ODE integrators  (:mod:`fireworks.nbodylib.integrators`)
=========================================================

This module contains a collection of integrators to integrate one step of the ODE N-body problem
The functions included in this module should follow the input/output structure
of the template method :func:`~integrator_template`.

All the functions need to have the following input parameters:

    - particles, an instance of the  class :class:`~fireworks.particles.Particles`
    - tstep, timestep to be used to advance the Nbody system using the integrator
    - acceleration_estimator, it needs to be a function from the module (:mod:`fireworks.nbodylib.dynamics`)
        following the input/output style of the template function
        (:func:`fireworks.nbodylib.dynamics.acceleration_estimate_template`).
    - softening, softening values used to estimate the acceleration

Then, all the functions need to return the a tuple with 5 elements:

    - particles, an instance of the  class :class:`~fireworks.particles.Particles` containing the
        updates Nbody properties after the integration timestep
    - tstep, the effective timestep evolved in the simulation (for some integrator this can be
        different wrt the input tstep)
    - acc, the acceleration estimated for each particle, it needs to be a Nx3 numpy array,
        can be set to None
    - jerk, time derivative of the acceleration, it is an optional value, if the method
        does not estimate it, just set this element to None. If it is not None, it has
        to be a Nx3 numpy array.
    - pot, gravitational potential at the position of each particle. it is an optional value, if the method
        does not estimate it, just set this element to None. If it is not None, it has
        to be a Nx1 numpy array.

"""
from typing import Optional, Tuple, Callable
import numpy as np
import numpy.typing as npt
from ..particles import Particles
try:
    import tsunami
    tsunami_load=True
except:
    tsunami_load=False

def integrator_template(particles: Particles,
                        tstep: float,
                        acceleration_estimator: Callable,
                        softening: float = 0.):
    """
    This is an example template of the function you have to implement for the N-body integrators.
    :param particles: Instance of the class :class:`~fireworks.particles.Particles`
    :param tstep: Times-step for current integration (notice some methods can use smaller sub-time step to
    achieve the final result
    :param acceleration_estimator: It needs to be a function from the module (:mod:`fireworks.nbodylib.dynamics`)
    following the input/output style of the template function  (:func:`fireworks.nbodylib.dynamics.acceleration_estimate_template`).
    :param softening: softening parameter for the acceleration estimate, can use 0 as default value
    :return: A tuple with 5 elements:

        - The updated particles instance
        - tstep, the effective timestep evolved in the simulation (for some integrator this can be
            different wrt the input tstep)
        - acc, Nx3 numpy array storing the final acceleration for each particle, ca be set to None
        - jerk, Nx3 numpy array storing the time derivative of the acceleration, can be set to None
        - pot, Nx1 numpy array storing the potential at each particle position, can be set to None

    """

    acc,jerk,potential=acceleration_estimator(particles,softening)

    #Exemple of an Euler estimate
    particles.pos = particles.pos + particles.vel*tstep # Update pos
    particles.vel = particles.vel + acc*tstep # Update vel
    particles.set_acc(acc) #Set acceleration

    # Now return the updated particles, the acceleration, jerk (can be None) and potential (can nbe None=

    return (particles, tstep, acc, jerk, potential)

def integrator_tsunami(particles: Particles,
                       tstep: float,
                       acceleration_estimator: Optional[Callable]= None,
                       softening: float = 0.):
    """
    Special integrator that is actually a wrapper of the TSUNAMI integrator.
    TSUNAMI is regularised and it has its own way to estimate the acceleration,
    set the timestep and update the system.
    Therefore in this case tstep should not be the timestep of the integration, but rather
    the final time of our simulations, or an intermediate time in which we want to store
    the properties or monitor the sytems.
    Example:

    >>> tstart=0
    >>> tintermediate=[5,10,15]
    >>> tcurrent=0
    >>> for t in tintermediate:
    >>>     tstep=t-tcurrent
    >>>     particles, efftime,_,_,_=integrator_tsunami(particles,tstep)
    >>>     # Here we can save stuff, plot stuff, etc.
    >>>     tcurrent=tcurrent+efftime

    .. note::
        In general the TSUNAMI integrator is much faster than any integrator with can implement
        in this module.
        However, Before to start the proper integration, this function needs to perform some preliminary
        steps to initialise the TSUNAMI integrator. This can add a  overhead to the function call.
        Therefore, do not use this integrator with too small timestep. Acutally, the best timstep is the
        one that bring the system directly to the final time. However, if you want to save intermediate steps
        you can split the integration time windows in N sub-parts, calling N times this function.

    .. warning::
        It is important to notice that given the nature of the integrator (based on chain regularisation)
        the final time won't be exactly the one put in input. Take this in mind when using this  integrator.
        Notice also that the TSUNAMI integrator will rescale your system to the centre of mass frame of reference.



    :param particles: Instance of the class :class:`~fireworks.particles.Particles`
    :param tstep: final time of the current integration
    :param acceleration_estimator: Not used
    :param softening: Not used
    :return: A tuple with 5 elements:

        - The updated particles instance
        - tstep, the effective timestep evolved in the simulation, it wont'be exaxtly the one in input
        - acc, it is None
        - jerk, it is None
        - pot, it is None

    """
    if not tsunami_load: return ImportError("Tsunami is not available")

    code = tsunami.Tsunami(1.0, 1.0)
    code.Conf.dcoll = 0.0 #Disable collisions
    # Disable extra forces (already disabled by default)
    code.Conf.wPNs = False  # PNs
    code.Conf.wEqTides = False  # Equilibrium tides
    code.Conf.wDynTides = False  # Dynamical tides

    r=np.ones_like(particles.mass)
    st=np.array(np.ones(len(particles.mass))*(-1), dtype=int)
    code.add_particle_set(particles.pos, particles.vel, particles.mass, r, st)
    # Synchronize internal code coordinates with the Python interface
    code.sync_internal_state(particles.pos, particles.vel)
    # Evolve system from 0 to tstep - NOTE: the system final time won't be exacly tstep, but close
    code.evolve_system(tstep)
    # Synchronize realt to the real internal system time
    time = code.time
    # Synchronize coordinates to Python interface
    code.sync_internal_state(particles.pos, particles.vel)

    return (particles, time, None, None, None)