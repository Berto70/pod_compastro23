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
    - external_accelerations, this is an optional input, if not None, it has to be a list
        of additional callable to estimate additional acceleration terms (e.g. an external potential or
        some drag term depending on the particles velocity). Notice that if the integrator uses the jerk
        all this additional terms should return the jerk otherwise the jerk estimate is biased.

Then, all the functions need to return the a tuple with 5 elements:

    - particles, an instance of the  class :class:`~fireworks.particles.Particles` containing the
        updates Nbody properties after the integration timestep
    - tstep, the effective timestep evolved in the simulation (for some integrator this can be
        different wrt the input tstep)
    - acc, the total acceleration estimated for each particle, it needs to be a Nx3 numpy array,
        can be set to None
    - jerk, total time derivative of the acceleration, it is an optional value, if the method
        does not estimate it, just set this element to None. If it is not None, it has
        to be a Nx3 numpy array.
    - pot, total  gravitational potential at the position of each particle. it is an optional value, if the method
        does not estimate it, just set this element to None. If it is not None, it has
        to be a Nx1 numpy array.

"""
from typing import Optional, Tuple, Callable, Union, List
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
                        acceleration_estimator: Union[Callable,List],
                        softening: float = 0.,
                        external_accelerations: Optional[List] = None,
                        args: Optional[dict] = None):
    """
    This is an example template of the function you have to implement for the N-body integrators.
    :param particles: Instance of the class :class:`~fireworks.particles.Particles`
    :param tstep: Times-step for current integration (notice some methods can use smaller sub-time step to
    achieve the final result
    :param acceleration_estimator: It needs to be a function from the module (:mod:`fireworks.nbodylib.dynamics`)
    following the input/output style of the template function  (:func:`fireworks.nbodylib.dynamics.acceleration_estimate_template`).
    :param softening: softening parameter for the acceleration estimate, can use 0 as default value
    :param external_accelerations: a list of additional force estimators (e.g. an external potential field) to
    consider to estimate the final acceleration (and if available jerk) on the particles
    :return: A tuple with 5 elements:

        - The updated particles instance
        - tstep, the effective timestep evolved in the simulation (for some integrator this can be
            different wrt the input tstep)
        - acc, Nx3 numpy array storing the final acceleration for each particle, ca be set to None
        - jerk, Nx3 numpy array storing the time derivative of the acceleration, can be set to None
        - pot, Nx1 numpy array storing the potential at each particle position, can be set to None

    """
    if args is not None:
        acc, jerk, potential = acceleration_estimator(particles, softening, **args)
    else:
        acc, jerk, potential = acceleration_estimator(particles, softening)

    #Check additional accelerations
    if external_accelerations is not None:
        for ext_acc_estimator in external_accelerations:
            acct, jerkt, potentialt = ext_acc_estimator(particles, softening)
            acc += acct
            if jerk is not None and jerkt is not None: jerk += jerkt
            if potential is not None and potentialt is not None: potential += potentialt

    #Exemple of an Euler estimate
    particles.pos = particles.pos + particles.vel*tstep # Update pos
    particles.vel = particles.vel + acc*tstep # Update vel
    particles.set_acc(acc) #Set acceleration

    # Now return the updated particles, the acceleration, jerk (can be None) and potential (can be None)

    return (particles, tstep, acc, jerk, potential)

def integrator_euler(particles: Particles,
                     tstep: float,
                     acceleration_estimator: Union[Callable,List],
                     softening: float = 0.,
                     external_accelerations: Optional[List] = None,
                     args: Optional[dict] = None):
    """
    Simple implementation of a Modified Euler integrator for N-body simulations.
    :param particles: Instance of the class :class:`~fireworks.particles.Particles`
    :param tstep: Times-step for current integration (notice some methods can use smaller sub-time step to
    achieve the final result
    :param acceleration_estimator: It needs to be a function from the module (:mod:`fireworks.nbodylib.dynamics`)
    following the input/output style of the template function  (:func:`fireworks.nbodylib.dynamics.acceleration_estimate_template`).
    :param softening: softening parameter for the acceleration estimate, can use 0 as default value
    :param external_accelerations: a list of additional force estimators (e.g. an external potential field) to
    consider to estimate the final acceleration (and if available jerk) on the particles
    :return: A tuple with 5 elements:

        - The updated particles instance
        - tstep, the effective timestep evolved in the simulation (for some integrator this can be
            different wrt the input tstep)
        - acc, Nx3 numpy array storing the final acceleration for each particle, ca be set to None
        - jerk, Nx3 numpy array storing the time derivative of the acceleration, can be set to None
        - pot, Nx1 numpy array storing the potential at each particle position, can be set to None

    """
    if args is not None:
        acc, jerk, potential = acceleration_estimator(particles, softening, **args)
    else:
        acc, jerk, potential = acceleration_estimator(particles, softening)

    # Check additional accelerations
    if external_accelerations is not None:
        for ext_acc_estimator in external_accelerations:
            acct, jerkt, potentialt = ext_acc_estimator(particles, softening)
            acc += acct
            if jerk is not None and jerkt is not None:
                jerk += jerkt
            if potential is not None and potentialt is not None:
                potential += potentialt

    # Euler integration
    particles.vel = particles.vel + acc * tstep  # Update vel
    particles.pos = particles.pos + particles.vel * tstep  # Update pos
    particles.set_acc(acc)  # Set acceleration

    # Return the updated particles, the acceleration, jerk (can be None), and potential (can be None)
    return (particles, tstep, acc, jerk, potential)

def integrator_hermite(particles: Particles,
                        tstep: float,
                        acceleration_estimator: Union[Callable,List],
                        softening: float = 0.,
                        external_accelerations: Optional[List] = None,
                        args: Optional[dict] = None):
    
    if args is not None:
        acc, jerk, potential = acceleration_estimator(particles, softening, **args)
    else:
        acc, jerk, potential = acceleration_estimator(particles, softening)

    # This integrator requires jerk
    if jerk is None: raise ValueError("Hermite integrator requires jerk")

    #Check additional accelerations
    if external_accelerations is not None:
        for ext_acc_estimator in external_accelerations:
            acct, jerkt, potentialt = ext_acc_estimator(particles, softening)
            acc += acct
            if jerk is not None and jerkt is not None: jerk += jerkt
            if potential is not None and potentialt is not None: potential += potentialt

    # Preditor sub-step
    # 9 
    vel_p = particles.vel + acc*tstep + (jerk * tstep**2)/2

    # 10
    pos_p = particles.pos + particles.vel*tstep + (acc * tstep**2)/2 + (jerk*tstep**3)/6

    # 11 # 12
    if args is not None:
        acc_p, jerk_p, _ = acceleration_estimator(Particles(pos_p, vel_p, particles.mass), softening=0.0, **args)
    else:
        acc_p, jerk_p, _ = acceleration_estimator(Particles(pos_p, vel_p, particles.mass), softening=0.0)

    #Check additional accelerations
    if external_accelerations is not None:        
        for ext_acc_estimator in external_accelerations:
            acct, jerkt, potentialt = ext_acc_estimator(particles, softening)
            acc_p += acct
            if jerk_p is not None and jerkt is not None: jerk_p += jerkt
            if potential is not None and potentialt is not None: potential += potentialt


    # Corrector sub-step
    # alternative more accurate version p. 34

    # First derivative jerk 
    j_1 = (-6*(acc - acc_p) - (4*jerk + 2*jerk_p)*tstep)* tstep**(-2)
    
    # Second derivative jerk
    j_2 = (12*(acc - acc_p) + 6*(jerk + jerk_p)*tstep)* tstep**(-3)
    
    # 13b
    particles.vel = vel_p + (tstep**3)*j_1/6 + (tstep**4)*j_2/24

    # 14b
    particles.pos = pos_p + (tstep**4)*j_1/24 + (tstep**5)*j_2/120

    particles.set_acc(acc)

    return (particles, tstep, acc, jerk, potential)

def integrator_leapfrog(particles: Particles,
                        tstep: float,
                        acceleration_estimator: Union[Callable,List],
                        softening: float = 0.,
                        external_accelerations: Optional[List] = None,
                        args: Optional[dict] = None):
    """
    Simple implementation of a symplectic Leapfrog (Verlet) integrator for N-body simulations.
    :param particles: Instance of the class :class:`~fireworks.particles.Particles`
    :param tstep: Times-step for current integration (notice some methods can use smaller sub-time step to
    achieve the final result
    :param acceleration_estimator: It needs to be a function from the module (:mod:`fireworks.nbodylib.dynamics`)
    following the input/output style of the template function  (:func:`fireworks.nbodylib.dynamics.acceleration_estimate_template`).
    :param softening: softening parameter for the acceleration estimate, can use 0 as default value
    :param external_accelerations: a list of additional force estimators (e.g. an external potential field) to
    consider to estimate the final acceleration (and if available jerk) on the particles
    :return: A tuple with 5 elements:

        - The updated particles instance
        - tstep, the effective timestep evolved in the simulation (for some integrator this can be
            different wrt the input tstep)
        - acc, Nx3 numpy array storing the final acceleration for each particle, ca be set to None
        - jerk, Nx3 numpy array storing the time derivative of the acceleration, can be set to None
        - pot, Nx1 numpy array storing the potential at each particle position, can be set to None

    """
    if args is not None:
        acc, jerk, potential = acceleration_estimator(particles, softening, **args)
    else:
        acc, jerk, potential = acceleration_estimator(particles, softening)

    # Check additional accelerations
    if external_accelerations is not None:
        for ext_acc_estimator in external_accelerations:
            acct, jerkt, potentialt = ext_acc_estimator(particles, softening)
            acc += acct
            if jerk is not None and jerkt is not None: jerk += jerkt
            if potential is not None and potentialt is not None: potential += potentialt

    # vel_m = particles.vel + 0.5*acc*tstep # half-step velocity
    # particles.pos = particles.pos + vel_m*tstep # Update pos
    # particles.vel = vel_m + 0.5*acc*tstep # Update vel
    # particles.set_acc(acc)  # Set acceleration


    # removing half-step velocity
    particles.pos = particles.pos + particles.vel*tstep + 0.5*acc*(tstep**2)

    if args is not None:
        acc2, jerk2, pot2 = acceleration_estimator(Particles(particles.pos, particles.vel, particles.mass), softening, **args)
    else:
        acc2, jerk2, pot2 = acceleration_estimator(Particles(particles.pos, particles.vel, particles.mass), softening)

    #Check additional accelerations
    if external_accelerations is not None:
        for ext_acc_estimator in external_accelerations:
            acct, jerkt, potentialt = ext_acc_estimator(particles, softening)
            acc2 += acct
            if jerk2 is not None and jerkt is not None: jerk2 += jerkt
            if pot2 is not None and potentialt is not None: pot2 += potentialt

    particles.vel = particles.vel + 0.5*(acc + acc2)*tstep
    particles.set_acc(acc2)

    return (particles, tstep, acc2, jerk2, pot2)

def integrator_heun(particles: Particles,
                   tstep: float,
                   acceleration_estimator: Union[Callable,List],
                   softening: float = 0.,
                   external_accelerations: Optional[List] = None,
                   args: Optional[dict] = None):
    
    # 2nd order RK, Ralston's method: minimizes the truncation error.
    if args is not None:
        acc, jerk, potential = acceleration_estimator(particles, softening, **args)
    else: 
        acc, jerk, potential = acceleration_estimator(particles, softening)

    # Check additional accelerations
    if external_accelerations is not None:
        for ext_acc_estimator in external_accelerations:
            acct, jerkt, potentialt = ext_acc_estimator(particles, softening)
            acc += acct
            if jerk is not None and jerkt is not None:
                jerk += jerkt
            if potential is not None and potentialt is not None:
                potential += potentialt

    k1r = particles.vel*tstep
    k1v = acc*tstep

    if args is not None:
        acc2, jerk2, pot2 = acceleration_estimator(Particles(particles.pos + (2/3)*k1r, particles.vel + (2/3)*k1v, particles.mass), softening, **args)
    else:
        acc2, jerk2, pot2 = acceleration_estimator(Particles(particles.pos + (2/3)*k1r, particles.vel + (2/3)*k1v, particles.mass), softening)
        
    #Check additional accelerations
    if external_accelerations is not None:
        for ext_acc_estimator in external_accelerations:
            acct, jerkt, potentialt = ext_acc_estimator(particles, softening)
            acc2 += acct
            if jerk2 is not None and jerkt is not None: jerk2 += jerkt
            if pot2 is not None and potentialt is not None: pot2 += potentialt

    k2r = (particles.vel + (2/3)*k1v)*tstep
    k2v = acc2*tstep

    particles.pos = particles.pos + (1/4)*(k1r + 3*k2r)
    particles.vel = particles.vel + (1/4)*(k1v + 3*k2v)

    particles.set_acc(acc2)

    return (particles, tstep, acc2, jerk2, pot2)

def integrator_rk4(particles: Particles,
                   tstep: float,
                   acceleration_estimator: Union[Callable,List],
                   softening: float = 0.,
                   external_accelerations: Optional[List] = None,
                   args: Optional[dict] = None):

    
    if args is not None:
        acc, jerk, potential = acceleration_estimator(particles, softening, **args)
    else:
        acc, jerk, potential = acceleration_estimator(particles, softening)


    # Check additional accelerations
    if external_accelerations is not None:
        for ext_acc_estimator in external_accelerations:
            acct, jerkt, potentialt = ext_acc_estimator(particles, softening)
            acc += acct
            if jerk is not None and jerkt is not None:
                jerk += jerkt
            if potential is not None and potentialt is not None:
                potential += potentialt

    k1r = particles.vel*tstep
    k1v = acc*tstep

    if args is not None:
        acc2, jerk2, pot2 = acceleration_estimator(Particles(particles.pos + (1/2)*k1r, particles.vel + (1/2)*k1v, particles.mass), softening, **args)
    else:
        acc2, jerk2, pot2 = acceleration_estimator(Particles(particles.pos + (1/2)*k1r, particles.vel + (1/2)*k1v, particles.mass), softening)

    #Check additional accelerations
    if external_accelerations is not None:
        for ext_acc_estimator in external_accelerations:
            acct, jerkt, potentialt = ext_acc_estimator(particles, softening)
            acc2 += acct
            if jerk2 is not None and jerkt is not None: jerk2 += jerkt
            if pot2 is not None and potentialt is not None: pot2 += potentialt

    k2r = (particles.vel + 0.5*k1v)*tstep
    k2v = acc2*tstep

    if args is not None:
        acc3, jerk3, pot3 = acceleration_estimator(Particles(particles.pos + (1/2)*k2r, particles.vel + (1/2)*k2v, particles.mass), softening, **args)
    else:
        acc3, jerk3, pot3 = acceleration_estimator(Particles(particles.pos + (1/2)*k2r, particles.vel + (1/2)*k2v, particles.mass), softening)

    #Check additional accelerations
    if external_accelerations is not None:
        for ext_acc_estimator in external_accelerations:
            acct, jerkt, potentialt = ext_acc_estimator(particles, softening)
            acc3 += acct
            if jerk3 is not None and jerkt is not None: jerk3 += jerkt
            if pot3 is not None and potentialt is not None: pot3 += potentialt

    k3r = (particles.vel + 0.5*k2v)*tstep
    k3v = acc3*tstep

    if args is not None:
        acc4, jerk4, pot4 = acceleration_estimator(Particles(particles.pos + k3r, particles.vel + k3v, particles.mass), softening, **args)
    else:
        acc4, jerk4, pot4 = acceleration_estimator(Particles(particles.pos + k3r, particles.vel + k3v, particles.mass), softening)

    #Check additional accelerations
    if external_accelerations is not None:
        for ext_acc_estimator in external_accelerations:
            acct, jerkt, potentialt = ext_acc_estimator(particles, softening)
            acc4 += acct
            if jerk4 is not None and jerkt is not None: jerk4 += jerkt
            if pot4 is not None and potentialt is not None: pot4 += potentialt

    k4r = (particles.vel + k3v)*tstep
    k4v = acc4*tstep

    particles.pos = particles.pos + (1/6)*(k1r + 2*k2r + 2*k3r + k4r)
    particles.vel = particles.vel + (1/6)*(k1v + 2*k2v + 2*k3v + k4v)
    particles.set_acc(acc4) 

    return (particles, tstep, acc4, jerk4, pot4)

def integrator_tsunami(particles: Particles,
                       tstep: float,
                       acceleration_estimator: Optional[Callable] = None,
                       softening: float = 0.,
                       external_accelerations: Optional[List] = None):
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
    >>>     ## NOTICE: Sometime the efftime can be so large that the current time is now larger than the current t
    >>>     ## the simple check below allow to skip these steps and go directly to the t in tintermediate for which
    >>>     ## t>tcurrent and we have to actually integrate the system
    >>>     if tstep<=: continue # continue means go to the next step (i.e. next t in the array)
    >>>
    >>>     particles, efftime,_,_,_=integrator_tsunami(particles,tstep)
    >>>     # Here we can save stuff, plot stuff, etc.
    >>>     tcurrent=tcurrent+efftime

    .. note::
        In general the TSUNAMI integrator is much faster than any integrator with can implement
        in this module.
        However, Before to start the proper integration, this function needs to perform some preliminary
        steps to initialise the TSUNAMI integrator. This can add a overhead to the function call.
        Therefore, do not use this integrator with too small timestep. Actually, the best timestep is the
        one that bring the system directly to the final time. However, if you want to save intermediate steps
        you can split the integration time windows in N sub-parts, calling N times this function.

    .. warning::
        It is important to notice that given the nature of the integrator (based on chain regularisation)
        the final time won't be exactly the one put in input. Take this in mind when using this integrator.
        Notice also that the TSUNAMI integrator will rescale your system to the centre of mass frame of reference.

    .. warning::
        Considering the way the TSUNAMI python wrapper is implemented, the particle positions and velocities are
        updated in place. So if you store the particle.pos or particle.vel inside your loop in a list, each time
        the integrator is called all the elements in the list are updated. Therefore, you will end with a list of pos
        and vel that are or equal to the positions and velocities updated in the last tsunami call.
        To avoid this issue, save a copy of the arrays in the list. For example

        >>> tstart=0
        >>> tintermediate=[5,10,15]
        >>> tcurrent=0
        >>> pos_list=[]
        >>> vel_list=[]
        >>> for t in tintermediate:
        >>>     tstep=t-tcurrent
        >>>     if tstep<=: continue # continue means go to the next step (i.e. next t in the array)
        >>>
        >>>     particles, efftime,_,_,_=integrator_tsunami(particles,tstep)
        >>>
        >>>     # Save the particles positions and velocities
        >>>     pos_list.append(particles.pos.copy())
        >>>     vel_list.append(particles.vel.copy())
        >>>
        >>>     # Here we can save stuff, plot stuff, etc.
        >>>     tcurrent=tcurrent+efftime


    :param particles: Instance of the class :class:`~fireworks.particles.Particles`
    :param tstep: final time of the current integration
    :param acceleration_estimator: Not used
    :param softening: Not used
    :param external_accelerations: Not used
    :return: A tuple with 5 elements:

        - The updated particles instance
        - tstep, the effective timestep evolved in the simulation, it wont'be exactly the one in input
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
