"""
====================================================================================================================
Collection of functions to estimate the Gravitational forces and accelerations (:mod:`fireworks.nbodylib.dynamics`)
====================================================================================================================

This module contains a collection of functions to estimate acceleration due to
gravitational  forces.

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

try:
    import pyfalcon
    pyfalcon_load=True
except:
    pyfalcon_load=False

def acceleration_estimate_template(particles: Particles, softening: float =0.) \
        -> Tuple[npt.NDArray[np.float64],Optional[npt.NDArray[np.float64]],Optional[npt.NDArray[np.float64]]]:
    """
    This an empty function that can be used as a basic template for
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


def acceleration_direct(particles: Particles, softening: float =0.) \
        -> Tuple[npt.NDArray[np.float64],Optional[npt.NDArray[np.float64]],Optional[npt.NDArray[np.float64]]]:
  
  
  #acc  = np.zeros(len(particles))
  jerk = None
  pot = None

  pos  = particles.pos
  #vel  = particles.vel
  mass = particles.mass
  N    = len(particles) # ? could be 

  # Diego: sposto qui acc e lo rendo una matrice, cosi acc[0,:] ax,ay,az della particella 0 
  acc  = np.zeros([N,3])

  # Idea: put condition : if softening!= 0 direct brute estimate, otherwise use softening

  # Using direct force estimate applcation2 - see slides Lecture 3 p.16


  def acc_2body(position_1,position_2,mass_2):
      
      """
      Implements definition of acceleration for two bodies i,j
    
      This is used in the following for loop
      """
      # Cartesian component of the i,j particles distance
      dx = position_1[0] - position_2[0]
      dy = position_1[1] - position_2[1]
      dz = position_1[2] - position_2[2]
      

      # Distance module
      r = np.sqrt(dx**2 + dy**2 + dz**2)

      # Cartesian component of the i,j force

      ## Diego: question -> shouldn't acceleration be divided my mass1?
      acceleration = np.zeros(3)
      acceleration[0] = mass_2 * dx / r**3
      acceleration[1] = mass_2 * dy / r**3
      acceleration[2] = mass_2 * dz / r**3

      return acceleration
  
  def direct_acc_no_softening(mass=mass): 
        
        for i in range(N-1):
            for j in range(i+1,N):
                # Compute relative acceleration given
                # position of particle i and j
                mass_1 = mass[i]
                mass_2 = mass[j]
                acc_ij = acc_2body(position_1=pos[i,:],position_2=pos[j,:],mass_2=mass_2)

                # Update array with accelerations
                acc[i,:] += acc_ij
                acc[j,:] -= mass_1 * acc_ij / mass_2 # because acc_2nbody already multiply by m[j]
        
        

  if softening == 0.:
      # If no softening compute acceleration values
      direct_acc_no_softening()

  else: print("non ho ancora implementato la funzione di softening, sorry")


  return (acc,jerk,pot)


def acceleration_direct_vectorised(particles: Particles, softening: float =0.) \
        -> Tuple[npt.NDArray[np.float64],Optional[npt.NDArray[np.float64]],Optional[npt.NDArray[np.float64]]]:
    
    #acc  = np.zeros(len(particles))
    jerk = None
    pot = None

    pos  = particles.pos
    #vel  = particles.vel
    mass = particles.mass
    #N    = len(particles)

    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]

    # Coordinate-wise distance for every particle.
    # dx[0,:] will be an array related to particle i = 0 where each position j is the diffrence x_0 - x_j
    # For example dx[0,3] is the difference of the x coordinate of particle 0 with particle 3 (x_0 - x_3)
    dx = x.T - x 
    dy = y.T - y
    dz = z.T - z 
    
    # increments is a data-cube of shape 3xNxN containing all coordinate-distances 
    # It contains the results of r_i - r_j (see equation at p.13 in "Lecture3 - Forces" slides)
    increments = np.array(dx,dy,dz)
    
    # To compute |r_i - r_j|**3 it is necessary to take the norm "channel wise", 
    # i.e. computed on the first axis (from left) of increments array. 
    # increments[:,0,1] = [x_0 - x_1, y_0-y_1, z_0-z_1] 
    distance = np.linalg.norm(increments,axis=0)
    
    # Now acceleration for every particle can be computed.
    # Sum is run over rows; the result will be a NxNx1 matrix,
    # where the first axis represents the coordinates (x,y,z) and the second axis is the particle.
    # acceleration_matrix[:,0,1] = [a_x,a_y,a_z] of particle 0 .
    acceleration_matrix = - np.sum(mass * increments / distance**3, axis=1)





