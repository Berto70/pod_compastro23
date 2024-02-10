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



def acceleration_direct(particles: Particles, softening: float =0., softening_type: str = None, ) \
        -> Tuple[npt.NDArray[np.float64],Optional[npt.NDArray[np.float64]],Optional[npt.NDArray[np.float64]]]:
    
  
    """
    Computes gravitational acceleration between particles using a direct method, considering optional softening.

    This function estimates the gravitational acceleration between particles within a system. If 'softening' is provided as 0, a direct estimate with two nested for loop is used;
    otherwise, the specified 'softening' parameter is utilized.

    :param particles: An instance of the class Particles.
    :param softening: Softening parameter for gravitational calculations. If softening == 0, specifying the softening_type dosen't change anything. 
    :param softening_type: The type of softening to use. Possible choice are 'Plummer' and 'Dehnen'. Default is 'None'. 
    :return: A tuple with 3 elements:

        - acc, Nx3 numpy array storing the acceleration for each particle.
        - jerk, Nx3 numpy array storing the time derivative of the acceleration (can be None).
        - pot, Nx1 numpy array storing the potential at each particle position (can be None).
    """
    # Using direct force estimate applcation2 - see slides Lecture 3 p.16
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
        acceleration = np.zeros(3)
        acceleration[0] = -mass_2 * dx / r**3
        acceleration[1] = -mass_2 * dy / r**3
        acceleration[2] = -mass_2 * dz / r**3

        return acceleration
        
    def acc_2body_Plummer_softening(position_1,position_2,mass_2, softening):
        
        """
        Implements definition of acceleration for two bodies i,j with Plummer softening 
        
        This is used in the following for loop
        """
        # Cartesian component of the i,j particles distance
        dx = position_1[0] - position_2[0]
        dy = position_1[1] - position_2[1]
        dz = position_1[2] - position_2[2]
        

        # Distance module
        r = np.sqrt(dx**2 + dy**2 + dz**2)

        # Cartesian component of the i,j force
        acceleration = np.zeros(3)
        acceleration[0] = -mass_2 * dx / (r**2 + softening**2)**(3/2)
        acceleration[1] = -mass_2 * dy / (r**2 + softening**2)**(3/2)
        acceleration[2] = -mass_2 * dz / (r**2 + softening**2)**(3/2)

        return acceleration
        
    def acc_2body_Dehnen_softening(position_1,position_2,mass_2, softening):
        
        """
        Implements definition of acceleration for two bodies i,j with Dehnen softening
        
        This is used in the following for loop
        """
        # Cartesian component of the i,j particles distance
        dx = position_1[0] - position_2[0]
        dy = position_1[1] - position_2[1]
        dz = position_1[2] - position_2[2]
        

        # Distance module
        r = np.sqrt(dx**2 + dy**2 + dz**2)

        # Cartesian component of the i,j force
        acceleration = np.zeros(3)
        acceleration[0] = -mass_2 * (5*softening**2 + 2*r**2) * dx / (2*(r**2 + softening**2)**(5/2))
        acceleration[1] = -mass_2 * (5*softening**2 + 2*r**2) * dy / (2*(r**2 + softening**2)**(5/2))
        acceleration[2] = -mass_2 * (5*softening**2 + 2*r**2) * dz / (2*(r**2 + softening**2)**(5/2))

        return acceleration
        
    jerk = None
    pot = None

    pos  = particles.pos
    mass = particles.mass
    N    = len(particles) 

    # acc[i,:] ax,ay,az of particle i 
    acc  = np.zeros([N,3])

    for i in range(N-1):
        for j in range(i+1,N):
            # Compute relative acceleration given
            # position of particle i and j
            mass_1 = mass[i]
            mass_2 = mass[j]
            if softening_type==None: #if this condition is met, the others are not considered
                acc_ij = acc_2body(position_1=pos[i,:],position_2=pos[j,:],mass_2=mass_2)
            elif softening_type=='Plummer': 
                 acc_ij = acc_2body_Plummer_softening(position_1=pos[i,:],position_2=pos[j,:],mass_2=mass_2, softening=softening)
            elif softening_type=='Dehnen': 
                 acc_ij = acc_2body_Dehnen_softening(position_1=pos[i,:],position_2=pos[j,:],mass_2=mass_2, softening=softening)
            elif softening_type not in ('Plummer', 'Dehnen'):
                 raise Exception("The softening must me either Plummer (default) or Dehnen")
                
            # Update array with accelerations
            acc[i,:] += acc_ij
            acc[j,:] -= mass_1 * acc_ij / mass_2 # because acc_2nbody already multiply by m[j]
        
    return (acc,jerk,pot)


<<<<<<< HEAD
def acceleration_direct_vectorized(particles: Particles, softening: float =0., return_jerk= True) \
=======
def acceleration_direct_vectorized(particles: Particles, softening: float =0., softening_type: str = None, return_jerk= False) \
>>>>>>> origin/main
        -> Tuple[npt.NDArray[np.float64],Optional[npt.NDArray[np.float64]],Optional[npt.NDArray[np.float64]]]:
    """
    This function compute the acceleration in a vectorized fashion using the broadcasting operations of numpy.array.
    If return_flag = True it returns also the jerk, otherwise it is set to None.

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
    :param softening_type: The type of softening to use. Possible choice are 'Plummer' and 'Dehnen'. Default is 'None'. 
    :return: A tuple with 3 elements:

        - acc, Nx3 numpy array storing the acceleration for each particle
        - jerk, Nx3 numpy array storing the time derivative of the acceleration, can be set to None
        - pot, Nx1 numpy array storing the potential at each particle position, can be set to None
    """
    N_particles =  len(particles)
    dx = particles.pos[:, 0].reshape(N_particles, 1) - particles.pos[:, 0] #broadcasting of (N,) on (N,1) array, obtain distance along x in an (N,N) matrix
    dy = particles.pos[:, 1].reshape(N_particles, 1) - particles.pos[:, 1] 
    dz = particles.pos[:, 2].reshape(N_particles, 1) - particles.pos[:, 2] 
      
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    r[r==0]=1
    
    dpos = np.concatenate((dx, dy, dz)).reshape((3,N_particles,N_particles)) 
<<<<<<< HEAD
    acc = - (dpos/r**3 @ particles.mass).T 

    jerk= None
    if return_jerk == True:
        dvx = particles.vel[:, 0].reshape(N_particles, 1) - particles.vel[:, 0]  #broadcasting of (N,) on (N,1) array, obtain distance along x in an (N,N) matrix
        dvy = particles.vel[:, 1].reshape(N_particles, 1) - particles.vel[:, 1]
        dvz = particles.vel[:, 2].reshape(N_particles, 1) - particles.vel[:, 2] 
    
        dvel = np.concatenate((dvx, dvy, dvz)).reshape((3,N_particles,N_particles))
          
        jerk = -((dvel/r**3 - 3*(np.sum((dpos*dvel), axis=0))*dpos/r**5) @ particles.mass).T                
        
=======
    if softening_type==None: #if this condition is met, the others are not considered
        acc = - (dpos/r**3 @ particles.mass).T
        jerk= None
        if return_jerk == True:
            dvx = particles.vel[:, 0].reshape(N_particles, 1) - particles.vel[:, 0]  #broadcasting of (N,) on (N,1) array, obtain distance along x in an (N,N) matrix
            dvy = particles.vel[:, 1].reshape(N_particles, 1) - particles.vel[:, 1]
            dvz = particles.vel[:, 2].reshape(N_particles, 1) - particles.vel[:, 2] 
        
            dvel = np.concatenate((dvx, dvy, dvz)).reshape((3,N_particles,N_particles))
              
            jerk = -((dvel/r**3 - 3*(np.sum((dpos*dvel), axis=0))*dpos/r**5) @ particles.mass).T                

    elif softening_type=='Plummer': 
        acc = - (dpos/(r**2 + softening**2)**(3/2) @ particles.mass).T
        jerk= None
        if return_jerk == True:
            dvx = particles.vel[:, 0].reshape(N_particles, 1) - particles.vel[:, 0]  #broadcasting of (N,) on (N,1) array, obtain distance along x in an (N,N) matrix
            dvy = particles.vel[:, 1].reshape(N_particles, 1) - particles.vel[:, 1]
            dvz = particles.vel[:, 2].reshape(N_particles, 1) - particles.vel[:, 2] 
        
            dvel = np.concatenate((dvx, dvy, dvz)).reshape((3,N_particles,N_particles))
              
            jerk = -((dvel/(r**2 + softening**2)**(3/2) - 3*(np.sum((dpos*dvel), axis=0))*dpos/(r**2 + softening**2)**(5/2)) @ particles.mass).T
            
    elif softening_type=='Dehnen': 
        acc = - (dpos* (5*softening**2 + 2*r**2)/(2*(r**2 + softening**2)**(5/2)) @ particles.mass).T
        jerk= None
        if return_jerk == True:
            dvx = particles.vel[:, 0].reshape(N_particles, 1) - particles.vel[:, 0]  #broadcasting of (N,) on (N,1) array, obtain distance along x in an (N,N) matrix
            dvy = particles.vel[:, 1].reshape(N_particles, 1) - particles.vel[:, 1]
            dvz = particles.vel[:, 2].reshape(N_particles, 1) - particles.vel[:, 2] 
        
            dvel = np.concatenate((dvx, dvy, dvz)).reshape((3,N_particles,N_particles))
              
            jerk = -(  (dpos*(4*np.sum((dpos*dvel), axis=0))/(2*(r**2 + softening**2)**(5/2)) + dvel*(5*softening**2 + 2*r**2)/((r**2 + softening**2)**(5/2)) - 5*dpos*(np.sum((dpos*dvel), axis=0)*(5*softening**2 +2*r**2) )/(2*(r**2 + softening**2)**(7/2))   ) @ particles.mass).T

    elif softening_type not in ('Plummer', 'Dehnen'):
        raise Exception("The softening must me either Plummer (default) or Dehnen")

     
>>>>>>> origin/main
    pot = None

    return acc, jerk, pot

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

    acc, pot = pyfalcon.gravity(particles.pos,particles.mass,softening,kernel=0)
    jerk = None

    return acc, jerk, pot

<<<<<<< HEAD
"""""

Di seguito la funzione di Diego. Non la usiamo nel pacchetto finale ma mi dispiace cancellarla.


def acceleration_direct_vectorised(particles: Particles, softening: float =0.) \
        -> Tuple[npt.NDArray[np.float64],Optional[npt.NDArray[np.float64]],Optional[npt.NDArray[np.float64]]]:
    
    
    jerk = None
    pot = None

    pos  = particles.pos
    mass = particles.mass
    N    = len(particles)

    # Forcing x,y,z arrays to be 2-d --> x.shape = (N,1): this is important for broadcasting.
    x = pos[:,0].reshape(N,1)
    y = pos[:,1].reshape(N,1)
    z = pos[:,2].reshape(N,1)

    # Coordinate-wise distance for every particle.
    # dx[0,:] will be an array related to particle i = 0 where each position j is the diffrence x_0 - x_j
    # For example dx[0,3] is the difference of the x coordinate of particle 0 with particle 3, i.e. (x_0 - x_3)
    dx = x - x.T 
    dy = y - y.T
    dz = z - z.T
    
    # differentials is a data-cube of shape 3xNxN containing all coordinate-distances.
    # It contains the results of r_i - r_j (see equation at p.13 in "Lecture3 - Forces" slides)
    differentials = np.array([dx,dy,dz])
    
    # To compute |r_i - r_j|**3 it is necessary to take the norm "channel/coordinate wise", 
    # i.e. computed on the first axis (from left) of differentials array. 
    # For example, differentials[:,0,1] = [x_0 - x_1, y_0-y_1, z_0-z_1] 

    distance = np.linalg.norm(differentials,axis=0)

    # distance[i,j] is the distance of particle i from particle j.
    # It is a NxN matrix.
    
    # Now acceleration for every particle can be computed.
    # Sum is run over rows; the result will be a NxNx1 matrix,
    # where the first axis (axis 0) represents the coordinates (x,y,z) and the second axis (axis 1) is the particle.
    # acceleration_matrix[:,0,1] = [a_x,a_y,a_z] of particle 0 .
   
    # Since diagonal of distance matrix is obviously zero (distance of every particle from itself),
    # add .00001 to it, in order to avoid division by 0.
    # To do this properly Softening techniques should be used.
    np.fill_diagonal(distance,np.array([.00001 for _ in np.diag(distance)]))

    # Add one dimension to distance. Important for broadcasting with differentials.
    distance = np.expand_dims(distance,axis=0)

    # Summing over rows, i.e. "collapse columns to each other": axis=2 ! (not 1) 
    acceleration_matrix = - np.sum(mass * differentials / (distance)**3, axis=2)
    
    return (acceleration_matrix.T,jerk,pot)


"""


=======
>>>>>>> origin/main
