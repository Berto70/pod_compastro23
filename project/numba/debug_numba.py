from fireworks.ic import ic_two_body as ic_two_body
from fireworks.ic import ic_random_uniform as ic_random_uniform

from fireworks.nbodylib import dynamics as dyn
from fireworks.nbodylib import integrators as intg

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple
import numpy.typing as npt
from fireworks.particles import Particles


import numba
from numba import prange, njit


mass1 = 1.0
mass2 = 1.0
rp = 1.0
e = 0.0

particles = ic_two_body(mass1, mass2, rp, e)

#particles = ic_random_uniform(100,[1,1],[1,10],[0,1])


pos = particles.pos
N_particles = len(particles)    
mass = particles.mass
softening = 1e-3


@njit(parallel=True)
def p_acc_compute(mass,position_1,position_2,softening,i,N):

    """
    Fixing particle i, this computes the acceleration of particle i due to all the other particles
    """
    acc_x = 0
    acc_y = 0
    acc_z = 0

    # deleting prange(i+1,N) for parallization wouldn't compute correctly the ji part 
    for j in prange(N):
        # Compute relative acceleration given
        # position of particle i and j
        mass_2 = mass[j]
        # This may be split into x,y,z for parallelization
        position_2=pos[j,:]
    
        # Cartesian component of the i,j particles distance
        dx = position_1[0] - position_2[0]
        dy = position_1[1] - position_2[1]
        dz = position_1[2] - position_2[2]
        

        # Distance module
        r = np.sqrt(dx**2 + dy**2 + dz**2)

        # Cartesian component of the i,j acceleration
        acceleration = np.zeros(3)
        acceleration[0] = -mass_2 * (5*softening**2 + 2*r**2) * dx / (2*(r**2 + softening**2)**(5/2))
        acceleration[1] = -mass_2 * (5*softening**2 + 2*r**2) * dy / (2*(r**2 + softening**2)**(5/2))
        acceleration[2] = -mass_2 * (5*softening**2 + 2*r**2) * dz / (2*(r**2 + softening**2)**(5/2))

        acc_x += acceleration[0]
        acc_y += acceleration[1]
        acc_z += acceleration[2]

    
    return acc_x,acc_y,acc_z


@njit(parallel=True)
def fast_acceleration_direct(pos,mass,N,softening):
     
    jerk = None
    pot = None
   
    # acc[i,:] ax,ay,az of particle i 
    acc  = np.zeros((N,3))

    # Fix particles i and paralelize over j, since I need the final matrix to be orderered by i
    # ranging over all N, not removing the last particle, since parallization would make it hard to compute the ji part 
    # ie computing for all particles all other particles, without any s      kzort of tricks to make less computations
    for i in range(N):
        
        position_1=pos[i,:]
        # paralellized acc computation
       # acc[i,:] = p_acc_compute(mass,position_1,pos,softening,i,N)
        a = p_acc_compute(mass,position_1,pos,softening,i,N)
        print(a)
                     
    return (acc,jerk,pot)



if __name__ == "__main__":
    
    acc,_,_ = fast_acceleration_direct(pos,mass,N_particles,softening)
    print(acc)
    print("done")

    #plt.plot(acc[:,0])
    #plt.show()