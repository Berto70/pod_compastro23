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

pos = particles.pos
vel = particles.vel
N = len(particles)    
mass = particles.mass
softening = 1e-3
tstep = 1e-3

#particles = ic_random_uniform(100,[1,1],[1,10],[0,1])


@njit(parallel=True,fastmath=True)
def acceleration_parallel(position_1,N,mass,i):
    # acceleration of particle i 
    acceleration = np.zeros(3)
    for j in prange(N):
        if i != j: 
            print("indii",i,j)
            # Cartesian component of the i,j particles distance
            position_2 = pos[j] 
            
            mass_2 = mass[j]
            dx = position_1[0] - position_2[0]
            dy = position_1[1] - position_2[1]
            dz = position_1[2] - position_2[2]
            
            # Distance module
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # Cartesian component of the i,j force
            # This shouldn't give problems during parallelization because I'm updating a specific element
            # and the order in j in which this is done is not relevant 
            
            acceleration[0] += -mass_2 * (5*softening**2 + 2*r**2) * dx / (2*(r**2 + softening**2)**(5/2))
            acceleration[1] += -mass_2 * (5*softening**2 + 2*r**2) * dy / (2*(r**2 + softening**2)**(5/2))
            acceleration[2] += -mass_2 * (5*softening**2 + 2*r**2) * dz / (2*(r**2 + softening**2)**(5/2))

    return acceleration


@njit(parallel=True)
def integrator_euler(pos,
                     vel,
                     acc,
                     tstep,
                     jerk=None,
                     potential=None
                     ):
    
    # removing check for external acceleration
    #
    #
    # Euler integration
    vel = vel + acc * tstep  # Update vel
    pos = pos + vel * tstep  # Update pos
    #particles.set_acc(acc)  # Set acceleration

    return (vel,pos)


@njit(parallel=True)
def full_parallel_evo(N,pos,vel,mass,tstep):
    for i in prange(N):
        position_1 = pos[i]
       # acc_i   = acceleration_parallel(position_1,N,mass,i)

        acceleration = np.zeros(3)
        for j in prange(N):
            if i != j: 
                print("indii",i,j)
                # Cartesian component of the i,j particles distance
                position_2 = pos[j] 
                
                mass_2 = mass[j]
                dx = position_1[0] - position_2[0]
                dy = position_1[1] - position_2[1]
                dz = position_1[2] - position_2[2]
                
                # Distance module
                r = np.sqrt(dx**2 + dy**2 + dz**2)
                
                # Cartesian component of the i,j force
                # This shouldn't give problems during parallelization because I'm updating a specific element
                # and the order in j in which this is done is not relevant 
                
                acceleration[0] += -mass_2 * (5*softening**2 + 2*r**2) * dx / (2*(r**2 + softening**2)**(5/2))
                acceleration[1] += -mass_2 * (5*softening**2 + 2*r**2) * dy / (2*(r**2 + softening**2)**(5/2))
                acceleration[2] += -mass_2 * (5*softening**2 + 2*r**2) * dz / (2*(r**2 + softening**2)**(5/2))


        print(f"acc_{i}",acceleration,"\n")


        #vel2,pos2 = integrator_euler(pos[i],vel[i],acceleration,tstep)

        vel2 = vel + acceleration * tstep  # Update vel
        pos2 = pos + vel * tstep

        print(f"new position of particle {i}:",pos2)
        print("\n")
        

if __name__ == "__main__":
    print("ciao")
    full_parallel_evo(N,pos,vel,mass,tstep)
    print("ho fatto")
    full_parallel_evo.parallel_diagnostics()