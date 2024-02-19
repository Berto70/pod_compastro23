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

import time
import pandas as pd



def acceleration_direct_slow(pos,mass,N,softening):
    jerk = None
    pot = None

    # acc[i,:] ax,ay,az of particle i 
    acc  = np.zeros([N,3])

    for i in range(N-1):
        for j in range(i+1,N):
            # Compute relative acceleration given
            # position of particle i and j
            mass_1 = mass[i]
            mass_2 = mass[j]

            # Compute acceleration of particle i due to particle j
            position_1=pos[i,:]
            position_2=pos[j,:]
            
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

            # Update array with accelerations
            acc[i,:] += acceleration
            acc[j,:] -= mass_1 * acceleration / mass_2 # because acc_2nbody already multiply by m[j]
        
    return (acc,jerk,pot)


@njit
def acceleration_direct_fast(pos,mass,N,softening):
    jerk = None
    pot = None

    # acc[i,:] ax,ay,az of particle i 
    #acc  = np.zeros([N,3])
    acc = np.zeros_like(pos)

    for i in range(N-1):
        for j in range(i+1,N):
            # Compute relative acceleration given
            # position of particle i and j
            mass_1 = mass[i]
            mass_2 = mass[j]

            # Compute acceleration of particle i due to particle j
            position_1=pos[i,:]
            position_2=pos[j,:]
            
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

            # Update array with accelerations
            acc[i,:] += acceleration
            acc[j,:] -= mass_1 * acceleration / mass_2 # because acc_2nbody already multiply by m[j]
        
    return (acc,jerk,pot)

def slow_acceleration_direct_vectorized(pos,N_particles,mass,softening):
    dx = pos[:, 0].reshape(N_particles, 1) - pos[:, 0] #broadcasting of (N,) on (N,1) array, obtain distance along x in an (N,N) matrix
    dy = pos[:, 1].reshape(N_particles, 1) - pos[:, 1] 
    dz = pos[:, 2].reshape(N_particles, 1) - pos[:, 2] 
    
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    r[r==0]=1
    
    dpos = np.concatenate((dx, dy, dz)).reshape((3,N_particles,N_particles)) 
    acc = - (dpos* (5*softening**2 + 2*r**2)/(2*(r**2 + softening**2)**(5/2)) @ mass).T
    
    jerk= None 
    pot = None

    return acc, jerk, pot


@njit
def fast_acceleration_direct_vectorized(pos,N_particles,mass,softening):
   
    dx = pos[:, 0].copy().reshape(N_particles, 1) - pos[:, 0] #broadcasting of (N,) on (N,1) array, obtain distance along x in an (N,N) matrix
    dy = pos[:, 1].copy().reshape(N_particles, 1) - pos[:, 1] 
    dz = pos[:, 2].copy().reshape(N_particles, 1) - pos[:, 2] 
      
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    #r[r==0]=1 not supported on numba
    r += np.eye(r.shape[0])
    
    dpos = np.concatenate((dx, dy, dz)).copy().reshape((3,N_particles,N_particles)) 
    acc = - np.sum(dpos* (5*softening**2 + 2*r**2)/(2*(r**2 + softening**2)**(5/2)) * mass,axis=2).T
   
    jerk= None
    pot = None

    return acc, jerk, pot



@njit(parallel=True)
def parallel_acceleration_direct_fast(pos,mass,N,softening):
    jerk = None
    pot = None

    # acc[i,:] ax,ay,az of particle i 
    #acc  = np.zeros([N,3])
    acc = np.zeros_like(pos)

    for i in prange(N-1):
        for j in range(i+1,N):
            # Compute relative acceleration given
            # position of particle i and j
            mass_1 = mass[i]
            mass_2 = mass[j]

            # Compute acceleration of particle i due to particle j
            position_1=pos[i,:]
            position_2=pos[j,:]
            
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

            # Update array with accelerations
            acc[i,:] += acceleration
            acc[j,:] -= mass_1 * acceleration / mass_2 # because acc_2nbody already multiply by m[j]
        
    return (acc,jerk,pot)



def main(n_particles):

    print("running with",n_particles)

    particles = ic_random_uniform(n_particles,[1,3],[1,3],[1,3])
    pos = particles.pos
    vel = particles.vel
    mass = particles.mass
    softening = 0.01
    tstep = 0.01
    N = n_particles # useless, but I mistakenly write N instead of n_particles or viceversa every time

    # Do not integrate, only one tstep
    # numba functions should be called once before measuring their performance, in order for numba to compile them
    _ = acceleration_direct_fast(pos,mass,N,softening)
    _ = fast_acceleration_direct_vectorized(pos,N,mass,softening)
    _ = parallel_acceleration_direct_fast(pos,mass,N,softening)

    ####
    # Begin measurements
    ### 

    # Direct
    
    print("direct slow")
    start_direct_slow = time.time()
    for t in range(100):
        _ = acceleration_direct_slow(pos,mass,N,softening)
    end_direct_slow = time.time()
    
    print("direct fast")
    start_direct_fast = time.time() 
    for t in range(100):
        _ = acceleration_direct_fast(pos,mass,N,softening)
    end_direct_fast = time.time()
    

    # Vectorized
    print("vect slow")
    start_vect_slow = time.time()
    for t in range(100):
        _ = slow_acceleration_direct_vectorized(pos,N,mass,softening)
    end_vect_slow = time.time()

    print("vect fast")
    start_vect_fast = time.time()
    for t in range(100):
        _ = fast_acceleration_direct_vectorized(pos,N,mass,softening)
    end_vect_fast = time.time()

    # Direct parallel
    print("direct parallel")
    start_direct_parallel = time.time()
    for t in range(100):
        _ = parallel_acceleration_direct_fast(pos,mass,N,softening)
    end_direct_parallel = time.time()

    direct_slow = (end_direct_slow - start_direct_slow)/100
    direct_fast = (end_direct_fast - start_direct_fast)/100
    vect_slow = (end_vect_slow - start_vect_slow)/100
    vect_fast = (end_vect_fast - start_vect_fast)/100
    direct_parallel = (end_direct_parallel - start_direct_parallel)/100


    df = pd.DataFrame({"N_particles":[n_particles],
                  "Direct_slow": [direct_slow],
                  "Direct_fast": [direct_fast],
                  "Vectorized_slow": [vect_slow],
                  "Vectorized_fast": [vect_fast],
                  "Direct_parallel": [direct_parallel]})

    df.to_csv("numba_timings.csv",mode="a",header=False)
    

if __name__ == "__main__":
    import sys
    n_particles = int(sys.argv[1])
    main(n_particles)