"""
Here a single run is evolved. It is the computation of the acceleration that is parallelized.
Using vectorized acceleration.
"""


from fireworks.ic import ic_two_body as ic_two_body
from fireworks.ic import ic_random_uniform as ic_random_uniform

from fireworks.nbodylib import dynamics as dyn
from fireworks.nbodylib import integrators as intg


import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple
import numpy.typing as npt
from fireworks.particles import Particles

import time
import multiprocessing
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import os 

from numba import njit
import pandas as pd



def acceleration_direct_vectorized(N_particles, pos, mass):
   
    dx = pos[:, 0].reshape(N_particles, 1) - pos[:, 0] #broadcasting of (N,) on (N,1) array, obtain distance along x in an (N,N) matrix
    dy = pos[:, 1].reshape(N_particles, 1) - pos[:, 1] 
    dz = pos[:, 2].reshape(N_particles, 1) - pos[:, 2] 
      
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    r[r==0]=1
    
    dpos = np.concatenate((dx, dy, dz)).reshape((3,N_particles,N_particles)) 


    acc = - (dpos/r**3 @ mass).T
    jerk= None
    pot = None

    return acc, jerk, pot

def parallel_acc(a,b):

    # global particles doesn't work
    global pos
    global N_particles
    global mass
  
    N_subset = abs(b-a)

    # Select particles from a to b to parallelize computation
    # Need to rewrite the function in order to compute quantities of subset of particles wrt all the others
    dx = pos[a:b, 0,np.newaxis] - pos[:, 0] #broadcasting of (N,) on (N,1) array, obtain distance along x in an (N,N) matrix
    dy = pos[a:b, 1,np.newaxis] - pos[:, 1] 
    dz = pos[a:b, 2,np.newaxis] - pos[:, 2] 
      
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    r[r==0]=1
    # New dpos shape is (3,N_subset,N_particles) since 
    # 3 is the number of dimensions, 
    # N_subset is the number of particles in the subset and
    # N_particles is the number of total particles
    # dpos is the distance vector between each particle in the subset and all the others

    dpos = np.concatenate((dx, dy, dz)).reshape((3,N_subset,N_particles)) 
   
    acc = - (dpos/r**3 @ mass).T
    jerk= None
    pot = None

    return acc, jerk, pot


def parallel_integrator(a,b):
    
    global vel
    global pos 
    global tstep
    # global acc 
    # acceleration is needed only to update vel and pos, so it is not needed as a global variable
    # should acc be allocated in a memory specific to the process?
    acc, _ , _ = parallel_acc(a,b) 

    # Euler integration
    vel[a:b] = vel[a:b] + acc * tstep  # Update vel
    pos[a:b] = pos[a:b] + vel[a:b] * tstep  # Update pos

    # no need to update a global acceleration

    # Return the updated particles, the acceleration, jerk (can be None), and potential (can be None)
    return pos[a:b], vel[a:b]


def parallel_evo(N_particles):

    #### MULTIPROCESSING ####
    # define the number of processes
    N_CORES = multiprocessing.cpu_count() # in my case 4 cores
    N_PROCESSES = min(N_CORES, N_particles)
    # create a pool of processes
    pool = Pool(N_PROCESSES) # ThreadPool is faster than simple Pool for I/O bound tasks


    # submit multiple instances of the function full_evo 
    # - starmap_async: allows to run the processes with a (iterable) list of arguments
    # - map_async    : is a similar function, supporting a single argument
    for _ in range(int(total_evo_time/tstep)): 

        if N_particles < N_PROCESSES:
            # 1 process per particle
            future_results = pool.starmap_async(parallel_integrator, 
                                        [(i, (i + 1)) for i in range(N_particles)])
        else:
            # divide in equal part the particles into N_PROCESSES
            future_results = pool.starmap_async(parallel_integrator, 
                                        [(i * N_particles // N_PROCESSES, (i + 1) * N_particles // N_PROCESSES) for i in range(N_PROCESSES)])



    # to get the results all processes must have been completed
    # the get() function is therefore _blocking_ (equivalent to join) 
    results = future_results.get()

    # close the pool
    # Warning multiprocessing.pool objects have internal resources that need to be properly managed 
    # (like any other resource) by using the pool as a context manager or by calling close() and terminate() manually. Failure to do this can lead to the process hanging on finalization.
    pool.close()

    return results


def make_plot(pos_fast,pos_slow):
    # pos_fast.shape = (N_iterations, N_particles, 3)
   
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # all iterations, particle i, x and y
    for i in range(pos_slow.shape[1]):
        ax[0].scatter(pos_slow[:,i,0],pos_slow[:,i,1],s=.5)
        ax[0].set_title("Serial")

    for j in range(pos_fast.shape[1]):
        ax[1].scatter(pos_fast[:,j,0],pos_fast[:,j,1],s=.5)
        ax[1].set_title("Parallel")

    counter = 0
    filename = "diagnostics.jpg"
    while os.path.exists(filename):
        counter += 1
        filename = f"diagnostics{counter}.jpg"
    plt.savefig(f"{filename}")
        



def main(n_particles):

    global pos
    global vel
    global mass
    global N_particles
    global tstep
    global total_evo_time
    
    #particles = ic_two_body(1,1,1,0)
    particles = ic_random_uniform(n_particles, [0,3],[0,3],[0,3])
    pos = particles.pos
    vel = particles.vel
    mass = particles.mass
    N_particles = len(particles)
    tstep = 0.01
    total_evo_time = tstep # *10
    print(particles)

    start_parallel = time.time()
    results = parallel_evo(N_particles)
    end_parallel = time.time()

    print(f"Parallel evolution took {end_parallel - start_parallel} seconds")

    # Now let's try to do the same with the serial version
    
    start_serial = time.time()

    positions_slow = []
    for _ in range(int(total_evo_time/tstep)):

        particles, tstep, acc, jerk, potential = intg.integrator_euler(particles=particles, tstep=tstep, acceleration_estimator= dyn.acceleration_direct_vectorized,softening="Dehnen")
        positions_slow.append(particles.pos)

    end_serial = time.time()

    print(f"Serial evolution took {end_serial - start_serial} seconds")
  

    parallel_time = end_parallel - start_parallel
    serial_time = end_serial - start_serial
    
    
    save_me = {"n_particles": n_particles,
                "parallel_time": parallel_time,
                "serial_time": serial_time,
                "Pool": True,
                "ThreadPool": False}
            

    # Convert the save_me dictionary to a DataFrame
    df = pd.DataFrame([save_me])
    df.to_csv("POOL_single_evo_parallel_computation.csv", mode='a',header=False)
       
    print("#########\n")

    #make_plot(np.array(positions),np.array(positions_slow))


if __name__=="__main__":
    import sys
    n_particles = int(sys.argv[1])
    main(n_particles)