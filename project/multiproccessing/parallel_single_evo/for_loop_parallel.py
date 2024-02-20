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

from numba import njit
import os 

import pandas as pd


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

def parallel_acceleration_direct(a,b,softening=0.1): 
    global pos
    global vel
    global mass_2
    
    jerk = None
    pot = None

    
    # mass = particles.mass[a:b]
    N    = len(particles)
    N_SUBSET  = len(pos[a:b]) 

    # acc[i,:] ax,ay,az of particle i 
    acc  = np.zeros([N_SUBSET,3])

    # For all particles in the subset compute acceleration wrt all the particles of the simulation
    for i in range(N_SUBSET): 
       # mass_1 = mass[i]
        for j in range(N-1):
            # Compute relative acceleration given
            # position of particle i and j

           
            acc_ij = acc_2body_Dehnen_softening(position_1=pos[i,:], # using pos (subset)
                                                position_2=pos[j,:], # using particles.pos (all particles)
                                                mass_2=mass[j], 
                                                softening=softening)
                 
            # Update array with accelerations
            acc[i,:] += acc_ij
            #acc[j,:] -= mass_1 * acc_ij / mass_2 # because acc_2nbody already multiply by m[j]
        
    return (acc,jerk,pot)



def parallel_integrator(a,b):
    global pos
    global vel

    acc, _ , _ = parallel_acceleration_direct(a,b) 

    # Euler integration
    vel[a:b] = vel[a:b] + acc * tstep  # Update vel
    pos[a:b] = pos[a:b] + vel[a:b] * tstep  # Update pos


    return pos[a:b], vel[a:b]



def make_plot(pos_fast,pos_slow):
    # pos_fast.shape = (N_iterations, N_particles, 3)


    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # all iterations, particle 0, x and y
    for i in range(pos_slow.shape[1]):
        ax[0].scatter(pos_slow[:,i,0],pos_slow[:,i,1],s=.5)
        ax[0].set_title("Serial")

    for j in range(pos_fast.shape[1]):
        ax[1].scatter(pos_fast[:,j,0],pos_fast[:,j,1],s=.5)
        ax[1].set_title("Parallel")

    counter = 0
    filename = "comparison.pdf"
    while os._exists(filename):
        counter += 1
        filename = f"comparison{counter}.pdf"
    plt.savefig(f"{filename}")
        


        

def main(n_particles):
    global pos
    global vel
    global mass
    global tstep
    global N_particles
    global particles

    particles = ic_random_uniform(n_particles, [0,3],[1,2],[3,4])
    pos = particles.pos
    vel = particles.vel
    mass = particles.mass
    N_particles = len(particles)
    tstep = 0.01

    print(particles)

  
    start_parallel = time.time()

    total_evo_time = tstep
    positions = []

    #### MULTIPROCESSING ####
    # define the number of processes
    N_CORES = multiprocessing.cpu_count() # in my case 4 cores
    
    # how does it know N_particles? Lol
    N_PROCESSES = min(N_CORES, N_particles)

    with Pool(N_PROCESSES) as p:

        positions = []

        if N_particles < N_PROCESSES:
            for _ in range(int(total_evo_time/tstep)):
              
                 future_results = p.starmap_async(parallel_integrator, 
                                            [(i, (i + 1)) for i in range(N_particles)]) 
                 # to get the results all processes must have been completed
                 # the get() function is therefore _blocking_ (equivalent to join)
        

                 results = future_results.get() # I dont want to close the pool... should I?
                 p.close()
                 pos_ = np.concatenate([results[i][0] for i in range(len(results))])
                 vel_ = np.concatenate([results[i][1] for i in range(len(results))])
                 positions.append(pos_)
                        
               
        else:
            for _ in range(int(total_evo_time/tstep)):
            # divide in equal part the particles into N_PROCESSES
                future_results = p.starmap_async(parallel_integrator, 
                                                [(i * N_particles // N_PROCESSES, (i + 1) * N_particles // N_PROCESSES) for i in range(N_PROCESSES)])
                # to get the results all processes must have been completed
                # the get() function is therefore _blocking_ (equivalent to join)
                
                results = future_results.get() # I dont want to close the pool... should I?
             
                pos_ = np.concatenate([results[i][0] for i in range(len(results))])
                vel_ = np.concatenate([results[i][1] for i in range(len(results))])
                positions.append(pos_)
                
        
        end_parallel = time.time()
        
        #positions = np.concatenate(positions)
        #print("positions shape",positions.shape)
      
        print(f"Parallel evolution took {end_parallel - start_parallel} seconds")
        parallel_time = end_parallel - start_parallel
        # close the pool
        p.close()
    
    #time.sleep(5)
    # Now let's try to do the same with the serial version

    start_serial = time.time()

    positions_slow = []
    for _ in range(int(total_evo_time/tstep)):

        acc = intg.integrator_euler(particles=particles, tstep=tstep, acceleration_estimator= dyn.acceleration_direct,softening="Dehnen")
        positions_slow.append(particles.pos)

    end_serial = time.time()

    print(f"Serial evolution took {end_serial - start_serial} seconds")
    serial_time = end_serial - start_serial
    #make_plot(np.array(positions),np.array(positions_slow))
    save_me = {"N_particles":n_particles,"parallel_time": parallel_time, "serial_time":serial_time, "tstep/tot_time":tstep/total_evo_time}
    pd.DataFrame([save_me.values()]).to_csv("for_loop_single_evo_mpx.csv",header=False,mode="a")
    
if __name__ == "__main__":
    import sys
    n_particles = int(sys.argv[1])
    main(n_particles)   
        
            


