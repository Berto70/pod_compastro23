"""
Evolve multiple integrators in parallel and compare the results with serial evolution.
This script is needed to save timing relative to different configurations of the evolutions.

"""

import fireworks
from fireworks.ic import ic_two_body as ic_two_body
from fireworks.ic import ic_random_uniform

from fireworks.nbodylib import dynamics as dyn
from fireworks.nbodylib import integrators as intg

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import multiprocessing
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool

import os 
import time

from numba import njit,prange



@njit
def fast_acceleration_direct_vectorized(in_list):
    pos,N_particles,mass,softening = in_list
   
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





# key is what you want to plot, simulation_data is the output of integration_loop function
def plot_sim(key: str, simulation_data: dict):

    # Get the list of integrators from the simulation_data dictionary
    integrators = list(simulation_data.keys())

    # Create a grid plot with subplots for each integrator
    fig, axs = plt.subplots(len(integrators),1, figsize=(8, 6 * len(integrators)),)

    # Iterate over each integrator and plot pos_list
    for i, integrator in enumerate(integrators):
        data = simulation_data[integrator][key]

        for j in range(data.shape[1]):
            axs[i].scatter(data[:, j, 0], data[:, j, 1], label=f"Body {j}",s=.5)
           # axs[i].plot(data[:, 1, 0], data[:, 1, 1], label="Star 2")
            axs[i].set_title(integrator)
            axs[i].legend()

    # Save the figure to a file
    filename = "parallel_plot.jpg"
    counter = 0 
    while os.path.exists(filename):
        counter += 1
        filename = f"parallel_plot_{counter}.jpg"
    print("saving plot to: ", filename)
    plt.savefig(f"{filename}")
    print("plot saved.")
    plt.close(fig)  # Close the figure to prevent it from being displayed


# key is what you want to plot, simulation_data is the output of integration_loop function
def plot_comparison(key: str, simulation_data_serial: dict, simulation_data_parallel: dict):

    # Get the list of integrators from the simulation_data dictionary
    integrators = list(simulation_data_serial.keys()) # assuming thet're the same for both dictionaries
    
    # Create a grid plot with subplots for each integrator
    fig, axs = plt.subplots(len(integrators),2, figsize=(16, 6 * len(integrators)))

    # Iterate over each integrator and plot pos_list
    for i, integrator in enumerate(integrators):
        data_serial   = simulation_data_serial[integrator][key]
        data_parallel = simulation_data_parallel[integrator][key]

        for j in range(data_serial.shape[1]):
            axs[i,0].scatter(data_serial[:, j, 0], data_serial[:, j, 1], label=f"Body {j}",s=.5)
            axs[i,0].set_title(integrator + " Serial")
            axs[i,0].legend()

            axs[i,1].scatter(data_parallel[:, j, 0], data_parallel[:, j, 1], label=f"Body {j}",s=.5)
            axs[i,1].set_title(integrator + " Parallel")
            axs[i,1].legend()

    # Save the figure to a file
    filename = "comparison_plot.jpg"
    counter = 0 
    while os.path.exists(filename):
        counter += 1
        filename = f"comparison_plot_{counter}.jpg"
    print("saving plot to: ", filename)
    plt.savefig(f"{filename}")
    print("plot saved.")
    plt.close(fig)  # Close the figure to prevent it from being displayed




def parallel_evo(pos,mass,N,softening,n_simulations):
    
    #### MULTIPROCESSING ####
    # define the number of processes
    #N_CORES = multiprocessing.cpu_count() # in my case 4 cores
    #N_INTEGRATORS = len(integrators)
    # start a timer
    #start = time.time()
    
    # create a pool of processes
    pool = Pool()


    # submit multiple instances of the function full_evo 
    # - starmap_async: allows to run the processes with a (iterable) list of arguments
    # - map_async    : is a similar function, supporting a single argument
   
    future_results = pool.map_async(fast_acceleration_direct_vectorized, [(pos,N,mass,softening) for _ in range(n_simulations)])

    # to get the results all processes must have been completed
    # the get() function is therefore _blocking_ (equivalent to join) 
    results = future_results.get()
  

    # close the pool
    # Warning multiprocessing.pool objects have internal resources that need to be properly managed 
    # (like any other resource) by using the pool as a context manager or by calling close() and terminate() manually. Failure to do this can lead to the process hanging on finalization.
    pool.close()

    return results


def main(n_particles=2, n_simulations=1):
    print("Starting main")
    print("n_particles: ", n_particles)
    print("n_simulations: ", n_simulations)

    particles = ic_random_uniform(n_particles, [1,3],[1,3],[1,3])

    pos = particles.pos
    mass = particles.mass
    N = len(particles)
    softening = 0.01

    # compile function
    _ = fast_acceleration_direct_vectorized((pos,N,mass,softening))

    # MULTIPROCESSING
    start_mp = time.time()
    _ = parallel_evo(pos,mass,N,softening,n_simulations)
    end_mp = time.time()

    parallel_time = end_mp - start_mp
    print("Multiprocessing time: ", parallel_time)
        
    # Serial
    start = time.time()
    for i in range(n_simulations):
        _ = fast_acceleration_direct_vectorized((pos,N,mass,softening))
    
    end = time.time()
    serial_time = end - start
    print("Serial time: ",serial_time)

    
    save_me = {"n_particles": n_particles,
               "n_simulations": n_simulations,
               "parallel_time":parallel_time,
               "serial_time": serial_time,
                }
    
    # Convert the save_me dictionary to a DataFrame
    df = pd.DataFrame([save_me])
    df.to_csv("mp_and_njit.csv", mode='a',header=False)
        
    print("#########\n")

    
    
if __name__ == "__main__":
    import sys
    n_particles = int(sys.argv[1])
    n_simulations = int(sys.argv[2])
    #std_numpy = bool(sys.argv[3])
    main(n_particles=n_particles, n_simulations=n_simulations,)
