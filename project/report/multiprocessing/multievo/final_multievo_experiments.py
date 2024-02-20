"""
Evolve multiple integrators in parallel and compare the results with serial evolution.
This script is needed to save timing relative to different configurations of the evolutions.
It will be used for creating data of:


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



def simulate(int_part,tstep=0.01,total_time = 10.0):
   
   integrator, particles = int_part
   N_particles = len(particles)
   
   integrator_name = integrator.__name__
   print("integrator_name: ", integrator_name)
   
   acc_list       = np.array([])
   pos_list       = np.array([])
   vel_list       = np.array([])
   kinetic_list   = np.array([])
   potential_list = np.array([])
   energy_list    = np.array([])
   
   for _ in range(int(total_time/tstep)):
        particles, tstep, acc, jerk, _ = integrator(particles=particles, 
                                                    tstep=tstep, 
                                                    acceleration_estimator=dyn.acceleration_direct_vectorized,
                                                    softening=0.1
                                                    )
     
     
        acc_list = np.append(acc_list, acc)
        pos_list = np.append(pos_list, particles.pos)
        vel_list = np.append(vel_list, particles.vel)

        kinetic_list   = np.append(kinetic_list, particles.Ekin())
        potential_list = np.append(potential_list, particles.Epot(softening=0.1))
        energy_list    = np.append(energy_list, particles.Etot(softening=0.1))


   acc_list = acc_list.reshape(int(total_time/tstep), N_particles, 3)
   pos_list = pos_list.reshape(int(total_time/tstep), N_particles, 3)
   vel_list = vel_list.reshape(int(total_time/tstep), N_particles, 3)
   
   return {"integrator_name": integrator_name,"acc_list": acc_list, "pos_list": pos_list, "vel_list": vel_list, "energy_list": energy_list}
      


#
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


# Plot for comparing multiple integrators 
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
    filename = "diagnostic_plot.jpg"
    counter = 0 
    while os.path.exists(filename):
        counter += 1
        filename = f"diagnostic_plot_{counter}.jpg"
    print("saving plot to: ", filename)
   
    plt.savefig(f"{filename}")
    print("plot saved.")
    plt.close(fig)  # Close the figure to prevent it from being displayed




def parallel_evo(integrators,particles):
    
    #### MULTIPROCESSING ####
    # define the number of processes
    #N_CORES = multiprocessing.cpu_count() # in my case 4 cores
    #N_INTEGRATORS = len(integrators)
    
    # create a pool of processes
    pool = Pool()


    # submit multiple instances of the function full_evo 
    # - starmap_async: allows to run the processes with a (iterable) list of arguments
    # - map_async    : is a similar function, supporting a single argument

    future_results = pool.map_async(simulate, [(integrator,particles) for integrator in integrators])

    # to get the results all processes must have been completed
    # the get() function is therefore _blocking_ (equivalent to join) 
    results = future_results.get()
  

    # close the pool
    # Warning multiprocessing.pool objects have internal resources that need to be properly managed 
    # (like any other resource) by using the pool as a context manager or by calling close() and terminate() manually. Failure to do this can lead to the process hanging on finalization.
    pool.close()

    return results


def main(n_particles=2, n_simulations=1 ):
    print("Starting main")
    print("n_particles: ", n_particles)
    print("n_simulations: ", n_simulations)

    
    
    particles = ic_random_uniform(n_particles, [1,3],[1,3],[1,3])
    
    """
    #Here I will parallelize the same simulation with the same integrators. 

    integrators = [intg.integrator_euler,
                    intg.integrator_hermite,
                    intg.integrator_leapfrog,
                    intg.integrator_heun,
                    intg.integrator_rk4,
                    ]
    """
    # Using only leapfrog integrator
    integrators = [intg.integrator_leapfrog for _ in range(n_simulations)]

    # MULTIPROCESSING
    start_mp = time.time()
    results = parallel_evo(integrators,particles)
    end_mp = time.time()

    parallel_time = end_mp - start_mp
    print("Multiprocessing time: ", parallel_time)
    
        
    # Serial
    start = time.time()
    results_serial = [simulate((integrator,particles)) for integrator in integrators]
    end = time.time()
    serial_time = end - start
    print("Serial time: ",serial_time)

    

    
    save_me = {"n_particles": n_particles,
               "n_simulations": n_simulations,
               "integrators": integrators[0].__name__,
                "parallel_time":parallel_time, 
                "serial_time": serial_time,
                }
    """
    Uncomment the following lines to save the results to a csv file
    # Convert the save_me dictionary to a DataFrame
    #df = pd.DataFrame([save_me])
    #df.to_csv("parallel_vs_serial_ONETSTEP.csv", mode='a',header=False)
    """   
    print("#########\n")
    """
    Uncomment this to plot the comparison between parallel and serial

    
    results_dict = {result["integrator_name"]: result for result in results}
    results_dict_serial = {result["integrator_name"]: result for result in results_serial}


    plot_comparison("pos_list", results_dict, results_dict_serial)
    """
    
if __name__ == "__main__":
    import sys
    n_particles = int(sys.argv[1])
    n_simulations = int(sys.argv[2])
    #std_numpy = bool(sys.argv[3])
    main(n_particles=n_particles, n_simulations=n_simulations)
