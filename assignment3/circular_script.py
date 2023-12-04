### Import useful libraries


import fireworks
from fireworks.ic import ic_two_body as ic_two_body

from fireworks.nbodylib import dynamics as dyn
from fireworks.nbodylib import integrators as intg

import matplotlib.pyplot as plt
import numpy as np


## Functions needed

def ininitialise(mass1=2, mass2=1, rp=2, e=0):
    particles = ic_two_body(mass1, mass2, rp, e)

    a = rp / (1 - e)  # Semi-major axis
    period = 2 * np.pi * np.sqrt(a**3 / (mass1 + mass2))

    return particles, period


def simulate(integrator,particles,tstep=0.01,total_time = 10):

  

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
                                               softening=0.1,
                                               )
      
      acc_list = np.append(acc_list, acc)
      pos_list = np.append(pos_list, particles.pos)
      vel_list = np.append(vel_list, particles.vel)

      kinetic_list   = np.append(kinetic_list, particles.Ekin())
      potential_list = np.append(potential_list, particles.Epot(softening=0.1))
      energy_list    = np.append(energy_list, particles.Etot(softening=0.1))


   acc_list = acc_list.reshape(int(total_time/tstep), 2, 3)
   pos_list = pos_list.reshape(int(total_time/tstep), 2, 3)
   vel_list = vel_list.reshape(int(total_time/tstep), 2, 3)

   return acc_list, pos_list, vel_list, kinetic_list, potential_list, energy_list
      

# Loop to simulate with different integrators
def integration_loop(period, integrators):


   simulation_data = {}

   #for integrator in [intg.integrator_euler, intg.integrator_leapfrog,intg.integrator_hermite]:
   for integrator in integrators:
      acc_list, pos_list, vel_list, kinetic_list, potential_list, energy_list = simulate(integrator,particles,tstep=.1,total_time = period)

      simulation_data[integrator.__name__] = {"acc_list": acc_list, "pos_list": pos_list, "vel_list": vel_list, "energy_list": energy_list}
      
   return simulation_data


# key is what you want to plot, simulation_data is the output of integration_loop function
def plot_sim(key: str, simulation_data: dict):

    # Get the list of integrators from the simulation_data dictionary
    integrators = list(simulation_data.keys())

    # Create a grid plot with subplots for each integrator
    fig, axs = plt.subplots(len(integrators), 1, figsize=(8, 6 * len(integrators)))

    # Iterate over each integrator and plot pos_list
    for i, integrator in enumerate(integrators):
        data = simulation_data[integrator][key]
        axs[i].plot(data[:, 0, 0], data[:, 0, 1], label="Star 1")
        axs[i].plot(data[:, 1, 0], data[:, 1, 1], label="Star 2")
        axs[i].set_title(integrator)
        axs[i].legend()


    plt.show()

## Use right syntax to make this scirpt run with the values you want

particles, period = ininitialise(mass1=1, mass2=1, rp=1, e=0)

simulation_data = integration_loop(10*period, integrators=[intg.integrator_hermite, intg.integrator_euler])

plot_sim("pos_list", simulation_data)

