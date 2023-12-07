import numpy as np
import pandas as pd
from tqdm import tqdm
import fireworks.ic as fic
import matplotlib.pyplot as plt
from fireworks.particles import Particles
import fireworks.nbodylib.dynamics as fdyn
import fireworks.nbodylib.integrators as fint
import fireworks.nbodylib.timesteps as fts

# Initialize two stars in a circular orbit
mass1 = 8.0
mass2 = 2.0
rp = 1.0
e = 0.0 # Set eccentricity to 0 for a circular orbit
part = fic.ic_two_body(mass1=mass1, mass2=mass2, rp=rp, e=e)

Etot_0, _, _ = part.Etot()

# Calculate the binary period Tperiod
a = rp / (1 - e)  # Semi-major axis
Tperiod = 2 * np.pi * np.sqrt(a**3 / (mass1 + mass2))

N_end = 10 # -> N_end*Tperiod

# config file
ic_param = np.array([mass1, mass2, rp, e, a, Etot_0, Tperiod, N_end])
np.savetxt('./fireworks_test/data/ass_3/ic_param_all.txt', ic_param)

#define number of time steps per time increment
time_increments = np.array([0.00001, 0.0001, 0.001])
n_ts = np.floor(N_end*Tperiod/time_increments)

integrator_dict = {'Euler_base': fint.integrator_template, 
                   'Euler_modified': fint.integrator_euler,
                   'Hermite': fint.integrator_hermite, 
                   'RK2-Heun': fint.integrator_heun, 
                   'Leapfrog': fint.integrator_leapfrog, 
                   'RK4': fint.integrator_rk4 
                    }

for dt in time_increments:
    N_ts = int(np.floor(N_end*Tperiod/dt))
    file_name = './fireworks_test/data/ass_3/dt_'+str(dt)
    data = {}
    for integrator_name, integrator in integrator_dict.items():
        if integrator_name == 'Hermite':
            array = np.zeros(shape=(N_ts, 5))
            part = fic.ic_two_body(mass1=mass1, mass2=mass2, rp=rp, e=e)
            for t_i in tqdm(range(N_ts), desc=str(dt) + ' ' + integrator_name):
                part, _, acc, _, _ = integrator(part,
                                                tstep=dt,
                                                acceleration_estimator=fdyn.acceleration_direct_vectorized)

                Etot_i, _, _ = part.Etot()
                
                array[t_i, :2] = part.pos[0, :2]
                array[t_i, 2:4]= part.pos[1, :2]
                array[t_i, 4]  = Etot_i

                dt = fts.euler_timestep(part, eta=0.001, acc = acc)
                
            data[integrator_name] = array
        else: 
            array = np.zeros(shape=(N_ts, 5))
            part = fic.ic_two_body(mass1=mass1, mass2=mass2, rp=rp, e=e)
            for t_i in tqdm(range(N_ts), desc=str(dt) + ' ' + integrator_name):
                part, _, acc, _, _ = integrator(part,
                                                tstep=dt,
                                                acceleration_estimator=fdyn.acceleration_direct_vectorized)

                Etot_i, _, _ = part.Etot()
                
                array[t_i, :2] = part.pos[0, :2]
                array[t_i, 2:4]= part.pos[1, :2]
                array[t_i, 4]  = Etot_i
                
            data[integrator_name] = array
        
    np.savez(file_name,**data)
            