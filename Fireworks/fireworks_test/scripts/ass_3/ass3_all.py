import numpy as np
import pandas as pd
import fireworks.ic as fic
import matplotlib.pyplot as plt
from fireworks.particles import Particles
import fireworks.nbodylib.dynamics as fdyn
import fireworks.nbodylib.integrators as fint

# Initialize two stars in a circular orbit
mass1 = 5.0
mass2 = 1.0
rp = 2.0
e = 0.5 # Set eccentricity to 0 for a circular orbit
part = fic.ic_two_body(mass1=mass1, mass2=mass2, rp=rp, e=e)

Etot_0, _, _ = part.Etot()

# Calculate the binary period Tperiod
a = rp / (1 - e)  # Semi-major axis
Tperiod = 2 * np.pi * np.sqrt(a**3 / (mass1 + mass2))

N_end = 10 # N_end*Tperiod

# config file
ic_param = np.array([mass1, mass2, rp, e, a, Etot_0, Tperiod, N_end])
np.savetxt('./fireworks_test/data/ass_3/ic_param.txt', ic_param)

#define number of time steps per time increment
time_increments = np.array([0.01])
n_ts = np.floor(N_end*Tperiod/time_increments)

integrator_dict = {'RK4': fint.integrator_rk4,
                   'Euler_base': fint.integrator_template,
                   'Leapfrog': fint.integrator_leapfrog}

for dt in time_increments:
    N_ts = int(np.floor(N_end*Tperiod/dt))
    file_name = './fireworks_test/data/ass_3/dt_'+str(dt)
    data = {}
    for integrator_name, integrator in integrator_dict.items():
        array = np.zeros(shape=(N_ts, 5))
        
        for t_i in range(N_ts):
            part, _, acc, _, _ = integrator(part,
                                            tstep=dt,
                                            acceleration_estimator=fdyn.acceleration_direct_vectorised)

            Etot_i, _, _ = part.Etot()
            
            array[t_i, :2] = part.pos[0, :2]
            array[t_i, 2:4]= part.pos[1, :2]
            array[t_i, 4]  = Etot_i
            
        data[integrator_name] = array
        
    np.savez(file_name,**data)
            