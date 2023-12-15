import numpy as np
import pandas as pd
from tqdm import tqdm
import fireworks.ic as fic
import matplotlib.pyplot as plt
from fireworks.particles import Particles
import fireworks.nbodylib.dynamics as fdyn
import fireworks.nbodylib.integrators as fint
import fireworks.nbodylib.timesteps as fts

np.random.seed(9725)

path = "/home/bertinelli/pod_compastro23/Fireworks/fireworks_test"

## TSUNAMI TRUE/FALSE CONDITION ##
tsunami_true = False

## TWO-BODY PROBLEM ##
# Initialize two stars in a circular orbit
mass1 = 8
mass2 = 2
rp = 1.
e = 0.0 # Set eccentricity to 0 for a circular orbit
part = fic.ic_two_body(mass1=mass1, mass2=mass2, rp=rp, e=e)
# print(part.pos, part.vel, part.mass)
Etot_0, _, _ = part.Etot()

# Calculate the binary period Tperiod
a = rp / (1 - e)  # Semi-major axis
Tperiod = 2 * np.pi * np.sqrt(a**3 / (mass1 + mass2))

## THREE-BODY PROBLEM ##

position = np.array([[0,0,0],
                         [0.5,0.866,9],
                         [1,0,0]])

vel = np.array([[0,0,0],
                [0,0,0],
                [0,0,0]])

mass = np.array([3,4,5])

# Create instances of the particles
particles = Particles(position, vel, mass)
Etot_0, _, _ = particles.Etot()

if tsunami_true == True: ## TSUNAMI INTEGRATOR ##

    tevol = 65

    tstart=0
    dt = 0.0001
    nsteps = int(np.floor(tevol/dt))
    tintermediate=np.linspace(0+0.00001, tevol, nsteps)

    tcurrent=0

    data = {}
    array = np.zeros(shape=(nsteps, 6))
    file_name = path + '/data/ass_3/dt_' + str(dt) + '_tusnami'

    pbar = tqdm(total=len(tintermediate))

    for t_i in tintermediate:

        tstep = t_i-tcurrent
        if tstep<=0: continue

        particles, efftime, _, _, _ = fint.integrator_tsunami(particles, tstep)

        # Here we can save stuff, plot stuff, etc.
        Etot_i, _, _ = part.Etot()
                
        array[t_i, :2] = part.pos[0, :2]
        array[t_i, 2:4]= part.pos[1, :2]
        array[t_i, 4]  = Etot_i
        array[t_i, 5]  = t_i

        pbar.update(1)

        tcurrent += efftime

    data['tsumani'] = array
    np.savez(file_name,**data)


else: ## OTHER INTEGRATORS ##

    # config file
    ic_param = np.array([mass1, mass2, rp, e, a, Etot_0, Tperiod, N_end])
    np.savetxt(path + '/data/ass_3/ic_param_all.txt', ic_param)

    N_end = 10 # -> N_end*Tperiod

    #define number of time steps per time increment
    time_increments = np.array([0.0001, 0.001, 0.01])
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
        file_name = path + '/data/ass_3/dt_'+str(dt)
        data = {}
        for integrator_name, integrator in integrator_dict.items():
            tot_time = 0
            N_ts_cum = 0
            if integrator_name == 'Hermite':
                array = np.zeros(shape=(N_ts, 5))
                part = fic.ic_two_body(mass1=mass1, mass2=mass2, rp=rp, e=e)
                dt_copy = dt.copy()
                for t_i in tqdm(range(N_ts), desc=str(dt_copy) + ' ' + integrator_name):
                    part, _, acc, _, _ = integrator(part,
                                                    tstep=dt_copy,
                                                    acceleration_estimator=fdyn.acceleration_direct_vectorized, args={'return_jerk': True})

                    Etot_i, _, _ = part.Etot()
                    
                    array[t_i, :2] = part.pos[0, :2]
                    array[t_i, 2:4]= part.pos[1, :2]
                    array[t_i, 4]  = Etot_i


                    dt_copy = fts.euler_timestep(part, eta=0.001, acc = acc)

                    tot_time += dt_copy
                    N_ts_cum += 1

                    if tot_time >= N_end*Tperiod:
                        break
                    elif N_ts_cum >= 10*N_ts:
                        break
                    
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

                    if tot_time >= N_end*Tperiod:
                        break
                    elif N_ts_cum >= 10*N_ts:
                        break
                    
                data[integrator_name] = array
            
        np.savez(file_name,**data)
            