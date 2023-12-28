# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tqdm import tqdm
import fireworks.ic as fic

from fireworks.particles import Particles
import fireworks.nbodylib.dynamics as fdyn
import fireworks.nbodylib.integrators as fint
import fireworks.nbodylib.timesteps as fts

import warnings
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

np.random.seed(9725)

path = "/home/bertinelli/pod_compastro23/Fireworks/fireworks_test"

## TSUNAMI TRUE/FALSE CONDITION ##
## TWO/NBODY TRUE/FALSE CONDITION ##
tsunami_true = False
two_body = True

## INITIAL CONDITIONS

if two_body == True:
    ## TWO-BODY PROBLEM ##
    # Initialize two body system

    mass1 = 8
    mass2 = 2
    rp = 0.1
    e = 0.9 # 0.0 for circular orbit
    part = fic.ic_two_body(mass1=mass1, mass2=mass2, rp=rp, e=e) # particles initialization
    part.pos = part.pos - part.com_pos() # correcting iniziatial position by C.O.M
    # print(part.pos, part.vel, part.mass)
    Etot_0, _, _ = part.Etot() # total energy of the system

    # Calculate the binary period Tperiod
    a = rp / (1 - e)  # Semi-major axis
    Tperiod = 2 * np.pi * np.sqrt(a**3 / (mass1 + mass2))

else:
    ## THREE-BODY PROBLEM ##

    position = np.array([[1,3,0],
                         [-2,-1,0],
                         [1,-1,0]])

    vel = np.array([[0,0,0],
                    [0,0,0],
                    [0,0,0]])

    mass = np.array([3,4,5])

    # Create instances of the particles
    part = Particles(position, vel, mass)
    Etot_0, _, _ = part.Etot()

## INTEGRATION 

if tsunami_true == True: ## TSUNAMI INTEGRATOR ##
    N_end = 10
    tevol = N_end*Tperiod
    time_increments = np.array([0.00001, 0.0001, 0.001])

    ic_param = np.array([mass1, mass2, rp, e, a, Etot_0, Tperiod, tevol])
    np.savetxt(path + '/data/ass_3/ic_param_tsu.txt', ic_param) # saving initial conditions
    
    data = {} # empty dict for data storage
    file_name = path + '/data/ass_3/data_tusnami_e%.2f'%(e)


    for dt in time_increments:
        tstart = 0
        N_ts = int(np.floor(tevol/dt))
        # nsteps = int(np.floor(tevol/dt))
        tintermediate=np.linspace(0+0.00001, tevol, N_ts)

        tcurrent=0

        array = np.zeros(shape=(N_ts, 6))

        # pbar = tqdm(total=len(tintermediate), desc=str(dt) + ' ' + 'tsunami')
        part = fic.ic_two_body(mass1=mass1, mass2=mass2, rp=rp, e=e) # re-init of Particles
        part.pos = part.pos - part.com_pos() # correcting iniziatial position by C.O.M
        for t_i, t in zip(range(N_ts), tintermediate):

            tstep = t-tcurrent
            if tstep<=0: continue

            part, efftime, _, _, _ = fint.integrator_tsunami(part, tstep)

            # Here we can save stuff, plot stuff, etc.
            Etot_i, _, _ = part.Etot()

            array[t_i, :2] = part.pos[0, :2].copy()
            array[t_i, 2:4]= part.pos[1, :2].copy()
            array[t_i, 4]  = Etot_i.copy()
            array[t_i, 5]  = efftime

            # pbar.update(1)

            tcurrent += efftime

        array = array[array[:,5] != 0] # discard entries = 0
        data[str(dt)] = array
    np.savez(file_name, **data)


else: ## OTHER INTEGRATORS ##

    N_end = 10 # -> N_end*Tperiod

    #define number of time steps per time increment
    time_increments = np.array([0.0001, 0.001, 0.01])
    # n_ts = np.floor(N_end*Tperiod/time_increments)

    # config file
    ic_param = np.array([mass1, mass2, rp, e, a, Etot_0, Tperiod, N_end])
    np.savetxt(path + '/data/ass_3/ic_param_all'+'_e_'+str(e)+'_rp_'+str(rp)+'.txt', ic_param)

    integrator_dict = {'Euler_base': fint.integrator_template, 
                    'Euler_modified': fint.integrator_euler,
                    'Hermite': fint.integrator_hermite, 
                    'RK2-Heun': fint.integrator_heun, 
                    'Leapfrog': fint.integrator_leapfrog, 
                    'RK4': fint.integrator_rk4 
                        }

    for dt in time_increments:
        N_ts = int(np.floor(N_end*Tperiod/dt))
        file_name = path + '/data/ass_3/dt_'+str(dt)+'_e_'+str(e)+'_rp_'+str(rp)
        data = {}
        for integrator_name, integrator in integrator_dict.items():
            tot_time = 0 # init flags count to 0
            N_ts_cum = 0

            if integrator_name == 'Hermite': # alone cause it requires jerk
                
                array = np.zeros(shape=(N_ts, 6))
                part = fic.ic_two_body(mass1=mass1, mass2=mass2, rp=rp, e=e)
                part.pos = part.pos - part.com_pos()
                dt_copy = dt.copy()
                for t_i in range(N_ts):
                # for t_i in tqdm(range(N_ts), desc=str(dt_copy) + ' ' + integrator_name):
                    part, dt_copy, acc, jerk, _ = integrator(part,
                                                    tstep=dt_copy,
                                                    acceleration_estimator=fdyn.acceleration_direct_vectorized, args={'return_jerk': True,
                                                                                                                      'softening_type': 'Plummer'})

                    Etot_i, _, _ = part.Etot()
                    
                    array[t_i, :2] = part.pos[0, :2]
                    array[t_i, 2:4]= part.pos[1, :2]
                    array[t_i, 4]  = Etot_i
                    array[t_i, 5]  = dt_copy

                    tot_time += dt_copy
                    N_ts_cum += 1

                    # break flags
                    if tot_time >= N_end*Tperiod:
                        print('Exceeded total time')
                        break
                    elif N_ts_cum >= 10*N_ts:
                        print('Exceeded number of time steps')
                        break
                    
                data[integrator_name] = array
            else: 
                array = np.zeros(shape=(N_ts, 6))
                part = fic.ic_two_body(mass1=mass1, mass2=mass2, rp=rp, e=e)
                part.pos = part.pos - part.com_pos()
                dt_copy = dt.copy()
                # for t_i in range(N_ts):
                for t_i in tqdm(range(N_ts), desc=str(dt_copy) + ' ' + integrator_name):
                    part, dt_copy, acc, _, _ = integrator(part,
                                                    tstep=dt_copy,
                                                    acceleration_estimator=fdyn.acceleration_direct_vectorized, args={
                                                                                                                      'softening_type': 'Plummer'})

                    Etot_i, _, _ = part.Etot()
                    
                    array[t_i, :2] = part.pos[0, :2]
                    array[t_i, 2:4]= part.pos[1, :2]
                    array[t_i, 4]  = Etot_i
                    array[t_i, 5]  = dt_copy

                    tot_time += dt_copy
                    N_ts_cum += 1

                    if tot_time >= N_end*Tperiod:
                        print('Exceeded time limit')
                        break
                    elif N_ts_cum >= 10*N_ts:
                        print('Exceeded number of time steps')
                        break
                    
                data[integrator_name] = array
            
        np.savez(file_name,**data)
            
