import fireworks
from fireworks.nbodylib import dynamics as dyn
from fireworks.nbodylib import integrators as intg
from fireworks import ic
import numpy as np
import matplotlib.pyplot as plt


def evolve(particles,integrator):

    period = 10
    tstep = .001

    tot_time = 0 

    acc_list = []
    potential_list = []
    positions = []
    velocities = []
    energy = []


    init_energy = particles.Etot()

    while tot_time < 10*period:
        tot_time +=tstep

        # Evolve the binary 

        (particles, tstep, acc, jerk, potential) = integrator(particles=particles, 
                                                                            tstep=tstep,
                                                                            acceleration_estimator=dyn.acceleration_direct_vectorized,
                                                                            softening=0.)
        
        acc_list.append(acc)
        potential_list.append(potential)
        positions.append(particles.pos)
        velocities.append(particles.vel)
        energy.append(particles.Etot())
        
    acc_list = np.array(acc_list).reshape(len(acc_list),2,3)
    potential_list = np.array(potential_list)
    positions = np.array(positions).reshape(len(positions),2,3)
    velocities = np.array(velocities).reshape(len(velocities),2,3)
    energy = np.array(energy)
    return (acc_list, potential_list, positions, velocities, energy)

