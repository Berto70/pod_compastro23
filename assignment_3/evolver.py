import fireworks
from fireworks.nbodylib import dynamics as dyn
from fireworks.nbodylib import integrators as intg
from fireworks import ic
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def evolve(particles, integrator, Tperiod, N_end, tstep = .001):

    positions = []
    velocities = []
    energy = []

    Etot_0, _, _ = particles.Etot()

    tot_time = 0

    pbar = tqdm(total=N_end*Tperiod/tstep)

    while tot_time < N_end*Tperiod:
        tot_time += tstep

        # Evolve the binary 

        particles, _, _, _, _ = integrator(particles=particles, 
                                        tstep=tstep,
                                        acceleration_estimator=dyn.acceleration_direct_vectorized,
                                        softening=0.)
        E_i, _, _ = particles.Etot()

        positions.append(particles.pos)
        velocities.append(particles.vel)
        energy.append(E_i)

        pbar.update(1)
    
    
    positions = np.array(positions).reshape(len(positions),2,3)
    velocities = np.array(velocities).reshape(len(velocities),2,3)
    energy = np.array(energy)
    return (positions, velocities, energy)
