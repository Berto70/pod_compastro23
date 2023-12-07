import numpy as np
import pandas as pd
from tqdm import tqdm
import fireworks.ic as fic
import matplotlib.pyplot as plt
from evolver import evolve

import matplotlib.animation as animation
from fireworks.particles import Particles
import fireworks.nbodylib.dynamics as fdyn
import fireworks.nbodylib.integrators as fint
import fireworks.nbodylib.timesteps as fts



mass1 = 8.0
mass2 = 2.0
rp = 1.0
e = 0.0 # Set eccentricity to 0 for a circular orbit
part = fic.ic_two_body(mass1=mass1, mass2=mass2, rp=rp, e=e)

# Calculate the binary period Tperiod
a = rp / (1 - e)  # Semi-major axis
Tperiod = 2 * np.pi * np.sqrt(a**3 / (mass1 + mass2))

N_end = 10 # -> N_end*Tperiod

positions, velocities, energy = evolve(part, fint.integrator_leapfrog, Tperiod, N_end, tstep = 0.0001)

print("Simulation Done")


fig, ax = plt.subplots()
# ax.set_xlim((-1,1))
# ax.set_ylim((-1,1))
#scat = ax.scatter(positions[0,0,0], positions[0,0,1], c="b", s=5, label="Star 1")


def update(frame, positions):
    # for each frame, update the data stored on each artist.
    x1 = positions[:frame,0,0]
    x2 = positions[:frame,1,0]
    y1 = positions[:frame,0,1]
    y2 = positions[:frame,1,1]
    # update the scatter plot:
    ax.plot(x1, y1, color="tab:blue", linewidth=5, label="Body 1")
    ax.plot(x2, y2, color="tab:orange", linewidth=5, label="Body 2")
    

print("Starting Animation")
gif = animation.FuncAnimation(fig=fig, func=update, frames=np.arange(0,len(positions)-1, 1000), fargs=(positions,))

gif.save("/home/bertinelli/pod_compastro23/assignment_3/animations/two_bodies_test.gif")

print("Animation Saved")
