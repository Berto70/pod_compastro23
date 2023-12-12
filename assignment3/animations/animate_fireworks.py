import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation

import fireworks
from fireworks.nbodylib import dynamics as dyn
from fireworks.nbodylib import integrators as intg
from fireworks import ic
import numpy as np
import matplotlib.pyplot as plt

from evolver import evolve


particles = ic.ic_two_body(1,1,1,0)
(acc_list, potential_list, positions, velocities, energy,init_energy,jerk_) = evolve(particles, intg.integrator_euler)

print("finished simulation")


fig, ax = plt.subplots()
ax.set_xlim((-1,1))
ax.set_ylim((-1,1))
#scat = ax.scatter(positions[0,0,0], positions[0,0,1], c="b", s=5, label="Star 1")


def update(frame,positions):
    # for each frame, update the data stored on each artist.
    x1 = positions[:frame,0,0]
    x2 = positions[:frame,1,0]
    y1 = positions[:frame,0,1]
    y2 = positions[:frame,1,1]
    # update the scatter plot:
    ax.scatter(x1, y1, c="b", s=5, label="Star 1")
    ax.scatter(x2, y2, c="r", s=5, label="Star 2")
    

print("starting ani")
ani = animation.FuncAnimation(fig=fig, func=update, frames=np.arange(0,len(positions)-1,1000),fargs=(positions,))

ani.save("animations/two_bodies.gif")
print("done")
