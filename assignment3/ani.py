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



fig, ax = plt.subplots()

scat = ax.scatter(positions[0,0,0], positions[0,0,1], c="b", s=5, label="Star 1")
ax.legend()

def update(frame):
    # for each frame, update the data stored on each artist.
    x = positions[:frame,0,0]
    y = positions[:frame,0,1]
    # update the scatter plot:
    data = np.stack([x, y]).T
    scat.set_offsets(data)
    return (scat)


ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=30)
plt.show()
