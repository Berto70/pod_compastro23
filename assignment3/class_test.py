#import fireworks
from fireworks.nbodylib import dynamics as dyn
from fireworks.nbodylib import integrators as intg
from fireworks import ic
import matplotlib.pyplot as plt

from evolver import evolve

from animate_fireworks_class import AnimateFireworks

import numpy as np

"""
particles = ic.ic_two_body(1,1,1,0)
(acc_list, potential_list, positions, velocities, energy,init_energy,jerk_) = evolve(particles, intg.integrator_euler)

print("finished simulation")
"""

positions = np.load("positions.npy")

af = AnimateFireworks(positions=positions,xlim=(-1,1),ylim=(-1,1),colors=['r','b'],labels=['1','2'],filename='ani_class_test.gif')


af.debug()