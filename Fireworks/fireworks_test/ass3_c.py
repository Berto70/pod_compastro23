import pytest
import numpy as np
import fireworks.ic as fic
from fireworks.particles import Particles

# pos = [-200, 200]
# vel = [-50, 50]
# mass = [0.01, 100]

part = fic.ic_two_body(mass1 = 1., mass2 = 2., rp = 2., e=0.9)

print(part.pos, part.vel, part.mass)