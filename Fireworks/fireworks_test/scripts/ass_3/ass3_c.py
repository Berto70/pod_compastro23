import numpy as np
import fireworks.ic as fic
import matplotlib.pyplot as plt
from fireworks.particles import Particles
import fireworks.nbodylib.dynamics as fdyn
import fireworks.nbodylib.integrators as fint
import fireworks.nbodylib.timesteps as fts

# Initialize two stars in a circular orbit
mass1 = 15.0
mass2 = 1.0
rp = 0.01
e = 0.7 # Set eccentricity to 0 for a circular orbit
part = fic.ic_two_body(mass1=mass1, mass2=mass2, rp=rp, e=e)
# print(part.pos, part.vel, part.mass)
Etot_0, _, _ = part.Etot()

# Calculate the binary period Tperiod
a = rp / (1 - e)  # Semi-major axis
Tperiod = 2 * np.pi * np.sqrt(a**3 / (mass1 + mass2))
# print("Binary Period Tperiod:", Tperiod)

t = 0.
tstep = 0.0001
N_end = 10

pos_i = []
vel_i = []
acc_i = []
mass_i = []
Etot_i = []


while t < N_end*Tperiod:
    t += tstep
    # tstep = 0.0001
    part, _, acc, _, _ = fint.integrator_heun(part, tstep=tstep, acceleration_estimator=fdyn.acceleration_direct_vectorized)
    pos_i.append(part.pos)
    vel_i.append(part.vel)
    mass_i.append(part.mass)
    acc_i.append(acc)

    Etot_j, _, _ = part.Etot()
    Etot_i.append(Etot_j)

    # t += tstep
    # tstep = fts.euler_timestep(part, eta=0.0001, acc = acc)

pos_i = np.array(pos_i)
vel_i = np.array(vel_i)
acc_i = np.array(acc_i)
mass_i = np.array(mass_i)
Etot_i = np.array(Etot_i)

np.save('./fireworks_test/data/ass_3/pos_i.npy', pos_i)
np.save('./fireworks_test/data/ass_3/vel_i.npy', vel_i)
np.save('./fireworks_test/data/ass_3/acc_i.npy', acc_i)
np.save('./fireworks_test/data/ass_3/mass_i.npy', mass_i)
np.save('./fireworks_test/data/ass_3/Etot_i.npy', Etot_i)

ic_param = np.array([mass1, mass2, rp, e, a, Etot_0, Tperiod, tstep, N_end])
np.savetxt('./fireworks_test/data/ass_3/ic_param.txt', ic_param)

print('pos_i', pos_i.shape)
print('vel_i', vel_i.shape)

# peri = np.sqrt(pos_i[:, :, 0]**2 + pos_i[:, :, 1]**2).min(axis=1)
# apo = np.sqrt(pos_i[:, :, 0]**2 + pos_i[:, :, 1]**2).max(axis=1)

# np.save('./fireworks_test/data/ass_3/periastron.npy', peri)
# np.save('./fireworks_test/data/ass_3/apoastron.npy', apo)

# print('peri', peri.shape)
# print('apo', apo.shape)
