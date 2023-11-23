import numpy as np
import pandas as pd
import fireworks.ic as fic
import matplotlib.pyplot as plt
from fireworks.particles import Particles
import fireworks.nbodylib.dynamics as fdyn
import fireworks.nbodylib.integrators as fint

# Initialize two stars in a circular orbit
mass1 = 2.0
mass2 = 1.0
rp = 2.0
e = 0.0  # Set eccentricity to 0 for a circular orbit
part = fic.ic_two_body(mass1=mass1, mass2=mass2, rp=rp, e=e)
# print(part.pos, part.vel, part.mass)
Etot_0, _, _ = part.Etot()

# Calculate the binary period Tperiod
a = rp / (1 - e)  # Semi-major axis
Tperiod = 2 * np.pi * np.sqrt(a**3 / (mass1 + mass2))
# print("Binary Period Tperiod:", Tperiod)

ic_param = np.array([mass1, mass2, rp, e, a, Etot_0, Tperiod])
np.savetxt('./fireworks_test/data/ass_3/ic_param.txt', ic_param)

pos_i = np.asarray((2, 3))
vel_i = np.asarray(part.vel)
acc_i = np.asarray(part.vel)
mass_i = []
Etot_i = []

integrator_dict = {'Euler_base': fint.integrator_template,
                'Euler_modified': fint.integrator_euler,
                'RK2-Heun': fint.integrator_heun,
                'Leapfrog': fint.integrator_leapfrog,
                'RK4': fint.integrator_rk4}

tstep_list = [1, 0.01]

for integrator_name, integrator in integrator_dict.items():
    for tstep in tstep_list:

        pos_ij = np.asarray(part.pos)
        vel_ij = np.asarray(part.vel)
        acc_ij = np.asarray(part.vel)
        mass_ij = []
        Etot_ij = []

        t = 0.

        while t < 10*Tperiod:
            part, _, acc, _, _ = integrator(part, tstep=tstep, acceleration_estimator=fdyn.acceleration_pyfalcon)
            pos_ij = np.vstack(part.pos)
            vel_ij = np.vstack(part.vel)
            mass_ij.append(part.mass)
            acc_ij = np.vstack(acc)

            Etot_j, _, _ = part.Etot()
            Etot_ij.append(Etot_j)

            t += tstep
        pos_i = np.vstack(pos_ij)
        vel_i = np.vstack(vel_ij)
        mass_i.append(mass_ij)
        acc_i = np.vstack(acc_ij)
        Etot_i.append(Etot_ij)


# pos_i = np.array(pos_i)
# vel_i = np.array(vel_i)
# acc_i = np.array(acc_i)
# mass_i = np.array(mass_i)
# Etot_i = np.array(Etot_i)

np.save('./fireworks_test/data/ass_3/pos_i.npy', pos_i)
np.save('./fireworks_test/data/ass_3/vel_i.npy', vel_i)
np.save('./fireworks_test/data/ass_3/acc_i.npy', acc_i)
np.save('./fireworks_test/data/ass_3/mass_i.npy', mass_i)
np.save('./fireworks_test/data/ass_3/Etot_i.npy', Etot_i)

# print('pos_i', pos_i.shape)
# print('vel_i', vel_i.shape)

# peri = np.sqrt(pos_i[:, :, 0]**2 + pos_i[:, :, 1]**2).min(axis=1)
# apo = np.sqrt(pos_i[:, :, 0]**2 + pos_i[:, :, 1]**2).max(axis=1)

# np.save('./fireworks_test/data/ass_3/periastron.npy', peri)
# np.save('./fireworks_test/data/ass_3/apoastron.npy', apo)

# print('peri', peri.shape)
# print('apo', apo.shape)