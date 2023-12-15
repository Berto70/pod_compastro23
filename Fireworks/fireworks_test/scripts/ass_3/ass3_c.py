import numpy as np
import fireworks.ic as fic
import matplotlib.pyplot as plt
from fireworks.particles import Particles
from tqdm import tqdm
import fireworks.nbodylib.dynamics as fdyn
import fireworks.nbodylib.integrators as fint
import fireworks.nbodylib.timesteps as fts

np.random.seed(9725)

path = "/home/bertinelli/pod_compastro23/Fireworks/fireworks_test"

# Initialize two stars in a circular orbit
mass1 = 8
mass2 = 2
rp = 1.
e = 0.0 # Set eccentricity to 0 for a circular orbit
part = fic.ic_two_body(mass1=mass1, mass2=mass2, rp=rp, e=e)
# print(part.pos, part.vel, part.mass)
Etot_0, _, _ = part.Etot()

# Calculate the binary period Tperiod
a = rp / (1 - e)  # Semi-major axis
Tperiod = 2 * np.pi * np.sqrt(a**3 / (mass1 + mass2))
# print("Binary Period Tperiod:", Tperiod)

t = 0.
tstep = 0.1
N_end = 10

pos_i = []
vel_i = []
acc_i = []
mass_i = []
Etot_i = []
# tstep_i = []

# position = np.array([[0,0,0],
#                      [0.5,0.866,9],
#                      [1,0,0]])

# vel = np.array([[0,0,0],
#                 [0,0,0],
#                 [0,0,0]])

# mass = np.array([3,4,5])

# # Create instances of the particles
# part = Particles(position, vel, mass)
# Etot_0, _, _ = part.Etot()

## OTHER INTEGRATORS ##

total = N_end*Tperiod/tstep
pbar = tqdm(total=total)

while t < N_end*Tperiod:
    # t += tstep
    part, _, acc, jerk, _ = fint.integrator_rk4(part, tstep, 
                                                         acceleration_estimator=fdyn.acceleration_direct_vectorized, args={'return_jerk': False})
    pos_i.append(part.pos)
    vel_i.append(part.vel)
    mass_i.append(part.mass)
    acc_i.append(acc)

    Etot_j, _, _ = part.Etot()
    Etot_i.append(Etot_j)

    # tstep = fts.adaptive_timestep_jerk(particles=part, eta=10e-4, acc=acc, tmin=0.0001, tmax=0.01)

    tstep = fts.adaptive_timestep(part, integrator=fint.integrator_rk4, int_rank=4, int_args={'particles': part, 'tstep': tstep, 'acceleration_estimator': fdyn.acceleration_direct_vectorized},
                                   predictor=fint.integrator_heun, pred_rank=2, pred_args={'particles': part, 'tstep': tstep, 'acceleration_estimator': fdyn.acceleration_direct_vectorized},
                                   epsilon=10e-9, tmax=0.01, tmin=0.0001)
    
    t += tstep
    pbar.update(1)

# ## TSUNAMI INTEGRATOR ##

# tstart=0
# tintermediate=np.linspace(0+0.00001, 65, 100000)
# tcurrent=0

# pbar = tqdm(total=len(tintermediate))

# for t in tintermediate:

#     tstep = t-tcurrent
#     if tstep<=0: continue

#     particles, efftime, _, _, _ = fint.integrator_tsunami(particles, tstep)

#      # Here we can save stuff, plot stuff, etc.
#     pos_i.append(particles.pos.copy())
#     vel_i.append(particles.vel.copy())
#     mass_i.append(particles.mass.copy())
#     Etot_j, _, _ = particles.Etot()
#     Etot_i.append(Etot_j.copy())
#     tstep_i.append(tstep)

#     pbar.update(1)

#     tcurrent += efftime


pos_i = np.array(pos_i)
vel_i = np.array(vel_i)
acc_i = np.array(acc_i)
mass_i = np.array(mass_i)
Etot_i = np.array(Etot_i)
# tstep_i = np.array(tstep_i)

np.save(path + '/data/ass_3/pos_i.npy', pos_i)
np.save(path + '/data/ass_3/vel_i.npy', vel_i)
np.save(path + '/data/ass_3/acc_i.npy', acc_i)
np.save(path + '/data/ass_3/mass_i.npy', mass_i)
np.save(path + '/data/ass_3/Etot_i.npy', Etot_i)
# np.save(path + '/data/ass_3/tstep_i.npy', tstep_i)

ic_param = np.array([mass1, mass2, rp, e, a, Etot_0, Tperiod, N_end])
np.savetxt(path + '/data/ass_3/ic_param.txt', ic_param)

# print('pos_i', pos_i.shape)
# print('\nvel_i', vel_i.shape)
# print('\ntstep_i', tstep_i.shape)

# # # peri = np.sqrt(pos_i[:, :, 0]**2 + pos_i[:, :, 1]**2).min(axis=1)
# # # apo = np.sqrt(pos_i[:, :, 0]**2 + pos_i[:, :, 1]**2).max(axis=1)

# # # np.save('./fireworks_test/data/ass_3/periastron.npy', peri)
# # # np.save('./fireworks_test/data/ass_3/apoastron.npy', apo)

# # # print('peri', peri.shape)
# # # print('apo', apo.shape)
