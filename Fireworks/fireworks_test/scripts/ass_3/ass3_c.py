import numpy as np
import fireworks.ic as fic
import matplotlib.pyplot as plt
from fireworks.particles import Particles
from tqdm import tqdm
import fireworks.nbodylib.dynamics as fdyn
import fireworks.nbodylib.integrators as fint
import fireworks.nbodylib.timesteps as fts

# Initialize two stars in a circular orbit
mass1 = 8.0
mass2 = 2.0
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
# tstep = 0.0001
N_end = 65

pos_i = []
vel_i = []
acc_i = []
mass_i = []
Etot_i = []
tstep_i = []

position = np.array([[0,0,0],
                     [0.5,0.866,9],
                     [1,0,0]])
vel = np.array([[0,0,0],
                [0,0,0],
                [0,0,0]])

mass = np.array([3,4,5])


# Create instances of the particles
particles = Particles(position, vel, mass)


# pbar = tqdm(total=N_end*Tperiod/tstep)

tstart=0

tintermediate=[5,10,15,20,30,40,50,60,65]

tcurrent=0

for t in tintermediate:
    tstep = t-tcurrent

    part, efftime, _, _, _ = fint.integrator_tsunami(particles, tstep)
     # Here we can save stuff, plot stuff, etc.
    pos_i.append(part.pos)
    vel_i.append(part.vel)
    mass_i.append(part.mass)
    Etot_j, _, _ = part.Etot()
    Etot_i.append(Etot_j)

    tcurrent += efftime

# while t < N_end*Tperiod:
#     # t += tstep
#     part, tstep, acc, _, _ = fint.integrator_rk4(particles, 
#                                                     tstep=tstep, 
#                                                     acceleration_estimator=fdyn.acceleration_pyfalcon
#                                                 )
#     pos_i.append(part.pos)
#     vel_i.append(part.vel)
#     mass_i.append(part.mass)
#     acc_i.append(acc)
#     tstep_i.append(tstep)

#     Etot_j, _, _ = part.Etot()
#     Etot_i.append(Etot_j)

#     # Update the progress bar
#     pbar.update(1)

#     t += tstep
#     # tstep = fts.euler_timestep(part, eta=0.0001, acc = acc)

pos_i = np.array(pos_i)
vel_i = np.array(vel_i)
# acc_i = np.array(acc_i)
mass_i = np.array(mass_i)
Etot_i = np.array(Etot_i)
tstep_i = np.array(tintermediate)

np.save('/ca23/ext_volume/pod_compastro23/Fireworks/fireworks_test/data/ass_3/pos_i.npy', pos_i)
np.save('/ca23/ext_volume/pod_compastro23/Fireworks/fireworks_test/data/ass_3/vel_i.npy', vel_i)
# np.save('/ca23/ext_volume/pod_compastro23/Fireworks/fireworks_test/data/ass_3/acc_i.npy', acc_i)
np.save('/ca23/ext_volume/pod_compastro23/Fireworks/fireworks_test/data/ass_3/mass_i.npy', mass_i)
np.save('/ca23/ext_volume/pod_compastro23/Fireworks/fireworks_test/data/ass_3/Etot_i.npy', Etot_i)
np.save('/ca23/ext_volume/pod_compastro23/Fireworks/fireworks_test/data/ass_3/tstep_i.npy', tstep_i)

ic_param = np.array([mass1, mass2, rp, e, a, Etot_0, Tperiod, tstep, N_end])
np.savetxt('/ca23/ext_volume/pod_compastro23/Fireworks/fireworks_test/data/ass_3/ic_param.txt', ic_param)

print('pos_i', pos_i.shape)
print('vel_i', vel_i.shape)
print('tsteo_i', tstep_i.shape)

# # peri = np.sqrt(pos_i[:, :, 0]**2 + pos_i[:, :, 1]**2).min(axis=1)
# # apo = np.sqrt(pos_i[:, :, 0]**2 + pos_i[:, :, 1]**2).max(axis=1)

# # np.save('./fireworks_test/data/ass_3/periastron.npy', peri)
# # np.save('./fireworks_test/data/ass_3/apoastron.npy', apo)

# # print('peri', peri.shape)
# # print('apo', apo.shape)
