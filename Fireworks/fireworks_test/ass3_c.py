import numpy as np
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

t = 0.

pos_i = []
vel_i = []
acc_i = []
mass_i = []
Etot_i = []


while t < 10*Tperiod:
    part, _, acc, _, _ = fint.integrator_template(part, tstep=0.01, acceleration_estimator=fdyn.acceleration_pyfalcon)
    pos_i.append(part.pos)
    vel_i.append(part.vel)
    mass_i.append(part.mass)
    acc_i.append(acc)

    Etot_j, _, _ = part.Etot()
    Etot_i.append(Etot_j)

    t += 0.01

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

print('pos_i', pos_i.shape)
print('vel_i', vel_i.shape)

peri = np.sqrt(pos_i[:, :, 0]**2 + pos_i[:, :, 1]**2).min(axis=1)
apo = np.sqrt(pos_i[:, :, 0]**2 + pos_i[:, :, 1]**2).max(axis=1)

np.save('./fireworks_test/data/ass_3/periastron.npy', peri)
np.save('./fireworks_test/data/ass_3/apoastron.npy', apo)

print('peri', peri.shape)
print('apo', apo.shape)



# # Calculate the angles of perihelion and aphelion
# theta_peri = np.arctan2(pos_i[:, :, 1], pos_i[:, :, 0])
# theta_apo = np.arctan2(pos_i[:, :, 1], pos_i[:, :, 0])

# # Calculate the rate of change of the angles
# dtheta_peri = np.gradient(theta_peri, axis=0)
# dtheta_apo = np.gradient(theta_apo, axis=0)

# print('dtheta_peri', dtheta_peri.shape)
# print('dtheta_apo', dtheta_apo.shape)

# fig, ax = plt.subplots(2, 1, figsize=(10, 20))
# # Plot position on x-y plane
# ax[0].plot(pos_i[:, :, 0], pos_i[:, :, 1], label=['Body 1', 'Body 2'], linewidth=0.01)
# ax[0].set_xlabel('X')
# ax[0].set_ylabel('Y')
# ax[0].legend()
# ax[0].set_title('Position on X-Y Plane')

# # Plot velocity on x-y plane
# ax[1].plot(np.linspace(0, 10*Tperiod, vel_i.shape[0]), vel_i[:, :, 1], label=['Body 1', 'Body 2'])
# ax[1].set_xlabel('absolute time')
# ax[1].set_ylabel('v_y')
# ax[1].legend()
# ax[1].set_title('Velocity on Y Plane')
# plt.savefig('./fireworks_test/plots/pos_vel_plot.pdf')

# fig1, ax1 = plt.subplots(1, 1, figsize=(8,5))
# # Plot position on x-y plane
# ax1.plot(np.linspace(0, 10*Tperiod, vel_i.shape[0]), peri, label='pericenter')
# ax1.plot(np.linspace(0, 10*Tperiod, vel_i.shape[0]), apo, label='apocenter')
# ax1.set_xlabel('absolute time')
# ax1.set_ylabel('absolute value')
# ax1.legend()
# ax1.set_title('Peri-apo evolution')
# plt.savefig('./fireworks_test/plots/peri_apo_plot.pdf')

# # # Plot the precession of perihelion and aphelion
# # fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))
# # ax2.plot(np.linspace(0, 10*Tperiod, vel_i.shape[0]), theta_peri[:, 0], label='Perihelion')
# # ax2.plot(np.linspace(0, 10*Tperiod, vel_i.shape[0]), theta_apo[:, 0], label='Aphelion')
# # ax2.set_xlabel('absolute time')
# # ax2.set_ylabel('Rate of change of angle')
# # ax2.legend()
# # ax2.set_title('Precession of Perihelion and Aphelion')

# # plt.savefig('./fireworks_test/plots/precession_plot.pdf')

# plt.show()