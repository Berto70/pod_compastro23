import numpy as np
import matplotlib.pyplot as plt

path = '/home/bertinelli/pod_compastro23/Fireworks/fireworks_test'
# Load data
ic_param = np.genfromtxt(path + '/data/ass_3/ic_param_tsu.txt')
mass1 = ic_param[0]
mass_2 = ic_param[1]
rp = ic_param[2]
e = ic_param[3]
a = ic_param[4]
Etot_0 = ic_param[5]
Tperiod = ic_param[6]
N_end = ic_param[7]

pos_i = np.load(path + '/data/ass_3/pos_i.npy', allow_pickle=True)
# vel_i = np.load(path + '/data/ass_3/vel_i.npy', allow_pickle=True)
# acc_i = np.load(path + '/data/ass_3/acc_i.npy', allow_pickle=True)
# mass_i = np.load(path + '/data/ass_3/mass_i.npy', allow_pickle=True)
Etot_i = np.load(path + '/data/ass_3/Etot_i.npy', allow_pickle=True)

# data_tsu = np.load(path + '/data/ass_3/data_tusnami_e0.00.npz', allow_pickle=True)

# data_data = data_tsu['0.001'][::1]

# print(Etot_i)
# print(Etot_i.shape)

# peri = np.load('./fireworks_test/data/ass_3/periastron.npy', allow_pickle=True)
# apo = np.load('./fireworks_test/data/ass_3/apoastron.npy', allow_pickle=True)

fig, ax = plt.subplots(2, 1, figsize=(10, 20))
# Plot position on x-y plane
# ax[0].plot(data_data[:, 2], data_data[:, 3], alpha=0.8, label='2')
# ax[0].plot(data_data[:, 0], data_data[:, 1], alpha=0.8, label='1')
ax[0].plot(pos_i[:, :, 0], pos_i[:, :, 1], label=['Body 1', 'Body 2'])
# ax[0].plot(pos_i[:, :, 2], pos_i[:, :, 3], label=['Body 1', 'Body 2', 'Body 3'], linewidth=0.0001)
# ax[0].plot(pos_i[:, :, 4], pos_i[:, :, 5], label=['Body 1', 'Body 2', 'Body 3'], linewidth=0.0001)
ax[0].set_xlabel('X')
ax[0].set_ylabel('Y')
ax[0].legend()
ax[0].set_title('Position on X-Y Plane')

# # Plot velocity on x-y plane
# ax[1].plot(np.linspace(0, N_end*Tperiod, vel_i.shape[0]), vel_i[:, :, 1], label=['Body 1', 'Body 2'])
# ax[1].set_xlabel('absolute time')
# ax[1].set_ylabel('v_y')
# ax[1].legend()
# ax[1].set_title('Velocity on Y Plane')
plt.savefig('/home/bertinelli/pod_compastro23/Fireworks/fireworks_test/plots/ass_3/pos_vel_plot.pdf')

# fig1, ax1 = plt.subplots(1, 1, figsize=(8,5))
# # Plot position on x-y plane
# ax1.plot(np.linspace(0, N_end*Tperiod, vel_i.shape[0]), peri, label='pericenter')
# ax1.plot(np.linspace(0, N_end*Tperiod, vel_i.shape[0]), apo, label='apocenter')
# ax1.set_xlabel('absolute time')
# ax1.set_ylabel('absolute value')
# ax1.legend()
# ax1.set_title('Peri-apo evolution')
# plt.savefig('ext_volume/pod_compastro23/Fireworks/fireworks_test/plots/ass_3/peri_apo_plot.pdf')

fig2, ax2 = plt.subplots(1, 1, figsize=(8,5))
# Plot position on x-y plane
ax2.plot(np.linspace(0, N_end*Tperiod, pos_i.shape[0]), np.abs((Etot_i-Etot_0)/Etot_0))
# ax2.plot(np.linspace(0, N_end*Tperiod, data_data[:, 4].shape[0]), np.abs((data_data[:, 4]-Etot_0)/Etot_0))
ax2.set_xlabel('absolute time')
ax2.set_ylabel('|(E-E0)/E0|')
ax2.set_yscale('log')  # Set y-axis to be in log scale
ax2.set_title('ΔE evolution')
plt.savefig('/home/bertinelli/pod_compastro23/Fireworks/fireworks_test/plots/ass_3/Etot.pdf')

# # Plot the precession of perihelion and aphelion
# fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))
# ax2.plot(np.linspace(0, 10*Tperiod, vel_i.shape[0]), theta_peri[:, 0], label='Perihelion')
# ax2.plot(np.linspace(0, 10*Tperiod, vel_i.shape[0]), theta_apo[:, 0], label='Aphelion')
# ax2.set_xlabel('absolute time')
# ax2.set_ylabel('Rate of change of angle')
# ax2.legend()
# ax2.set_title('Precession of Perihelion and Aphelion')

# plt.savefig('./fireworks_test/plots/ass_3/precession_plot.pdf')

plt.show()