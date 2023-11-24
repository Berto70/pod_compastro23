import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Load data
ic_param = np.genfromtxt('./fireworks_test/data/ass_3/ic_param.txt')
mass1 = ic_param[0]
mass_2 = ic_param[1]
rp = ic_param[2]
e = ic_param[3]
a = ic_param[4]
Etot_0 = ic_param[5]
Tperiod = ic_param[6]
N_end = ic_param[7]

data_001 = np.load('./fireworks_test/data/ass_3/dt_0.01.npz')
# data_01 = np.load('./fireworks_test/data/ass_3/dt_0.1.npz')
# data_1 = np.load('./fireworks_test/data/ass_3/dt_100.0.npz')

data_001_base = data_001['Euler_base']
# data_001_mod = data_001['Euler_modified']
# data_001_rk2 = data_001['RK2-Heun']
data_001_leap = data_001['Leapfrog']
data_001_rk4 = data_001['RK4']

# data_01_base = data_001['Euler_base']
# data_01_mod = data_001['Euler_modified']
# data_01_rk2 = data_001['RK2-Heun']
# data_01_leap = data_001['Leapfrog']
# data_01_rk4 = data_001['RK4']

# data_1_base = data_001['Euler_base']
# data_1_mod = data_001['Euler_modified']
# data_1_rk2 = data_001['RK2-Heun']
# data_1_leap = data_001['Leapfrog']
# data_1_rk4 = data_001['RK4']

plt.rcParams['font.size'] = '16'
plt.rcParams['lines.linewidth'] = '2'

fig, ax = plt.subplots(2, 3, figsize=(30, 20))
gs = GridSpec(2,3)
# Plot position on x-y plane
ax[0,0].plot(data_001_base[:, 0], data_001_base[:, 1], alpha=0.8, label='euler', c='orange')
# ax[0,0].plot(data_01_base[:, 0], data_01_base[:, 1], alpha=0.8, label='h=0.1', c='green')
# ax[0,0].plot(data_1_base[:, 0], data_1_base[:, 1], alpha=0.8, label='h=100', c='blue')
ax[0,0].plot(data_001_leap[:, 0], data_001_leap[:, 1], linestyle='--', c='cyan', label='leap')
ax[0,0].plot(data_001_rk4[:, 0], data_001_rk4[:, 1], linestyle='--', c='green', label='rk4')
# ax[0,0].plot(data_01_base[:, 2], data_01_base[:, 3], linestyle='--', c='green')
# ax[0,0].plot(data_1_base[:, 2], data_1_base[:, 3], linestyle='--', c='blue')
ax[0,0].set_xlabel('X')
ax[0,0].set_ylabel('Y')
ax[0,0].set_title('Euler_base')
ax[0,0].legend()

# ax[0,1].plot(data_001_mod[:, 0], data_001_mod[:, 1], alpha=0.8, label='h=0.01')
# ax[0,1].plot(data_01_mod[:, 0], data_01_mod[:, 1], alpha=0.8, label='h=0.1')
# ax[0,1].plot(data_1_mod[:, 0], data_1_mod[:, 1], alpha=0.8, label='h=1')
# ax[0,1].set_xlabel('X')
# ax[0,1].set_ylabel('Y')
# ax[0,1].set_title('Euler_modified')
# ax[0,1].legend()

# ax[0,2].plot(data_001_rk2[:, 0], data_001_rk2[:, 1], alpha=0.8, label='h=0.01')
# ax[0,2].plot(data_01_rk2[:, 0], data_01_rk2[:, 1], alpha=0.8, label='h=0.1')
# ax[0,2].plot(data_1_rk2[:, 0], data_1_rk2[:, 1], alpha=0.8, label='h=0.1')
# ax[0,2].set_xlabel('X')
# ax[0,2].set_ylabel('Y')
# ax[0,2].set_title('RK2-Heun')
# ax[0,2].legend()

# ax[1,0].plot(data_001_leap[:, 0], data_001_leap[:, 1], alpha=0.8, label='h=0.01')
# ax[1,0].plot(data_01_leap[:, 0], data_01_leap[:, 1], alpha=0.8, label='h=0.1')
# ax[1,0].plot(data_1_leap[:, 0], data_1_leap[:, 1], alpha=0.8, label='h=1')
# ax[1,0].set_xlabel('X')
# ax[1,0].set_ylabel('Y')
# ax[1,0].set_title('Leapfrog')
# ax[1,0].legend()

# ax[1,1].plot(data_001_rk4[:, 0], data_001_rk4[:, 1], alpha=0.8, label='h=0.01')
# ax[1,1].plot(data_01_rk4[:, 0], data_01_rk4[:, 1], alpha=0.8, label='h=0.1')
# ax[1,1].plot(data_1_rk4[:, 0], data_1_rk4[:, 1], alpha=0.8, label='h=1')
# ax[1,1].set_xlabel('X')
# ax[1,1].set_ylabel('Y')
# ax[1,1].set_title('RK4')
# ax[1,1].legend()
ax11 = plt.subplot(gs[0, -2:])
ax11.axis('off')  # Turn off the axes for the empty subplot
ax12 = plt.subplot(gs[1, :])
ax12.axis('off')  # Turn off the axes for the empty subplot

fig.suptitle('Position on X-Y Plane', fontsize=52)

# # Plot velocity on x-y plane
# ax[1].plot(np.linspace(0, N_end*Tperiod, vel_i.shape[0]), vel_i[:, :, 1], label=['Body 1', 'Body 2'])
# ax[1].set_xlabel('absolute time')
# ax[1].set_ylabel('v_y')
# ax[1].legend()
# ax[1].set_title('Velocity on Y Plane')
plt.savefig('./fireworks_test/plots/ass_3/pos_all.pdf')

# fig1, ax1 = plt.subplots(1, 1, figsize=(8,5))
# # Plot position on x-y plane
# ax1.plot(np.linspace(0, N_end*Tperiod, vel_i.shape[0]), peri, label='pericenter')
# ax1.plot(np.linspace(0, N_end*Tperiod, vel_i.shape[0]), apo, label='apocenter')
# ax1.set_xlabel('absolute time')
# ax1.set_ylabel('absolute value')
# ax1.legend()
# ax1.set_title('Peri-apo evolution')
# plt.savefig('./fireworks_test/plots/ass_3/peri_apo_plot.pdf')

plt.rcParams['font.size'] = '11'
plt.rcParams['lines.linewidth'] = '1'

fig2, ax2 = plt.subplots(1, 1, figsize=(8,5))
# Plot position on x-y plane
ax2.plot(np.linspace(0, N_end*Tperiod, data_001_base[:, 4].shape[0]), np.abs((data_001_base[:, 4]-Etot_0)/Etot_0), label='base')
ax2.plot(np.linspace(0, N_end*Tperiod, data_001_rk4[:, 4].shape[0]), np.abs((data_001_rk4[:, 4]-Etot_0)/Etot_0), label='rk4')
ax2.plot(np.linspace(0, N_end*Tperiod, data_001_leap[:, 4].shape[0]), np.abs((data_001_leap[:, 4]-Etot_0)/Etot_0), label='leap')

ax2.set_xlabel('absolute time')
ax2.set_ylabel('|(E-E0)/E0|')
ax2.set_yscale('log')  # Set y-axis to be in log scale
ax2.set_title('ΔE evolution (Euler_base)')
ax2.legend()
plt.savefig('./fireworks_test/plots/ass_3/Etot_all.pdf')

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