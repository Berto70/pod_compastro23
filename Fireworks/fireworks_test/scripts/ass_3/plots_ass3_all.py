import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.gridspec import GridSpec

# LOAD DATA

ic_param = np.genfromtxt('./fireworks_test/data/ass_3/ic_param_all.txt')
mass1 = ic_param[0]
mass_2 = ic_param[1]
rp = ic_param[2]
e = ic_param[3]
a = ic_param[4]
Etot_0 = ic_param[5]
Tperiod = ic_param[6]
N_end = ic_param[7]

data_00001 = np.load('./fireworks_test/data/ass_3/dt_0.0001.npz', allow_pickle=True)
data_0001 = np.load('./fireworks_test/data/ass_3/dt_0.001.npz', allow_pickle=True)
data_001 = np.load('./fireworks_test/data/ass_3/dt_0.01.npz', allow_pickle=True)

data_00001_base = data_00001['Euler_base']
data_00001_mod = data_00001['Euler_modified']
data_00001_rk2 = data_00001['RK2-Heun']
data_00001_leap = data_00001['Leapfrog']
data_00001_rk4 = data_00001['RK4']

data_0001_base = data_0001['Euler_base']
data_0001_mod = data_0001['Euler_modified']
data_0001_rk2 = data_0001['RK2-Heun']
data_0001_leap = data_0001['Leapfrog']
data_0001_rk4 = data_0001['RK4']

data_001_base = data_001['Euler_base']
data_001_mod = data_001['Euler_modified']
data_001_rk2 = data_001['RK2-Heun']
data_001_leap = data_001['Leapfrog']
data_001_rk4 = data_001['RK4']

# POSITION X-Y PLOTS

custom_cycler1 = (cycler(color=['orange', 'seagreen', 'navy']))

plt.rcParams['font.size'] = '16'
plt.rcParams['lines.linewidth'] = '2'
plt.rcParams['axes.titlesize'] = '20'
plt.rcParams['axes.titlepad'] = '15'
plt.rcParams['axes.labelsize'] = '18'
plt.rcParams['axes.labelpad'] = '8'
plt.rcParams['axes.titleweight'] = '600'
plt.rcParams['axes.labelweight'] = '500'
plt.rc('axes', prop_cycle=custom_cycler1)

fig, ax = plt.subplots(2, 3, figsize=(30, 20))
gs = GridSpec(2,3)
# Plot position on x-y plane
ax[0,0].plot(data_00001_base[:, 0], data_00001_base[:, 1], alpha=0.8, label='h=0.0001')
ax[0,0].plot(data_0001_base[:, 0], data_0001_base[:, 1], alpha=0.5, label='h=0.001')
ax[0,0].plot(data_001_base[:, 0], data_001_base[:, 1], alpha=0.3, label='h=0.01')
ax[0,0].set_xlabel('X')
ax[0,0].set_ylabel('Y')
ax[0,0].set_title('Euler_base')
ax[0,0].set_prop_cycle(custom_cycler1)
ax[0,0].legend()

ax[0,1].plot(data_00001_mod[:, 0], data_00001_mod[:, 1], alpha=0.8, label='h=0.0001')
ax[0,1].plot(data_0001_mod[:, 0], data_0001_mod[:, 1], alpha=0.5, label='h=0.001')
ax[0,1].plot(data_001_mod[:, 0], data_001_mod[:, 1], alpha=0.3, label='h=0.01')
ax[0,1].set_xlabel('X')
ax[0,1].set_ylabel('Y')
ax[0,1].set_title('Euler_modified')
ax[0,1].set_prop_cycle(custom_cycler1)
ax[0,1].legend()

ax[0,2].plot(data_00001_rk2[:, 0], data_00001_rk2[:, 1], alpha=0.8, label='h=0.0001')
ax[0,2].plot(data_0001_rk2[:, 0], data_0001_rk2[:, 1], alpha=0.5, label='h=0.001')
ax[0,2].plot(data_001_rk2[:, 0], data_001_rk2[:, 1], alpha=0.3, label='h=0.01')
ax[0,2].set_xlabel('X')
ax[0,2].set_ylabel('Y')
ax[0,2].set_title('RK2-Heun')
ax[0,2].set_prop_cycle(custom_cycler1)
ax[0,2].legend()

ax[1,0].plot(data_00001_leap[:, 0], data_00001_leap[:, 1], alpha=0.8, label='h=0.0001')
ax[1,0].plot(data_0001_leap[:, 0], data_0001_leap[:, 1], alpha=0.5, label='h=0.001')
ax[1,0].plot(data_001_leap[:, 0], data_001_leap[:, 1], alpha=0.3, label='h=0.01')
ax[1,0].set_xlabel('X')
ax[1,0].set_ylabel('Y')
ax[1,0].set_title('Leapfrog')
ax[1,0].set_prop_cycle(custom_cycler1)
ax[1,0].legend()

ax[1,1].plot(data_00001_rk4[:, 0], data_00001_rk4[:, 1], alpha=0.8, label='h=0.0001')
ax[1,1].plot(data_0001_rk4[:, 0], data_0001_rk4[:, 1], alpha=0.5, label='h=0.001')
ax[1,1].plot(data_001_rk4[:, 0], data_001_rk4[:, 1], alpha=0.3, label='h=0.01')
ax[1,1].set_xlabel('X')
ax[1,1].set_ylabel('Y')
ax[1,1].set_title('RK4')
ax[1,1].set_prop_cycle(custom_cycler1)
ax[1,1].legend()

# ax11 = plt.subplot(gs[0, -2:])
# ax11.axis('off')  # Turn off the axes for the empty subplot
ax12 = plt.subplot(gs[1, -1])
ax12.axis('off')  # Turn off the axes for the empty subplot

fig.suptitle('Position on X-Y Plane', fontsize=52, fontweight='600')

plt.savefig('./fireworks_test/plots/ass_3/pos_all.pdf')


# ENERGY ERR PLOTS

custom_cycler2 = (cycler(color=['orange', 'seagreen', 'navy']) + cycler(linestyle=['-', '--', '-.']))

plt.rcParams['font.size'] = '12'
plt.rcParams['lines.linewidth'] = '2'
plt.rc('axes', prop_cycle=custom_cycler2)


fig2, ax2 = plt.subplots(2, 3, figsize=(30, 20))
gs2 = GridSpec(2,3)
# Plot position on x-y plane
ax2[0,0].plot(np.linspace(0, N_end*Tperiod, data_00001_base[:, 4].shape[0]), np.abs((data_00001_base[:, 4]-Etot_0)/Etot_0), alpha=0.8, label='h=0.0001')
ax2[0,0].plot(np.linspace(0, N_end*Tperiod, data_0001_base[:, 4].shape[0]), np.abs((data_0001_base[:, 4]-Etot_0)/Etot_0), alpha=0.8, label='h=0.001')
ax2[0,0].plot(np.linspace(0, N_end*Tperiod, data_001_base[:, 4].shape[0]), np.abs((data_001_base[:, 4]-Etot_0)/Etot_0), alpha=0.8, label='h=0.01')
ax2[0,0].set_xlabel('absolute time')
ax2[0,0].set_ylabel('|(E-E0)/E0|')
ax2[0,0].set_title('Euler_base')
ax2[0,0].set_yscale('log')
ax2[0,0].set_prop_cycle(custom_cycler2)
ax2[0,0].legend()

ax2[0,1].plot(np.linspace(0, N_end*Tperiod, data_00001_mod[:, 4].shape[0]), np.abs((data_00001_mod[:, 4]-Etot_0)/Etot_0), alpha=0.8, label='h=0.0001')
ax2[0,1].plot(np.linspace(0, N_end*Tperiod, data_0001_mod[:, 4].shape[0]), np.abs((data_0001_mod[:, 4]-Etot_0)/Etot_0), alpha=0.8, label='h=0.001')
ax2[0,1].plot(np.linspace(0, N_end*Tperiod, data_001_mod[:, 4].shape[0]), np.abs((data_001_mod[:, 4]-Etot_0)/Etot_0), alpha=0.8, label='h=0.01')
ax2[0,1].set_xlabel('absolute time')
ax2[0,1].set_ylabel('|(E-E0)/E0|')
ax2[0,1].set_title('Euler_modiefied')
ax2[0,1].set_yscale('log')
ax2[0,1].set_prop_cycle(custom_cycler2)
ax2[0,1].legend()

ax2[0,2].plot(np.linspace(0, N_end*Tperiod, data_00001_rk2[:, 4].shape[0]), np.abs((data_00001_rk2[:, 4]-Etot_0)/Etot_0), alpha=0.8, label='h=0.0001')
ax2[0,2].plot(np.linspace(0, N_end*Tperiod, data_0001_rk2[:, 4].shape[0]), np.abs((data_0001_rk2[:, 4]-Etot_0)/Etot_0), alpha=0.8, label='h=0.001')
ax2[0,2].plot(np.linspace(0, N_end*Tperiod, data_001_rk2[:, 4].shape[0]), np.abs((data_001_rk2[:, 4]-Etot_0)/Etot_0), alpha=0.8, label='h=0.01')
ax2[0,2].set_xlabel('absolute time')
ax2[0,2].set_ylabel('|(E-E0)/E0|')
ax2[0,2].set_title('RK2-Heun')
ax2[0,2].set_yscale('log')
ax2[0,2].set_prop_cycle(custom_cycler2)
ax2[0,2].legend()

ax2[1,0].plot(np.linspace(0, N_end*Tperiod, data_00001_leap[:, 4].shape[0]), np.abs((data_00001_leap[:, 4]-Etot_0)/Etot_0), alpha=0.8, label='h=0.0001')
ax2[1,0].plot(np.linspace(0, N_end*Tperiod, data_0001_leap[:, 4].shape[0]), np.abs((data_0001_leap[:, 4]-Etot_0)/Etot_0), alpha=0.8, label='h=0.001')
ax2[1,0].plot(np.linspace(0, N_end*Tperiod, data_001_leap[:, 4].shape[0]), np.abs((data_001_leap[:, 4]-Etot_0)/Etot_0), alpha=0.8, label='h=0.01')
ax2[1,0].set_xlabel('absolute time')
ax2[1,0].set_ylabel('|(E-E0)/E0|')
ax2[1,0].set_title('Leapfrog')
ax2[1,0].set_yscale('log')
ax2[1,0].set_prop_cycle(custom_cycler2)
ax2[1,0].legend()

ax2[1,1].plot(np.linspace(0, N_end*Tperiod, data_00001_rk4[:, 4].shape[0]), np.abs((data_00001_rk4[:, 4]-Etot_0)/Etot_0), alpha=0.8, label='h=0.0001')
ax2[1,1].plot(np.linspace(0, N_end*Tperiod, data_0001_rk4[:, 4].shape[0]), np.abs((data_0001_rk4[:, 4]-Etot_0)/Etot_0), alpha=0.8, label='h=0.001')
ax2[1,1].plot(np.linspace(0, N_end*Tperiod, data_001_rk4[:, 4].shape[0]), np.abs((data_001_rk4[:, 4]-Etot_0)/Etot_0), alpha=0.8, label='h=0.01')
ax2[1,1].set_xlabel('absolute time')
ax2[1,1].set_ylabel('|(E-E0)/E0|')
ax2[1,1].set_title('RK4')
ax2[1,1].set_yscale('log')
ax2[1,1].set_prop_cycle(custom_cycler2)
ax2[1,1].legend()

ax212 = plt.subplot(gs2[1, -1])
ax212.axis('off')  # Turn off the axes for the empty subplot

fig2.suptitle('ΔE evolution', fontsize=52, fontweight='600')

plt.savefig('./fireworks_test/plots/ass_3/Etot_all.pdf')