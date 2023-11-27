import numpy as np
from matplotlib import pyplot as plt, ticker as mticker
from cycler import cycler
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
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

derr_00001_base = np.abs((data_00001_base[:,4]-Etot_0)/Etot_0)
derr_00001_mod = np.abs((data_00001_mod[:,4]-Etot_0)/Etot_0)
derr_00001_rk2 = np.abs((data_00001_rk2[:,4]-Etot_0)/Etot_0)
derr_00001_leap = np.abs((data_00001_leap[:,4]-Etot_0)/Etot_0)
derr_00001_rk4 = np.abs((data_00001_rk4[:,4]-Etot_0)/Etot_0)

derr_0001_base = np.abs((data_0001_base[:,4]-Etot_0)/Etot_0)
derr_0001_mod = np.abs((data_0001_mod[:,4]-Etot_0)/Etot_0)
derr_0001_rk2 = np.abs((data_0001_rk2[:,4]-Etot_0)/Etot_0)
derr_0001_leap = np.abs((data_0001_leap[:,4]-Etot_0)/Etot_0)
derr_0001_rk4 = np.abs((data_0001_rk4[:,4]-Etot_0)/Etot_0)

derr_001_base = np.abs((data_001_base[:,4]-Etot_0)/Etot_0)
derr_001_mod = np.abs((data_001_mod[:,4]-Etot_0)/Etot_0)
derr_001_rk2 = np.abs((data_001_rk2[:,4]-Etot_0)/Etot_0)
derr_001_leap = np.abs((data_001_leap[:,4]-Etot_0)/Etot_0)
derr_001_rk4 = np.abs((data_001_rk4[:,4]-Etot_0)/Etot_0)

timesteps = np.array([0.0001, 0.001, 0.01])


# custom_cycler1 = (cycler(color=['firebrick','lightgreen', 'purple', 'orange', 'navy']) + cycler(linestyle=['--', '-.', ':', '-', '-']))

plt.rcParams['font.size'] = '16'
plt.rcParams['lines.linewidth'] = '4'
plt.rcParams['axes.titlesize'] = '20'
plt.rcParams['axes.titlepad'] = '15'
plt.rcParams['axes.labelsize'] = '24'
plt.rcParams['axes.labelpad'] = '12'
plt.rcParams['axes.titleweight'] = '600'
plt.rcParams['axes.labelweight'] = '500'
plt.rcParams['xtick.labelsize'] = '20'
plt.rcParams['ytick.labelsize'] = '20'
plt.rcParams['xtick.major.size'] = '10'
plt.rcParams['ytick.major.size'] = '10'
plt.rcParams['ytick.minor.size'] = '4'
# plt.rc('axes', prop_cycle=custom_cycler1)

# locminy = mticker.LogLocator(base=10, subs=np.arange(2, 10) * .1, numticks=20) # subs=(0.2,0.4,0.6,0.8)
# locmajy = mticker.LogLocator(base=10,numticks=100)
# plt.rcParams['lines.markersize'] = '10'
fig7, ax7 = plt.subplots(2, 3, figsize=(40, 27))
gs7 = GridSpec(2,3)

ax7[0,0].boxplot([derr_00001_base, derr_0001_base, derr_001_base], labels=timesteps, notch=True, vert=True, patch_artist=True)
ax7[0,0].set_title('Euler_base')

ax7[0,1].boxplot([derr_00001_mod, derr_0001_mod, derr_001_mod], labels=timesteps, notch=True, vert=True, patch_artist=True)
ax7[0,1].set_title('Euler_modified')

ax7[0,2].boxplot([derr_00001_rk2, derr_0001_rk2, derr_001_rk2], labels=timesteps, notch=True, vert=True, patch_artist=True)
ax7[0,2].set_title('RK2-Heun')

ax7[1,0].boxplot([derr_00001_leap, derr_0001_leap, derr_001_leap], labels=timesteps, notch=True, vert=True, patch_artist=True)
ax7[1,0].set_title('Leapfrog')

ax7[1,1].boxplot([derr_00001_rk4, derr_0001_rk4, derr_001_rk4], labels=timesteps, notch=True, vert=True, patch_artist=True)
ax7[1,1].set_title('RK4')

ax7[1,2].axis('off')

fig7.suptitle('Relative Energy errors', fontsize=52, fontweight='600')

for i in range(2):
    for j in range(3):
        ax7[i,j].set_xticklabels(['0.0001', '0.001', '0.01'])
        ax7[i,j].set_xlabel('Time Step')
        ax7[i,j].set_ylabel('|(E-E0)/E0|')
        ax7[i,j].yaxis.grid(True, which='major', alpha=0.5)

plt.savefig('./fireworks_test/plots/ass_3/Etot_all2.pdf')