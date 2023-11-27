import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Load data from file
path = '/home/bertinelli/pod_compastro23/Fireworks/fireworks_test/'
data_5 = np.genfromtxt(path +'dt_5_vect.txt', delimiter=' ')
data_678 = np.genfromtxt(path +'dt_678_vect.txt', delimiter=' ')
data_910 = np.genfromtxt(path +'dt_910_vect.txt', delimiter=' ')
data_2550 = np.genfromtxt(path +'dt_2550_vect.txt', delimiter=' ')
data_diego = np.genfromtxt(path +'dt_vect_diego.txt', delimiter=' ')
pyfalc = np.genfromtxt(path +'dt_pyfalcon.txt', delimiter=' ')

N_5 = data_5[0, :]
N_678 = data_678[0, :]
N_910 = data_910[0, :]
N_2550 = data_2550[0, :]

dt_5 = data_5[1:, :]
dt_678 = data_678[1:, :]
dt_910 = data_910[1:, :]
dt_2550 = data_2550[1:, :]

dt_diego = data_diego[1:, :]
dt_pyfalc = pyfalc[1:, :]

func_list = ['acc_vect_vepe',
             'acc_vect_gia',
             'acc_onearray_vepe',
             'acc_opt_gia',
             'acc_vect_diego', 
             'pyfalcon']

array_tot = np.hstack((dt_5, dt_678, dt_910, dt_2550))
array_tot_diego = np.vstack((array_tot, dt_diego, dt_pyfalc))

N_tot = np.hstack((N_5, N_678, N_910, N_2550))

pdf = matplotlib.backends.backend_pdf.PdfPages(path + "array_tot_pyfalc.pdf")

# Generate some sample data
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 15))

# Main plot
for i in range(array_tot_diego.shape[0]):
    ax[0].plot(N_tot, array_tot_diego[i], 'o-', label=func_list[i])
    ax[1].plot(N_tot, array_tot_diego[i], 'o-', label=func_list[i])

ax[0].legend(loc='best')
ax[0].grid(linestyle='dotted')
ax[0].set_xlabel('Number of particles')
ax[0].set_ylabel('Time spent for acceleration estimation (s)')

# Inset plot
axins = inset_axes(ax[0], width="45%", height="37%", bbox_to_anchor=(.036, .02, 1, 1), loc='center left', bbox_transform=ax[0].transAxes)
for i in range(array_tot_diego.shape[0]):
    axins.plot(N_tot, array_tot_diego[i], 'o-', label=func_list[i])

axins.set_xlim(-200, 11000)
axins.set_ylim(-7, 55)
axins.grid(linestyle='dotted', alpha=0.5)

# Second main plot
ax[1].legend(loc='best')
ax[1].grid(linestyle='dotted')
ax[1].set_xlabel('Number of particles')
ax[1].set_ylabel('Time spent for acceleration estimation (s) [log scale]')
ax[1].yaxis.tick_right()
ax[1].set_yscale('log', base=10)

#fig.tight_layout(rect=[0, 0.03, 1, 0.95])
pdf.savefig(fig, dpi=300)

pdf.close()