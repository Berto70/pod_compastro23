import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Load data from file
path = '/home/bertinelli/pod_compastro23/Fireworks/fireworks_test/data/'
data = np.genfromtxt(path +'dt_tot.txt', delimiter=' ')
N_list_dir = data[0, 0:7]
N_list_vect = data[0, :]
dt_dir = data[1:4, 0:7]
dt_vect = data[4:, :]

func_list = ['acc_dir_vepe', 
             'acc_dir_gia', 
             'acc_dir_diego', 
             'acc_vect_vepe', 
             'acc_onearray_vepe', 
             'jerk_vepe', 
             'acc_vect_gia', 
             'acc_opt_gia', 
             'acc_vect_diego', 
             'acceleration_pyfalcon']



pdf = matplotlib.backends.backend_pdf.PdfPages("plots/plot_vect_model.pdf")


# Generate some sample data
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 15))

# Main plot
# for i in range(dt_dir.shape[0]):
#     ax[0].plot(N_list_dir, dt_dir[i], 'o-', label=func_list[i])
#     ax[1].plot(N_list_dir, dt_dir[i], 'o-', label=func_list[i])

for j in range(dt_vect.shape[0]):
    ax[0].plot(N_list_vect, dt_vect[j], 'o-', label=func_list[j+dt_dir.shape[0]])
    ax[1].plot(N_list_vect, dt_vect[j], 'o-', label=func_list[j+dt_dir.shape[0]])

# ax[0].plot(N_list_vect, dt_vect[6, :], 'o-', label=func_list[9])
# ax[1].plot(N_list_dir, dt_vect[6, 0:7], 'o-', label=func_list[9])

ax[0].legend(loc='best')
ax[0].grid(linestyle='dotted')
ax[0].set_xlabel('Number of particles')
ax[0].set_ylabel('Time spent for acceleration estimation (s)')

# Inset plot
axins = inset_axes(ax[0], width="45%", height="37%", bbox_to_anchor=(.036, .0, 1, 1), loc='center left', bbox_transform=ax[0].transAxes)
for i in range(dt_vect.shape[0]):
    axins.plot(N_list_vect, dt_vect[i], 'o-', label=func_list[i+dt_dir.shape[0]])

axins.set_xlim(-200, 12000)
axins.set_ylim(-7, 55)
axins.grid(linestyle='dotted', alpha=0.5)


# Second main plot
ax[1].legend(loc='best')
ax[1].grid(linestyle='dotted')
ax[1].set_xlabel('Number of particles')
ax[1].set_ylabel('Time spent for acceleration estimation (s) [log scale]')
ax[1].yaxis.tick_right()
ax[1].set_yscale('log', base=10)

ax[0].set_title('Vectorized acceleration estimation methods')
ax[1].set_title('Vectorized acceleration estimation methods (log-scale))')

#fig.tight_layout(rect=[0, 0.03, 1, 0.95])
pdf.savefig(fig, dpi=300)

pdf.close()
