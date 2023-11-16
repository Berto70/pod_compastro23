import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Load data from file
path = '/home/bertinelli/pod_compastro23/Fireworks/fireworks_test/'
data = np.genfromtxt(path +'dt_678_vect.txt', delimiter=' ')
N_list = data[0, :]
dt = data[1:, :]
func_list = ['acc_vect_vepe',
             'acc_vect_gia',
             'acc_onearray_vepe',
             'acc_opt_gia']

pdf = matplotlib.backends.backend_pdf.PdfPages(path + "plot_678_vect.pdf")

# Generate some sample data
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 15))

# Main plot
for i in range(dt.shape[0]):
    ax[0].plot(N_list, dt[i], 'o-', label=func_list[i])
    ax[1].plot(N_list, dt[i], 'o-', label=func_list[i])

ax[0].legend(loc='best')
ax[0].grid(linestyle='dotted')
ax[0].set_xlabel('Number of particles')
ax[0].set_ylabel('Time spent for acceleration estimation (s)')

# # Inset plot
# axins = inset_axes(ax[0], width="45%", height="30%", bbox_to_anchor=(.032, .02, 1, 1), loc='center left', bbox_transform=ax[0].transAxes)
# for i in range(dt.shape[0]):
#     axins.plot(N_list, dt[i], 'o-', label=func_list[i])

# axins.set_ylim(-0.5, 18)
# axins.grid(linestyle='dotted', alpha=0.5)

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
