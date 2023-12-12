import numpy as np
import pandas as pd
from tqdm import tqdm
import fireworks.ic as fic
from matplotlib import pyplot as plt, ticker as mticker, animation as animation
from evolver import evolve

from fireworks.particles import Particles
import fireworks.nbodylib.dynamics as fdyn
import fireworks.nbodylib.integrators as fint
import fireworks.nbodylib.timesteps as fts

path = '/ca23/ext_volume/pod_compastro23/Fireworks/fireworks_test'
# path to data
path_data = path + "/data/ass_3/"

# ic_param = np.genfromtxt(path_data + 'ic_param_all.txt')
# mass_1 = ic_param[0]
# mass_2 = ic_param[1]
# rp = ic_param[2]
# e = ic_param[3]
# a = ic_param[4]
# Etot_0 = ic_param[5]
# Tperiod = ic_param[6]
# N_end = ic_param[7]

# data_int = np.load(path_data + 'dt_0.001.npz', allow_pickle=True)
intr = np.load(path + '/data/ass_3/pos_i.npy', allow_pickle=True)
# downsampling
m = 10

# data_base = data_int['Euler_base'][::m]
# data_mod = data_int['Euler_modified'][::m]
# # data_her = data_int['Hermite'][::m]
# data_rk2 = data_int['RK2-Heun'][::m]
# data_leap = data_int['Leapfrog'][::m]
# data_rk4 = data_int['RK4'][::m]


# EDIT THIS LINES

# intr = data_base
title = 'tsunami'


fig, ax = plt.subplots(figsize=(15,15))

plt.rcParams['font.size'] = '16'
plt.rcParams['lines.linewidth'] = '4'
plt.rcParams['axes.titlesize'] = '20'
plt.rcParams['axes.titlepad'] = '17'
plt.rcParams['axes.labelsize'] = '24'
plt.rcParams['legend.fontsize'] = '20'
plt.rcParams['axes.labelpad'] = '12'
plt.rcParams['axes.titleweight'] = '600'
plt.rcParams['axes.labelweight'] = '500'
plt.rcParams['xtick.labelsize'] = '20'
plt.rcParams['ytick.labelsize'] = '20'
plt.rcParams['xtick.major.size'] = '10'
plt.rcParams['ytick.major.size'] = '10'
plt.rcParams['ytick.minor.size'] = '4'
# plt.rcParams['figure.dpi'] = '10'

# Calculate the number of extra frames to add at the beginning and the end
pause_duration = 0.5  # pause duration in seconds start
pause_duration_end = 1  # pause duration in seconds end
frame_rate = 15  # frame rate of the animation
extra_frames = pause_duration * frame_rate
extra_frames_end = pause_duration_end * frame_rate

# Create an array of frame indices
frames = np.concatenate([
    np.full(int(extra_frames), 0),  # initial frame (pause
    np.arange(0, len(intr)-1, 100),  # original frames
    np.full(int(extra_frames_end), len(intr)-2)  # extra frames at the end
])

def update_pos(frame): 

    ax.clear()

    # for each frame, update the data stored on each artist.
    x1 = intr[:frame, 0, 0]
    x2 = intr[:frame, 1, 0]
    x3 = intr[:frame, 2, 0]
    y1 = intr[:frame, 0, 1]
    y2 = intr[:frame, 1, 1]
    y3 = intr[:frame, 2, 1]

    ax.scatter(intr[0,0, 0], intr[0,0, 1], color="tab:red", s=100, marker="x", zorder=10, label="Initial Position")
    ax.scatter(intr[0,1, 0], intr[0,1,1], color="tab:red", s=100, marker="x", zorder=10)
    ax.scatter(intr[0,2,0], intr[0,2,1], color="tab:red", s=100, marker="x", zorder=10)
    ax.plot(x1, y1, color="tab:blue", label="Body 1", alpha=0.8)
    ax.plot(x2, y2, color="tab:orange", label="Body 2", alpha=0.8)
    ax.plot(x3, y3, color="tab:green", label="Body 3", alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()

    # ax.set_xlim(np.min(intr[:, 2])-0.5, np.max(intr[:, 2])+0.5)
    # ax.set_ylim(np.min(intr[:, 3])-0.5, np.max(intr[:, 3])+0.5)
    
    fig.suptitle('Position on X-Y Plane', 
                 fontsize=24, fontweight='600')

print("Starting Position Animation")

gif_pos = animation.FuncAnimation(fig=fig, func=update_pos, frames=frames,)

gif_pos.save("/ca23/ext_volume/pod_compastro23/assignment_3/animations/two_bodies_test.gif", writer="pillow")

print("Position Animation Saved")

# def update_en(frame):
    
#         ax.clear()
    
#         # for each frame, update the data stored on each artist.
#         x = np.linspace(0, N_end*Tperiod, intr[:, 4].shape[0])
#         x_i = x[:frame]
#         # y = intr[:frame, 4]
#         y = np.abs((intr[:, 4]-Etot_0)/Etot_0)
#         y_i = y[:frame]
    
#         ax.plot(x_i, y_i, color="tab:blue", label=title)

#         ax.set_title('Energy evolution')
#         ax.set_xlabel('absolute time')
#         ax.set_ylabel('|(E-E0)/E0')
#         ax.legend()

#         locminy = mticker.LogLocator(base=10, subs=np.arange(2, 10) * .1, numticks=200) # subs=(0.2,0.4,0.6,0.8)
#         locmajy = mticker.LogLocator(base=10, numticks=100)

#         ax.yaxis.set_minor_locator(locminy)
#         ax.yaxis.set_major_locator(locmajy)
#         ax.yaxis.set_minor_formatter(mticker.NullFormatter())
#         ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

#         ax.set_xlim(np.min(x)-2, np.max(x)+2)
#         # ax.set_ylim(np.min(intr[:, 4])-0.1, np.max(intr[:, 4])+50)
#         ax.set_ylim(np.min(y), 10e-1)
#         ax.set_yscale("log")
    
#         fig.suptitle('Energy Evolution\n(M1=%.1f, M2=%.1f, e=%.1f, rp=%.2f, T=%.2f)'%(mass_1, mass_2, e, rp, Tperiod), 
#                     fontsize=24, fontweight='600')
        
# print ("\nStarting Energy Animation")

# gif_en = animation.FuncAnimation(fig=fig, func=update_en, frames=frames,)
# gif_en.save("/home/bertinelli/pod_compastro23/assignment_3/animations/energy_test.gif", writer="pillow")

# print("Energy Animation Saved")


# fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(9, 18))

# def update_all(frame):

#     ax[0].clear()
#     ax[1].clear()

#     x1 = intr[:frame, 0]
#     x2 = intr[:frame, 2]
#     y1 = intr[:frame, 1]
#     y2 = intr[:frame, 3]

#     ax[0].scatter(intr[0,0], intr[0,1], color="tab:red", s=100, marker="x", zorder=10, label="Initial Position")
#     ax[0].scatter(intr[0,2], intr[0,3], color="tab:red", s=100, marker="x", zorder=10)
#     ax[0].plot(x1, y1, color="tab:blue", label="Body 1", alpha=0.8)
#     ax[0].plot(x2, y2, color="tab:orange", label="Body 2", alpha=0.8)

#     ax[0].set_title('Leapfrog')
#     ax[0].set_xlabel('X')
#     ax[0].set_ylabel('Y')
#     ax[0].legend(loc='upper right')

#     ax[0].set_xlim(np.min(intr[:, 2])-0.5, np.max(intr[:, 2])+0.5)
#     ax[0].set_ylim(np.min(intr[:, 3])-0.5, np.max(intr[:, 3])+0.5)
    
#     fig.suptitle('Position on X-Y Plane\n(M1=%.1f, M2=%.1f, e=%.1f, rp=%.2f, T=%.2f)'%(mass_1, mass_2, e, rp, Tperiod), 
#                  fontsize=24, fontweight='600')
    
#     # for each frame, update the data stored on each artist.
#     x = np.linspace(0, N_end*Tperiod, intr[:, 4].shape[0])
#     x_i = x[:frame]
#     y = intr[:frame, 4]
#     y = np.abs((intr[:, 4]-Etot_0)/Etot_0)
#     y_i = y[:frame]

#     ax[1].plot(x_i, y_i, color="tab:blue", label=title)

#     ax[1].set_title('Energy evolution')
#     ax[1].set_xlabel('absolute time')
#     ax[1].set_ylabel('|(E-E0)/E0')
#     ax[1].legend()

#     locminy = mticker.LogLocator(base=10, subs=np.arange(2, 10) * .1, numticks=200) # subs=(0.2,0.4,0.6,0.8)
#     locmajy = mticker.LogLocator(base=10, numticks=100)

#     ax[1].yaxis.set_minor_locator(locminy)
#     ax[1].yaxis.set_major_locator(locmajy)
#     ax[1].yaxis.set_minor_formatter(mticker.NullFormatter())
#     ax[1].yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

#     ax[1].set_xlim(np.min(x)-2, np.max(x)+2)
#     # ax[1].set_ylim(np.min(data_leap[:, 4]), np.max(data_leap[:, 4]))
#     # ax[1].set_ylim(np.min(y), np.max(y))
#     ax[1].set_ylim(np.min(y), 10e-1)
#     ax[1].set_yscale("log")

#     # Adjust the space between the subplots.
#     plt.tight_layout()

# print('\nStarting Both Animation')

# gif = animation.FuncAnimation(fig=fig, func=update_all, frames=frames,)
# gif.save("/home/bertinelli/pod_compastro23/assignment_3/animations/both.gif", writer="pillow")

# print("Both Animation Saved")