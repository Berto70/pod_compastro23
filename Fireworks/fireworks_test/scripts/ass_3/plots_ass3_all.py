import numpy as np
from matplotlib import pyplot as plt, ticker as mticker
from cycler import cycler
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages

path = "/home/bertinelli/pod_compastro23/Fireworks/fireworks_test"

# LOAD DATA #####

ic_param = np.genfromtxt(path + '/data/ass_3/ic_param_all.txt')
mass_1 = ic_param[0]
mass_2 = ic_param[1]
rp = ic_param[2]
e = ic_param[3]
a = ic_param[4]
Etot_0 = ic_param[5]
Tperiod = ic_param[6]
N_end = ic_param[7]

ic_param_tsu = np.genfromtxt(path + '/data/ass_3/ic_param_tsu.txt')
tevol_tsu = ic_param_tsu[7]

#downsample
n = 100
m = 10
w = 1

data_00001 = np.load(path + '/data/ass_3/dt_0.0001.npz', allow_pickle=True)
data_0001 = np.load(path + '/data/ass_3/dt_0.001.npz', allow_pickle=True)
data_001 = np.load(path + '/data/ass_3/dt_0.01.npz', allow_pickle=True)
data_tsu = np.load(path + '/data/ass_3/data_tsunami.npz', allow_pickle=True)

data_00001_base = data_00001['Euler_base'][::n]
data_00001_mod = data_00001['Euler_modified'][::n]
data_00001_her = data_00001['Hermite'][::n]
data_00001_rk2 = data_00001['RK2-Heun'][::n]
data_00001_leap = data_00001['Leapfrog'][::n]
data_00001_rk4 = data_00001['RK4'][::n]
data_00001_tsu = data_tsu['0.00001'][::n]

data_0001_base = data_0001['Euler_base'][::m]
data_0001_mod = data_0001['Euler_modified'][::m]
data_0001_her = data_0001['Hermite'][::m]
data_0001_rk2 = data_0001['RK2-Heun'][::m]
data_0001_leap = data_0001['Leapfrog'][::m]
data_0001_rk4 = data_0001['RK4'][::m]
data_0001_tsu = data_tsu['0.0001'][::m]

data_001_base = data_001['Euler_base'][::w]
data_001_mod = data_001['Euler_modified'][::w]
data_001_her = data_001['Hermite'][::w]
data_001_rk2 = data_001['RK2-Heun'][::w]
data_001_leap = data_001['Leapfrog'][::w]
data_001_rk4 = data_001['RK4'][::w]
data_001_tsu = data_tsu['0.001'][::w]


with PdfPages('/home/bertinelli/pod_compastro23/Fireworks/fireworks_test/plots/ass_3/ass_3_plots_e%.1f_rp%.2f_both.pdf' % (e, rp)) as pdf:

    # POSITION X-Y PLOTS

    plt.rcParams['font.size'] = '16'
    plt.rcParams['lines.linewidth'] = '4'
    plt.rcParams['axes.titlesize'] = '28'
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
    plt.rcParams['figure.dpi'] = '10'

    custom_cycler1 = (cycler(color=['orange', 'seagreen', 'navy']))
    plt.rc('axes', prop_cycle=custom_cycler1)

    fig, ax = plt.subplots(3, 3, figsize=(40, 27))
    gs = GridSpec(3,3)
    # Plot position on x-y plane
    ax[0,0].plot(data_001_base[:, 0], data_001_base[:, 1], alpha=0.5, label='h=0.001')
    ax[0,0].plot(data_0001_base[:, 0], data_0001_base[:, 1], alpha=0.5, label='h=0.0001')
    ax[0,0].plot(data_00001_base[:, 0], data_00001_base[:, 1], alpha=0.8, label='h=0.00001')
    ax[0,0].plot(data_001_base[:, 2], data_001_base[:, 3], alpha=0.5, linestyle='--')
    ax[0,0].plot(data_0001_base[:, 2], data_0001_base[:, 3], alpha=0.5, linestyle='--')
    ax[0,0].plot(data_00001_base[:, 2], data_00001_base[:, 3], alpha=0.8, linestyle='--')
    ax[0,0].set_title('Euler_base')

    ax[0,1].plot(data_001_mod[:, 0], data_001_mod[:, 1], alpha=0.5, label='h=0.001')
    ax[0,1].plot(data_0001_mod[:, 0], data_0001_mod[:, 1], alpha=0.5, label='h=0.0001')
    ax[0,1].plot(data_00001_mod[:, 0], data_00001_mod[:, 1], alpha=0.8, label='h=0.00001')
    ax[0,1].plot(data_001_mod[:, 2], data_001_mod[:, 3], alpha=0.5, linestyle='--')
    ax[0,1].plot(data_0001_mod[:, 2], data_0001_mod[:, 3], alpha=0.5, linestyle='--')
    ax[0,1].plot(data_00001_mod[:, 2], data_00001_mod[:, 3], alpha=0.8, linestyle='--')
    ax[0,1].set_title('Euler_modified')

    ax[0,2].plot(data_001_her[:, 0], data_001_her[:, 1], alpha=0.5, label='h=0.001')
    ax[0,2].plot(data_0001_her[:, 0], data_0001_her[:, 1], alpha=0.5, label='h=0.0001')
    ax[0,2].plot(data_00001_her[:, 0], data_00001_her[:, 1], alpha=0.8, label='h=0.00001')
    ax[0,2].plot(data_001_her[:, 2], data_001_her[:, 3], alpha=0.5, linestyle='--')
    ax[0,2].plot(data_0001_her[:, 2], data_0001_her[:, 3], alpha=0.5, linestyle='--')
    ax[0,2].plot(data_00001_her[:, 2], data_00001_her[:, 3], alpha=0.8, linestyle='--')
    ax[0,2].set_title('Hermite')

    ax[1,0].plot(data_001_rk2[:, 0], data_001_rk2[:, 1], alpha=0.5, label='h=0.001')
    ax[1,0].plot(data_0001_rk2[:, 0], data_0001_rk2[:, 1], alpha=0.5, label='h=0.0001')
    ax[1,0].plot(data_00001_rk2[:, 0], data_00001_rk2[:, 1], alpha=0.8, label='h=0.00001')
    ax[1,0].plot(data_001_rk2[:, 2], data_001_rk2[:, 3], alpha=0.5, linestyle='--')
    ax[1,0].plot(data_0001_rk2[:, 2], data_0001_rk2[:, 3], alpha=0.5, linestyle='--')
    ax[1,0].plot(data_00001_rk2[:, 2], data_00001_rk2[:, 3], alpha=0.8, linestyle='--')
    ax[1,0].set_title('RK2-Heun')

    ax[1,1].plot(data_001_leap[:, 0], data_001_leap[:, 1], alpha=0.5, label='h=0.001')
    ax[1,1].plot(data_0001_leap[:, 0], data_0001_leap[:, 1], alpha=0.5, label='h=0.0001')
    ax[1,1].plot(data_00001_leap[:, 0], data_00001_leap[:, 1], alpha=0.8, label='h=0.00001')
    ax[1,1].plot(data_001_leap[:, 2], data_001_leap[:, 3], alpha=0.5, linestyle='--')
    ax[1,1].plot(data_0001_leap[:, 2], data_0001_leap[:, 3], alpha=0.5, linestyle='--')
    ax[1,1].plot(data_00001_leap[:, 2], data_00001_leap[:, 3], alpha=0.8, linestyle='--')
    ax[1,1].set_title('Leapfrog')

    ax[1,2].plot(data_001_rk4[:, 0], data_001_rk4[:, 1], alpha=0.5, label='h=0.001')
    ax[1,2].plot(data_0001_rk4[:, 0], data_0001_rk4[:, 1], alpha=0.5, label='h=0.0001')
    ax[1,2].plot(data_00001_rk4[:, 0], data_00001_rk4[:, 1], alpha=0.8, label='h=0.00001')
    ax[1,2].plot(data_001_rk4[:, 2], data_001_rk4[:, 3], alpha=0.5, linestyle='--')
    ax[1,2].plot(data_0001_rk4[:, 2], data_0001_rk4[:, 3], alpha=0.5, linestyle='--')
    ax[1,2].plot(data_00001_rk4[:, 2], data_00001_rk4[:, 3], alpha=0.8, linestyle='--')
    ax[1,2].set_title('RK4')

    ax[2,0].plot(data_001_tsu[:, 0], data_001_tsu[:, 1], alpha=0.5, label='h=0.001')
    ax[2,0].plot(data_0001_tsu[:, 0], data_0001_tsu[:, 1], alpha=0.5, label='h=0.0001')
    ax[2,0].plot(data_00001_tsu[:, 0], data_00001_tsu[:, 1], alpha=0.8, label='h=0.00001')
    ax[2,0].plot(data_001_tsu[:, 2], data_001_tsu[:, 3], alpha=0.5, linestyle='--')
    ax[2,0].plot(data_0001_tsu[:, 2], data_0001_tsu[:, 3], alpha=0.5, linestyle='--')
    ax[2,0].plot(data_00001_tsu[:, 2], data_00001_tsu[:, 3], alpha=0.8, linestyle='--')
    ax[2,0].set_title('Tsunami')
    
    for i in range(3):
        for j in range(3):
            ax[i,j].set_xlabel('X [N-Body units]')
            ax[i,j].set_ylabel('Y [N-Body units]')
            ax[i,j].set_prop_cycle(custom_cycler1)
            ax[i,j].legend()            
            ax[i,j].set_xlim(np.min(data_00001_mod[:, 2])-0.05, np.max(data_00001_mod[:, 2])+0.05)
            ax[i,j].set_ylim(np.min(data_00001_mod[:, 3])-0.05, np.max(data_00001_mod[:, 3])+0.05)
            ax[0,0].set_xlim(np.min(data_00001_base[:, 2])-0.05, np.max(data_00001_base[:, 2])+0.05)
            ax[0,0].set_ylim(np.min(data_00001_base[:, 3])-0.05, np.max(data_00001_base[:, 3])+0.05)

        # default ax[i,j].set_xlim(np.min(data_00001_mod[:, 0])-0.05, np.max(data_00001_mod[:, 0])+0.05)
        #         ax[i,j].set_ylim(np.min(data_00001_mod[:, 1])-0.05, np.max(data_00001_mod[:, 1])+0.05)
        #         ax[0,0].set_xlim(np.min(data_001_base[:, 0])-0.05, np.max(data_001_base[:, 0])+0.05)
        #         ax[0,0].set_ylim(np.min(data_001_base[:, 1])-0.05, np.max(data_001_base[:, 1])+0.05)


    fig.delaxes(ax[2,1], ax[2,2])


    fig.suptitle('Position on X-Y Plane\n(M1=%.1f, M2=%.1f, e=%.1f, rp=%.2f, T=%.2f)'%(mass_1, mass_2, e, rp, Tperiod), 
                 fontsize=52, fontweight='600')

    pdf.savefig(dpi = 100)
    plt.close()

##############################################################################################################
    # ENERGY ERR PLOTS (DIFF INT)

    # plt.rcParams['font.size'] = '12'
    # plt.rcParams['lines.linewidth'] = '2'

    custom_cycler2 = (cycler(color=['orange','lightgreen', 'navy']) + cycler(linestyle=['-', '-.', '--']))
    plt.rc('axes', prop_cycle=custom_cycler2)

    locminy = mticker.LogLocator(base=10, subs=np.arange(2, 10) * .1, numticks=200) # subs=(0.2,0.4,0.6,0.8)
    locmajy = mticker.LogLocator(base=10, numticks=100)

    fig2, ax2 = plt.subplots(3, 3, figsize=(40, 27))
    gs2 = GridSpec(3,3)
    # Plot position on x-y plane
    ax2[0,0].plot(np.linspace(0, N_end*Tperiod, data_001_base[:, 4].shape[0]), np.abs((data_001_base[:, 4]-Etot_0)/Etot_0)
                  , alpha=0.8, label='h=0.001')
    ax2[0,0].plot(np.linspace(0, N_end*Tperiod, data_0001_base[:, 4].shape[0]), np.abs((data_0001_base[:, 4]-Etot_0)/Etot_0)
                  , alpha=0.8, label='h=0.0001')
    ax2[0,0].plot(np.linspace(0, N_end*Tperiod, data_00001_base[:, 4].shape[0]), np.abs((data_00001_base[:, 4]-Etot_0)/Etot_0)
                  , alpha=0.8, label='h=0.00001')
    ax2[0,0].set_title('Euler_base')

    ax2[0,1].plot(np.linspace(0, N_end*Tperiod, data_001_mod[:, 4].shape[0]), np.abs((data_001_mod[:, 4]-Etot_0)/Etot_0)
                  , alpha=0.8, label='h=0.001')
    ax2[0,1].plot(np.linspace(0, N_end*Tperiod, data_0001_mod[:, 4].shape[0]), np.abs((data_0001_mod[:, 4]-Etot_0)/Etot_0)
                  , alpha=0.8, label='h=0.0001')
    ax2[0,1].plot(np.linspace(0, N_end*Tperiod, data_00001_mod[:, 4].shape[0]), np.abs((data_00001_mod[:, 4]-Etot_0)/Etot_0)
                  , alpha=0.8, label='h=0.00001')
    ax2[0,1].set_title('Euler_modiefied')

    # ax2[0,1].yaxis.set_minor_locator(locminy)
    # ax2[0,1].yaxis.set_major_locator(locmajy)
    # ax2[0,1].yaxis.set_minor_formatter(mticker.NullFormatter())
    # ax2[0,1].yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

    ax2[0,2].plot(np.linspace(0, N_end*Tperiod, data_001_her[:, 4].shape[0]), np.abs((data_001_her[:, 4]-Etot_0)/Etot_0)
                    , alpha=0.8, label='h=0.001')
    ax2[0,2].plot(np.linspace(0, N_end*Tperiod, data_0001_her[:, 4].shape[0]), np.abs((data_0001_her[:, 4]-Etot_0)/Etot_0)
                    , alpha=0.8, label='h=0.0001')
    ax2[0,2].plot(np.linspace(0, N_end*Tperiod, data_00001_her[:, 4].shape[0]), np.abs((data_00001_her[:, 4]-Etot_0)/Etot_0)
                    , alpha=0.8, label='h=0.00001')
    ax2[0,2].set_title('Hermite')
    
    ax2[1,0].plot(np.linspace(0, N_end*Tperiod, data_001_rk2[:, 4].shape[0]), np.abs((data_001_rk2[:, 4]-Etot_0)/Etot_0)
                  , alpha=0.8, label='h=0.001')
    ax2[1,0].plot(np.linspace(0, N_end*Tperiod, data_0001_rk2[:, 4].shape[0]), np.abs((data_0001_rk2[:, 4]-Etot_0)/Etot_0)
                  , alpha=0.8, label='h=0.0001')
    ax2[1,0].plot(np.linspace(0, N_end*Tperiod, data_00001_rk2[:, 4].shape[0]), np.abs((data_00001_rk2[:, 4]-Etot_0)/Etot_0)
                  , alpha=0.8, label='h=0.00001')
    ax2[1,0].set_title('RK2-Heun')

    ax2[1,1].plot(np.linspace(0, N_end*Tperiod, data_001_leap[:, 4].shape[0]), np.abs((data_001_leap[:, 4]-Etot_0)/Etot_0)
                  , alpha=0.8, label='h=0.001')
    ax2[1,1].plot(np.linspace(0, N_end*Tperiod, data_0001_leap[:, 4].shape[0]), np.abs((data_0001_leap[:, 4]-Etot_0)/Etot_0)
                  , alpha=0.8, label='h=0.0001')
    ax2[1,1].plot(np.linspace(0, N_end*Tperiod, data_00001_leap[:, 4].shape[0]), np.abs((data_00001_leap[:, 4]-Etot_0)/Etot_0)
                  , alpha=0.8, label='h=0.00001')
    ax2[1,1].set_title('Leapfrog')

    ax2[1,2].plot(np.linspace(0, N_end*Tperiod, data_001_rk4[:, 4].shape[0]), np.abs((data_001_rk4[:, 4]-Etot_0)/Etot_0)
                  , alpha=0.8, label='h=0.001')
    ax2[1,2].plot(np.linspace(0, N_end*Tperiod, data_0001_rk4[:, 4].shape[0]), np.abs((data_0001_rk4[:, 4]-Etot_0)/Etot_0)
                  , alpha=0.8, label='h=0.0001')
    ax2[1,2].plot(np.linspace(0, N_end*Tperiod, data_00001_rk4[:, 4].shape[0]), np.abs((data_00001_rk4[:, 4]-Etot_0)/Etot_0)
                  , alpha=0.8, label='h=0.00001')
    ax2[1,2].set_title('RK4')

    ax2[2,0].plot(np.linspace(0, tevol_tsu, data_001_tsu[:, 4].shape[0]), np.abs((data_001_tsu[:, 4]-Etot_0)/Etot_0)
                    , alpha=0.8, label='h=0.001')
    ax2[2,0].plot(np.linspace(0, tevol_tsu, data_0001_tsu[:, 4].shape[0]), np.abs((data_0001_tsu[:, 4]-Etot_0)/Etot_0)
                    , alpha=0.8, label='h=0.0001')
    ax2[2,0].plot(np.linspace(0, tevol_tsu, data_00001_tsu[:, 4].shape[0]), np.abs((data_00001_tsu[:, 4]-Etot_0)/Etot_0)
                    , alpha=0.8, label='h=0.00001')
    ax2[2,0].set_title('Tsunami')

    for j in range(3):
        for i in range(3):
            ax2[i,j].set_xlabel('Time [N-Body units]')
            ax2[i,j].set_ylabel('|(E-E0)/E0|')
            ax2[i,j].set_yscale('log')
            ax2[i,j].set_prop_cycle(custom_cycler2)
            ax2[i,j].legend()

            # ax2[i,j].yaxis.set_minor_locator(locminy)
            # ax2[i,j].yaxis.set_major_locator(locmajy)
            # ax2[i,j].yaxis.set_minor_formatter(mticker.NullFormatter())
            # ax2[i,j].yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

    ax2[0,1].yaxis.set_minor_locator(locminy)
    ax2[0,1].yaxis.set_major_locator(locmajy)
    ax2[0,1].yaxis.set_minor_formatter(mticker.NullFormatter())
    ax2[0,1].yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

    ax2[0,2].yaxis.set_minor_locator(locminy)
    ax2[0,2].yaxis.set_major_locator(locmajy)
    ax2[0,2].yaxis.set_minor_formatter(mticker.NullFormatter())
    ax2[0,2].yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

    
    fig2.delaxes(ax2[2,1], ax2[2,2])

    fig2.suptitle('ΔE evolution\n(M1=%.1f, M2=%.1f, e=%.1f, rp=%.2f, T=%.2f)'%(mass_1, mass_2, e, rp, Tperiod), 
                  fontsize=52, fontweight='600')

    pdf.savefig(dpi=100)
    plt.close()

##############################################################################################################
    # ENERGY ERR PLOTS (DIFF TSTEP)
    # plt.rcParams['axes.titlepad'] = '20'

    locminy = mticker.LogLocator(base=10, subs=np.arange(2, 10) * .1, numticks=200) # subs=(0.2,0.4,0.6,0.8)
    locmajy = mticker.LogLocator(base=10, numticks=100)

    custom_cycler3 = (cycler(color=['firebrick','lightgreen', 'purple', 'orange', 'navy', 'tab:blue', 'tab:red']) + cycler(linestyle=['--', '-.', '--', ':', '-', '-', ':']))
    plt.rc('axes', prop_cycle=custom_cycler3)

    fig3, ax3 = plt.subplots(1, 3, figsize=(40, 17))
    gs3 = GridSpec(1,3)

    ax3[0].set_position([0.1, 0.1, 0.25, 0.7])  # Adjust these values as needed
    ax3[1].set_position([0.4, 0.1, 0.25, 0.7])
    ax3[2].set_position([0.7, 0.1, 0.25, 0.7])

    ax3[0].plot(np.linspace(0, N_end*Tperiod, data_001_base[:, 4].shape[0]), np.abs((data_001_base[:, 4]-Etot_0)/Etot_0), 
                alpha=0.8, label='Euler_base')
    ax3[0].plot(np.linspace(0, N_end*Tperiod, data_001_mod[:, 4].shape[0]), np.abs((data_001_mod[:, 4]-Etot_0)/Etot_0), 
                alpha=0.8, label='Euler_modified')
    ax3[0].plot(np.linspace(0, N_end*Tperiod, data_001_her[:, 4].shape[0]), np.abs((data_001_her[:, 4]-Etot_0)/Etot_0),
                 alpha=0.8, label='Hermite')
    ax3[0].plot(np.linspace(0, N_end*Tperiod, data_001_rk2[:, 4].shape[0]), np.abs((data_001_rk2[:, 4]-Etot_0)/Etot_0), 
                alpha=0.8, label='RK2-Heun')
    ax3[0].plot(np.linspace(0, N_end*Tperiod, data_001_leap[:, 4].shape[0]), np.abs((data_001_leap[:, 4]-Etot_0)/Etot_0), 
                alpha=0.8, label='Leapfrog')
    ax3[0].plot(np.linspace(0, N_end*Tperiod, data_001_rk4[:, 4].shape[0]), np.abs((data_001_rk4[:, 4]-Etot_0)/Etot_0), 
                alpha=0.8, label='RK4')   
    ax[0].plot(np.linspace(0, tevol_tsu, data_001_tsu[:, 4].shape[0]), np.abs((data_001_tsu[:, 4]-Etot_0)/Etot_0),
                alpha=0.8, label='Tsunami') 
    ax3[0].set_title('h=0.001')
    ax3[0].legend(loc='lower right')

    ax3[1].plot(np.linspace(0, N_end*Tperiod, data_0001_base[:, 4].shape[0]), np.abs((data_0001_base[:, 4]-Etot_0)/Etot_0), 
                alpha=0.8, label='Euler_base')
    ax3[1].plot(np.linspace(0, N_end*Tperiod, data_0001_mod[:, 4].shape[0]), np.abs((data_0001_mod[:, 4]-Etot_0)/Etot_0), 
                alpha=0.8, label='Euler_modified')
    ax3[1].plot(np.linspace(0, N_end*Tperiod, data_0001_her[:, 4].shape[0]), np.abs((data_0001_her[:, 4]-Etot_0)/Etot_0),
                 alpha=0.8, label='Hermite')
    ax3[1].plot(np.linspace(0, N_end*Tperiod, data_0001_rk2[:, 4].shape[0]), np.abs((data_0001_rk2[:, 4]-Etot_0)/Etot_0), 
                alpha=0.8, label='RK2-Heun')
    ax3[1].plot(np.linspace(0, N_end*Tperiod, data_0001_leap[:, 4].shape[0]), np.abs((data_0001_leap[:, 4]-Etot_0)/Etot_0), 
                alpha=0.8, label='Leapfrog')
    ax3[1].plot(np.linspace(0, N_end*Tperiod, data_0001_rk4[:, 4].shape[0]), np.abs((data_0001_rk4[:, 4]-Etot_0)/Etot_0), 
                alpha=0.8, label='RK4') 
    ax3[1].plot(np.linspace(0, tevol_tsu, data_0001_tsu[:, 4].shape[0]), np.abs((data_0001_tsu[:, 4]-Etot_0)/Etot_0),
                alpha=0.8, label='Tsunami')   
    ax3[1].set_title('h=0.0001')
    ax3[1].legend(loc='upper right')

    ax3[2].plot(np.linspace(0, N_end*Tperiod, data_00001_base[:, 4].shape[0]), np.abs((data_00001_base[:, 4]-Etot_0)/Etot_0),
                 alpha=0.8, label='Euler_base')
    ax3[2].plot(np.linspace(0, N_end*Tperiod, data_00001_mod[:, 4].shape[0]), np.abs((data_00001_mod[:, 4]-Etot_0)/Etot_0),
                 alpha=0.8, label='Euler_modified')
    ax3[2].plot(np.linspace(0, N_end*Tperiod, data_00001_her[:, 4].shape[0]), np.abs((data_00001_her[:, 4]-Etot_0)/Etot_0),
                 alpha=0.8, label='Hermite')
    ax3[2].plot(np.linspace(0, N_end*Tperiod, data_00001_rk2[:, 4].shape[0]), np.abs((data_00001_rk2[:, 4]-Etot_0)/Etot_0),
                 alpha=0.8, label='RK2-Heun')
    ax3[2].plot(np.linspace(0, N_end*Tperiod, data_00001_leap[:, 4].shape[0]), np.abs((data_00001_leap[:, 4]-Etot_0)/Etot_0),
                 alpha=0.8, label='Leapfrog')
    ax3[2].plot(np.linspace(0, N_end*Tperiod, data_00001_rk4[:, 4].shape[0]), np.abs((data_00001_rk4[:, 4]-Etot_0)/Etot_0),
                 alpha=0.8, label='RK4')
    ax3[2].plot(np.linspace(0, tevol_tsu, data_00001_tsu[:, 4].shape[0]), np.abs((data_00001_tsu[:, 4]-Etot_0)/Etot_0),
                 alpha=0.8, label='Tsunami')    
    ax3[2].set_title('h=0.00001')
    ax3[2].legend(loc='upper right')   

    for i in range(3):
        ax3[i].set_xlabel('absolute time')
        ax3[i].set_ylabel('|(E-E0)/E0|')
        ax3[i].set_yscale('log')
        ax3[i].set_prop_cycle(custom_cycler3)

    ax3[0].yaxis.set_minor_locator(locminy)
    ax3[0].yaxis.set_major_locator(locmajy)
    ax3[0].yaxis.set_minor_formatter(mticker.NullFormatter())
    ax3[0].yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
    
    ax3[1].yaxis.set_minor_locator(locminy)
    ax3[1].yaxis.set_major_locator(locmajy)
    ax3[1].yaxis.set_minor_formatter(mticker.NullFormatter())
    ax3[1].yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

    ax3[2].yaxis.set_minor_locator(locminy)
    ax3[2].yaxis.set_major_locator(locmajy)
    ax3[2].yaxis.set_minor_formatter(mticker.NullFormatter())
    ax3[2].yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

    fig3.suptitle('ΔE evolution\n(M1=%.1f, M2=%.1f, e=%.1f, rp=%.2f, T=%.2f)'%(mass_1, mass_2, e, rp, Tperiod), 
                  fontsize=52, fontweight='600')
    
    pdf.savefig(dpi=100)
    plt.close()

# ##############################################################################################################
#     # TOT ENERGY PLOTS (DIFF INT)

#     # plt.rcParams['font.size'] = '12'
#     # plt.rcParams['lines.linewidth'] = '2'

#     custom_cycler4 = (cycler(color=['orange', 'seagreen', 'navy']) + cycler(linestyle=['-', '--', '-.']))
#     plt.rc('axes', prop_cycle=custom_cycler4)

#     locminy = mticker.LogLocator(base=10, subs=np.arange(2, 10) * .1, numticks=200) # subs=(0.2,0.4,0.6,0.8)
#     locmajy = mticker.LogLocator(base=10, numticks=100)

#     fig4, ax4 = plt.subplots(2, 3, figsize=(40, 27))
#     gs4 = GridSpec(2,3)
#     # Plot position on x-y plane
#     ax4[0,0].plot(np.linspace(0, N_end*Tperiod, data_00001_base[:, 4].shape[0]), np.abs(data_00001_base[:, 4]), alpha=0.8, label='h=0.00001')
#     ax4[0,0].plot(np.linspace(0, N_end*Tperiod, data_0001_base[:, 4].shape[0]), np.abs(data_0001_base[:, 4]), alpha=0.8, label='h=0.0001')
#     ax4[0,0].plot(np.linspace(0, N_end*Tperiod, data_001_base[:, 4].shape[0]), np.abs(data_001_base[:, 4]), alpha=0.8, label='h=0.001')
#     ax4[0,0].set_title('Euler_base')

#     ax4[0,1].plot(np.linspace(0, N_end*Tperiod, data_00001_mod[:, 4].shape[0]), np.abs(data_00001_mod[:, 4]), alpha=0.8, label='h=0.00001')
#     ax4[0,1].plot(np.linspace(0, N_end*Tperiod, data_0001_mod[:, 4].shape[0]), np.abs(data_0001_mod[:, 4]), alpha=0.8, label='h=0.0001')
#     ax4[0,1].plot(np.linspace(0, N_end*Tperiod, data_001_mod[:, 4].shape[0]), np.abs(data_001_mod[:, 4]), alpha=0.8, label='h=0.001')
#     ax4[0,1].set_title('Euler_modiefied')

#     ax4[0,2].plot(np.linspace(0, N_end*Tperiod, data_00001_rk2[:, 4].shape[0]), np.abs(data_00001_rk2[:, 4]), alpha=0.8, label='h=0.00001')
#     ax4[0,2].plot(np.linspace(0, N_end*Tperiod, data_0001_rk2[:, 4].shape[0]), np.abs(data_0001_rk2[:, 4]), alpha=0.8, label='h=0.0001')
#     ax4[0,2].plot(np.linspace(0, N_end*Tperiod, data_001_rk2[:, 4].shape[0]), np.abs(data_001_rk2[:, 4]), alpha=0.8, label='h=0.001')
#     ax4[0,2].set_title('RK2-Heun')

#     ax4[1,0].plot(np.linspace(0, N_end*Tperiod, data_00001_leap[:, 4].shape[0]), np.abs(data_00001_leap[:, 4]), alpha=0.8, label='h=0.00001')
#     ax4[1,0].plot(np.linspace(0, N_end*Tperiod, data_0001_leap[:, 4].shape[0]), np.abs(data_0001_leap[:, 4]), alpha=0.8, label='h=0.0001')
#     ax4[1,0].plot(np.linspace(0, N_end*Tperiod, data_001_leap[:, 4].shape[0]), np.abs(data_001_leap[:, 4]), alpha=0.8, label='h=0.001')
#     ax4[1,0].set_title('Leapfrog')

#     ax4[1,1].plot(np.linspace(0, N_end*Tperiod, data_00001_rk4[:, 4].shape[0]), np.abs(data_00001_rk4[:, 4]), alpha=0.8, label='h=0.00001')
#     ax4[1,1].plot(np.linspace(0, N_end*Tperiod, data_0001_rk4[:, 4].shape[0]), np.abs(data_0001_rk4[:, 4]), alpha=0.8, label='h=0.0001')
#     ax4[1,1].plot(np.linspace(0, N_end*Tperiod, data_001_rk4[:, 4].shape[0]), np.abs(data_001_rk4[:, 4]), alpha=0.8, label='h=0.001')
#     ax4[1,1].set_title('RK4')

#     for i in range(2):
#         for j in range(3):
#             ax4[i,j].set_xlabel('absolute time')
#             ax4[i,j].set_ylabel('Etot')
#             ax4[i,j].set_yscale('log')
#             ax4[i,j].set_prop_cycle(custom_cycler4)
#             ax4[i,j].legend()

#     for i in range(2):
#         for j in range(3):
#             ax2[i,j].yaxis.set_minor_locator(locminy)
#             ax2[i,j].yaxis.set_major_locator(locmajy)
#             ax2[i,j].yaxis.set_minor_formatter(mticker.NullFormatter())
#             ax2[i,j].yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

#     # ax2[0,1].yaxis.set_minor_locator(locminy)
#     # ax2[0,1].yaxis.set_major_locator(locmajy)
#     # ax2[0,1].yaxis.set_minor_formatter(mticker.NullFormatter())
#     # ax2[0,1].yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

#     # ax2[0,2].yaxis.set_minor_locator(locminy)
#     # ax2[0,2].yaxis.set_major_locator(locmajy)
#     # ax2[0,2].yaxis.set_minor_formatter(mticker.NullFormatter())
#     # ax2[0,2].yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

#     # ax212 = plt.subplot(gs2[1, -1])
#     # ax212.axis('off')  # Turn off the axes for the empty subplot
#     fig4.delaxes(ax4[1,-1])

#     fig4.suptitle('Etot evolution', fontsize=52, fontweight='600')

#     # plt.savefig('./fireworks_test/plots/ass_3/Etot_all.pdf')
#     pdf.savefig(dpi=100)
#     plt.close()

# ##############################################################################################################
#     # TOT ENERGY PLOTS (DIFF TSTEP)

#     custom_cycler5 = (cycler(color=['firebrick','lightgreen', 'purple', 'orange', 'navy']) + cycler(linestyle=['--', '-.', ':', '-', '-']))
#     plt.rc('axes', prop_cycle=custom_cycler5)

#     fig5, ax5 = plt.subplots(1, 3, figsize=(40, 15))
#     gs5 = GridSpec(1,3)

#     ax5[2].plot(np.linspace(0, N_end*Tperiod, data_00001_base[:, 4].shape[0]), np.abs(data_00001_base[:, 4]), alpha=0.8, label='Euler_base')
#     ax5[2].plot(np.linspace(0, N_end*Tperiod, data_00001_mod[:, 4].shape[0]), np.abs(data_00001_mod[:, 4]), alpha=0.8, label='Euler_modified')
#     ax5[2].plot(np.linspace(0, N_end*Tperiod, data_00001_rk2[:, 4].shape[0]), np.abs(data_00001_rk2[:, 4]), alpha=0.8, label='RK2-Heun')
#     ax5[2].plot(np.linspace(0, N_end*Tperiod, data_00001_leap[:, 4].shape[0]), np.abs(data_00001_leap[:, 4]), alpha=0.8, label='Leapfrog')
#     ax5[2].plot(np.linspace(0, N_end*Tperiod, data_00001_rk4[:, 4].shape[0]), np.abs(data_00001_rk4[:, 4]), alpha=0.8, label='RK4')
#     ax5[2].set_title('h=0.00001')
#     ax5[2].legend(loc='upper right')

#     ax5[1].plot(np.linspace(0, N_end*Tperiod, data_0001_base[:, 4].shape[0]), np.abs(data_0001_base[:, 4]), alpha=0.8, label='Euler_base')
#     ax5[1].plot(np.linspace(0, N_end*Tperiod, data_0001_mod[:, 4].shape[0]), np.abs(data_0001_mod[:, 4]), alpha=0.8, label='Euler_modified')
#     ax5[1].plot(np.linspace(0, N_end*Tperiod, data_0001_rk2[:, 4].shape[0]), np.abs(data_0001_rk2[:, 4]), alpha=0.8, label='RK2-Heun')
#     ax5[1].plot(np.linspace(0, N_end*Tperiod, data_0001_leap[:, 4].shape[0]), np.abs(data_0001_leap[:, 4]), alpha=0.8, label='Leapfrog')
#     ax5[1].plot(np.linspace(0, N_end*Tperiod, data_0001_rk4[:, 4].shape[0]), np.abs(data_0001_rk4[:, 4]), alpha=0.8, label='RK4')
#     ax5[1].set_title('h=0.0001')
#     ax5[1].legend(loc='upper right')

#     ax5[0].plot(np.linspace(0, N_end*Tperiod, data_001_base[:, 4].shape[0]), np.abs(data_001_base[:, 4]), alpha=0.8, label='Euler_base')
#     ax5[0].plot(np.linspace(0, N_end*Tperiod, data_001_mod[:, 4].shape[0]), np.abs(data_001_mod[:, 4]), alpha=0.8, label='Euler_modified')
#     ax5[0].plot(np.linspace(0, N_end*Tperiod, data_001_rk2[:, 4].shape[0]), np.abs(data_001_rk2[:, 4]), alpha=0.8, label='RK2-Heun')
#     ax5[0].plot(np.linspace(0, N_end*Tperiod, data_001_leap[:, 4].shape[0]), np.abs(data_001_leap[:, 4]), alpha=0.8, label='Leapfrog')
#     ax5[0].plot(np.linspace(0, N_end*Tperiod, data_001_rk4[:, 4].shape[0]), np.abs(data_001_rk4[:, 4]), alpha=0.8, label='RK4')
#     ax5[0].set_title('h=0.001')
#     ax5[0].legend(loc='lower right')

#     for i in range(3):
#         ax5[i].set_xlabel('absolute time')
#         ax5[i].set_ylabel('|(E-E0)/E0|')
#         ax5[i].set_yscale('log')
#         ax5[i].set_prop_cycle(custom_cycler5)

#     fig5.suptitle('Etot evolution', fontsize=52, fontweight='600')
#     pdf.savefig(dpi=100)
#     plt.close()

##############################################################################################################
    # ENERGY ERR VS TIMESTEP

    max_00001_base = np.max(np.abs((data_00001_base[:,4]-Etot_0)/Etot_0))
    avg_00001_base = np.mean(np.abs((data_00001_base[:,4]-Etot_0)/Etot_0))
    max_00001_mod = np.max(np.abs((data_00001_mod[:,4]-Etot_0)/Etot_0))
    avg_00001_mod = np.mean(np.abs((data_00001_base[:,4]-Etot_0)/Etot_0))
    max_00001_her = np.max(np.abs((data_00001_her[:,4]-Etot_0)/Etot_0))
    avg_00001_her = np.mean(np.abs((data_00001_base[:,4]-Etot_0)/Etot_0))
    max_00001_rk2 = np.max(np.abs((data_00001_base[:,4]-Etot_0)/Etot_0))
    avg_00001_rk2 = np.mean(np.abs((data_00001_base[:,4]-Etot_0)/Etot_0))
    max_00001_leap = np.max(np.abs((data_00001_base[:,4]-Etot_0)/Etot_0))
    avg_00001_leap = np.mean(np.abs((data_00001_base[:,4]-Etot_0)/Etot_0))
    max_00001_rk4 = np.max(np.abs((data_00001_base[:,4]-Etot_0)/Etot_0))
    avg_00001_rk4 = np.mean(np.abs((data_00001_base[:,4]-Etot_0)/Etot_0))
    max_00001_tsu = np.max(np.abs((data_00001_tsu[:,4]-Etot_0)/Etot_0))
    avg_00001_tsu = np.mean(np.abs((data_00001_base[:,4]-Etot_0)/Etot_0))

    max_0001_base = np.max(np.abs((data_0001_base[:,4]-Etot_0)/Etot_0))
    avg_0001_base = np.mean(np.abs((data_0001_base[:,4]-Etot_0)/Etot_0))
    max_0001_mod = np.max(np.abs((data_0001_mod[:,4]-Etot_0)/Etot_0))
    avg_0001_mod = np.mean(np.abs((data_0001_base[:,4]-Etot_0)/Etot_0))
    max_0001_her = np.max(np.abs((data_0001_her[:,4]-Etot_0)/Etot_0))
    avg_0001_her = np.mean(np.abs((data_0001_base[:,4]-Etot_0)/Etot_0))
    max_0001_rk2 = np.max(np.abs((data_0001_base[:,4]-Etot_0)/Etot_0))
    avg_0001_rk2 = np.mean(np.abs((data_0001_base[:,4]-Etot_0)/Etot_0))
    max_0001_leap = np.max(np.abs((data_0001_base[:,4]-Etot_0)/Etot_0))
    avg_0001_leap = np.mean(np.abs((data_0001_base[:,4]-Etot_0)/Etot_0))
    max_0001_rk4 = np.max(np.abs((data_0001_base[:,4]-Etot_0)/Etot_0))
    avg_0001_rk4 = np.mean(np.abs((data_0001_base[:,4]-Etot_0)/Etot_0))
    max_0001_tsu = np.max(np.abs((data_0001_tsu[:,4]-Etot_0)/Etot_0))
    avg_0001_tsu = np.mean(np.abs((data_0001_base[:,4]-Etot_0)/Etot_0))

    max_001_base = np.max(np.abs((data_001_base[:,4]-Etot_0)/Etot_0))
    avg_001_base = np.mean(np.abs((data_001_base[:,4]-Etot_0)/Etot_0))
    max_001_mod = np.max(np.abs((data_001_mod[:,4]-Etot_0)/Etot_0))
    avg_001_mod = np.mean(np.abs((data_001_base[:,4]-Etot_0)/Etot_0))
    max_001_her = np.max(np.abs((data_001_her[:,4]-Etot_0)/Etot_0))
    avg_001_her = np.mean(np.abs((data_001_base[:,4]-Etot_0)/Etot_0))
    max_001_rk2 = np.max(np.abs((data_001_base[:,4]-Etot_0)/Etot_0))
    avg_001_rk2 = np.mean(np.abs((data_001_base[:,4]-Etot_0)/Etot_0))
    max_001_leap = np.max(np.abs((data_001_base[:,4]-Etot_0)/Etot_0))
    avg_001_leap = np.mean(np.abs((data_001_base[:,4]-Etot_0)/Etot_0))
    max_001_rk4 = np.max(np.abs((data_001_base[:,4]-Etot_0)/Etot_0))
    avg_001_rk4 = np.mean(np.abs((data_001_base[:,4]-Etot_0)/Etot_0))
    max_001_tsu = np.max(np.abs((data_001_tsu[:,4]-Etot_0)/Etot_0))
    avg_001_tsu = np.mean(np.abs((data_001_base[:,4]-Etot_0)/Etot_0))

    timesteps = np.array([0.00001, 0.0001, 0.001])

    tsu_tstep = np.array([data_00001_tsu[5][-1], data_0001_tsu[5][-1], data_001_tsu[5][-1]])

    plt.rcParams['lines.markersize'] = '10'
    fig6, ax6 = plt.subplots(3, 3, figsize=(40, 27))
    gs6 = GridSpec(3,3)

    ax6[0,0].loglog(timesteps, [max_00001_base, max_0001_base, max_001_base], 
                    marker='o', label='Max Energy Error', color='firebrick', linestyle='--')
    ax6[0,0].loglog(timesteps, [avg_00001_base, avg_0001_base, avg_001_base], 
                    marker='s', label='Avg Energy Error', color='seagreen', linestyle='-')
    ax6[0,0].set_title('Euler_base')
    
    ax6[0,1].loglog(timesteps, [max_00001_mod, max_0001_mod, max_001_mod], 
                    marker='o', label='Max Energy Error', color='firebrick', linestyle='--')
    ax6[0,1].loglog(timesteps, [avg_00001_mod, avg_0001_mod, avg_001_mod], 
                    marker='s', label='Avg Energy Error', color='seagreen', linestyle='-')
    ax6[0,1].set_title('Euler_modified')

    ax6[0,2].loglog(timesteps, [max_00001_her, max_0001_her, max_001_her],
                    marker='o', label='Max Energy Error', color='firebrick', linestyle='--')
    ax6[0,2].loglog(timesteps, [avg_00001_her, avg_0001_her, avg_001_her],
                    marker='s', label='Avg Energy Error', color='seagreen', linestyle='-')
    ax6[0,2].set_title('Hermite')

    ax6[1,0].loglog(timesteps, [max_00001_rk2, max_0001_rk2, max_001_rk2], 
                    marker='o', label='Max Energy Error', color='firebrick', linestyle='--')
    ax6[1,0].loglog(timesteps, [avg_00001_rk2, avg_0001_rk2, avg_001_rk2], 
                    marker='s', label='Avg Energy Error', color='seagreen', linestyle='-')
    ax6[1,0].set_title('RK2-Heun')

    ax6[1,1].loglog(timesteps, [max_00001_leap, max_0001_leap, max_001_leap], 
                    marker='o', label='Max Energy Error', color='firebrick', linestyle='--')
    ax6[1,1].loglog(timesteps, [avg_00001_leap, avg_0001_leap, avg_001_leap], 
                    marker='s', label='Avg Energy Error', color='seagreen', linestyle='-')
    ax6[1,1].set_title('Leapfrog')

    ax6[1,2].loglog(timesteps, [max_00001_rk4, max_0001_rk4, max_001_rk4], 
                    marker='o', label='Max Energy Error', color='firebrick', linestyle='--')
    ax6[1,2].loglog(timesteps, [avg_00001_rk4, avg_0001_rk4, avg_001_rk4], 
                    marker='s', label='Avg Energy Error', color='seagreen', linestyle='-')
    ax6[1,2].set_title('RK4')

    ax6[2,0].loglog(tsu_tstep, [max_00001_tsu, max_0001_tsu, max_001_tsu],
                    marker='o', label='Max Energy Error', color='firebrick', linestyle='--')
    ax6[2,0].loglog(tsu_tstep, [avg_00001_tsu, avg_0001_tsu, avg_001_tsu],
                    marker='s', label='Avg Energy Error', color='seagreen', linestyle='-')
    ax6[2,0].set_title('Tsunami')

    fig6.delaxes(ax2[2,1], ax2[2,2])

    fig6.suptitle('Energy Error vs. Time Step\n(M1=%.1f, M2=%.1f, e=%.1f, rp=%.2f, T=%.2f)'%(mass_1, mass_2, e, rp, Tperiod),
                   fontsize=52, fontweight='600')


    for i in range(3):
        for j in range(3):
            ax6[i,j].set_xlabel('Time Step')
            ax6[i,j].set_ylabel('Energy error')
            ax6[i,j].legend(loc='best')
            ax6[i,j].grid(True, which='both', alpha=0.3)

    pdf.savefig(dpi=100)
    plt.close()

##############################################################################################################
    # ENERGY ERR VS TIMESTEP (BOXPLOT)

    derr_00001_base = np.abs((data_00001_base[:,4]-Etot_0)/Etot_0)
    derr_00001_mod = np.abs((data_00001_mod[:,4]-Etot_0)/Etot_0)
    derr_00001_her = np.abs((data_00001_her[:,4]-Etot_0)/Etot_0)
    derr_00001_rk2 = np.abs((data_00001_rk2[:,4]-Etot_0)/Etot_0)
    derr_00001_leap = np.abs((data_00001_leap[:,4]-Etot_0)/Etot_0)
    derr_00001_rk4 = np.abs((data_00001_rk4[:,4]-Etot_0)/Etot_0)
    derr_00001_tsu = np.abs((data_00001_tsu[:,4]-Etot_0)/Etot_0)

    derr_0001_base = np.abs((data_0001_base[:,4]-Etot_0)/Etot_0)
    derr_0001_mod = np.abs((data_0001_mod[:,4]-Etot_0)/Etot_0)
    derr_0001_her = np.abs((data_0001_her[:,4]-Etot_0)/Etot_0)
    derr_0001_rk2 = np.abs((data_0001_rk2[:,4]-Etot_0)/Etot_0)
    derr_0001_leap = np.abs((data_0001_leap[:,4]-Etot_0)/Etot_0)
    derr_0001_rk4 = np.abs((data_0001_rk4[:,4]-Etot_0)/Etot_0)
    derr_0001_tsu = np.abs((data_0001_tsu[:,4]-Etot_0)/Etot_0)

    derr_001_base = np.abs((data_001_base[:,4]-Etot_0)/Etot_0)
    derr_001_mod = np.abs((data_001_mod[:,4]-Etot_0)/Etot_0)
    derr_001_her = np.abs((data_001_her[:,4]-Etot_0)/Etot_0)
    derr_001_rk2 = np.abs((data_001_rk2[:,4]-Etot_0)/Etot_0)
    derr_001_leap = np.abs((data_001_leap[:,4]-Etot_0)/Etot_0)
    derr_001_rk4 = np.abs((data_001_rk4[:,4]-Etot_0)/Etot_0)
    derr_001_tsu = np.abs((data_001_tsu[:,4]-Etot_0)/Etot_0)

    timesteps = np.array([0.00001, 0.0001, 0.001])
    tsu_tstep = np.array([data_00001_tsu[5][-1], data_0001_tsu[5][-1], data_001_tsu[5][-1]])

    custom_cycler7 = (cycler(color=['tab:blue', 'tab:orange']))
    plt.rc('axes', prop_cycle=custom_cycler7)

    fig7, ax7 = plt.subplots(3, 3, figsize=(40, 27))
    gs7 = GridSpec(3,3)

    ax7[0,0].boxplot([derr_00001_base, derr_0001_base, derr_001_base], 
                     labels=timesteps, notch=True, vert=True, patch_artist=True, boxprops=dict(alpha=0.8), showfliers=False)
    ax7[0,0].set_title('Euler_base')

    ax7[0,1].boxplot([derr_00001_mod, derr_0001_mod, derr_001_mod], 
                     labels=timesteps, notch=True, vert=True, patch_artist=True, boxprops=dict(alpha=0.8), showfliers=False)
    ax7[0,1].set_title('Euler_modified')

    ax7[0,2].boxplot([derr_00001_her, derr_0001_her, derr_001_her],
                     labels=timesteps, notch=True, vert=True, patch_artist=True, boxprops=dict(alpha=0.8), showfliers=False)
    ax7[0,2].set_title('Hermite')

    ax7[1,0].boxplot([derr_00001_rk2, derr_0001_rk2, derr_001_rk2], 
                     labels=timesteps, notch=True, vert=True, patch_artist=True, boxprops=dict(alpha=0.8), showfliers=False)
    ax7[1,0].set_title('RK2-Heun')

    ax7[1,1].boxplot([derr_00001_leap, derr_0001_leap, derr_001_leap], 
                     labels=timesteps, notch=True, vert=True, patch_artist=True, boxprops=dict(alpha=0.8), showfliers=False)
    ax7[1,1].set_title('Leapfrog')

    ax7[1,2].boxplot([derr_00001_rk4, derr_0001_rk4, derr_001_rk4], 
                     labels=timesteps, notch=True, vert=True, patch_artist=True, boxprops=dict(alpha=0.8), showfliers=False)
    ax7[1,2].set_title('RK4')

    ax7[2,0].boxplot([derr_00001_tsu, derr_0001_tsu, derr_001_tsu],
                        labels=tsu_tstep, notch=True, vert=True, patch_artist=True, boxprops=dict(alpha=0.8), showfliers=False)
    ax7[2,0].set_title('Tsunami')

    fig7.delaxes(ax2[2,1], ax2[2,2])

    fig7.suptitle('Relative Energy errors\n(M1=%.1f, M2=%.1f, e=%.1f, rp=%.2f, T=%.2f)'%(mass_1, mass_2, e, rp, Tperiod),
                   fontsize=52, fontweight='600')

    for i in range(3):
        for j in range(3):
            ax7[i,j].set_xticklabels(['0.00001', '0.0001', '0.001'])
            ax7[i,j].set_xlabel('Time Step')
            ax7[i,j].set_ylabel('|(E-E0)/E0|')
            ax7[i,j].set_yscale('log')
            ax7[i,j].yaxis.grid(True, which='major', alpha=0.5)
            ax7[i,j].set_prop_cycle(custom_cycler7)

    ax7[2,0].set_xticklabels(str(tsu_tstep))
       
    ax7[0,1].yaxis.set_minor_locator(locminy)
    ax7[0,1].yaxis.set_minor_formatter(mticker.NullFormatter())
    ax7[0,1].yaxis.set_major_locator(locmajy)
    ax7[0,1].yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

    ax7[0,2].yaxis.set_minor_locator(locminy)
    ax7[0,2].yaxis.set_minor_formatter(mticker.NullFormatter())
    ax7[0,2].yaxis.set_major_locator(locmajy)
    ax7[0,2].yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
    

    pdf.savefig(dpi=100)
    plt.close()

##############################################################################################################
    # ENERGY ERR VS INTEGRATOR (BOXPLOT)

    ints = np.array(['Euler_b', 'Euler_mod', 'Her', 'RK2', 'Leap', 'RK4', 'Tsu'])

    custom_cycler8 = (cycler(color=['tab:blue', 'tab:orange']))
    plt.rc('axes', prop_cycle=custom_cycler8)

    fig8, ax8 = plt.subplots(1, 3, figsize=(40, 17))
    gs8 = GridSpec(1,3)

    ax8[0].set_position([0.1, 0.1, 0.25, 0.7])  # Adjust these values as needed
    ax8[1].set_position([0.4, 0.1, 0.25, 0.7])
    ax8[2].set_position([0.7, 0.1, 0.25, 0.7])

    ax8[0].boxplot([derr_00001_base, derr_00001_mod, derr_00001_her, derr_00001_rk2, derr_00001_leap, derr_00001_rk4, derr_00001_tsu], 
                  labels=ints, notch=True, vert=True, patch_artist=True, boxprops=dict(alpha=0.8), showfliers=False)
    ax8[0].set_title('h=0.00001 (Tsu=%.6f)'%tsu_tstep[0])
    
    ax8[1].boxplot([derr_0001_base, derr_0001_mod, derr_0001_her, derr_0001_rk2, derr_0001_leap, derr_0001_rk4, derr_0001_tsu],
                  labels=ints, notch=True, vert=True, patch_artist=True, boxprops=dict(alpha=0.8), showfliers=False)
    ax8[1].set_title('h=0.0001 (Tsu=%.6f)'%tsu_tstep[1])
    
    ax8[2].boxplot([derr_001_base, derr_001_mod, derr_001_her, derr_001_rk2, derr_001_leap, derr_001_rk4, derr_001_tsu],
                  labels=ints, notch=True, vert=True, patch_artist=True, boxprops=dict(alpha=0.8), showfliers=False)
    ax8[2].set_title('h=0.001 (Tsu=%.6f)'%tsu_tstep[2])

    fig8.suptitle('Relative Energy errors\n(M1=%.1f, M2=%.1f, e=%.1f, rp=%.2f, T=%.2f)'%(mass_1, mass_2, e, rp, Tperiod), 
                  fontsize=52, fontweight='600')
    
    for i in range(3):
        # ax8[i,j].set_xticklabels(['0.0001', '0.001', '0.01'])
        ax8[i].set_xlabel('Integrator')
        ax8[i].set_ylabel('|(E-E0)/E0|')
        ax8[i].set_yscale('log')
        ax8[i].yaxis.grid(True, which='major', alpha=0.5)
        ax8[i].set_prop_cycle(custom_cycler8)

        ax8[i].yaxis.set_minor_locator(locminy)
        ax8[i].yaxis.set_major_locator(locmajy)
        ax8[i].yaxis.set_minor_formatter(mticker.NullFormatter())
        ax8[i].yaxis.set_major_formatter(mticker.LogFormatterSciNotation())


    pdf.savefig(dpi=100)
    plt.close()

##############################################################################################################

# # ENERGY ERR PLOTS (DIFF INT)

#     custom_cycler9 = (cycler(color=['orange','lightgreen', 'navy']) + cycler(linestyle=['-', '--', '-.']))
#     plt.rc('axes', prop_cycle=custom_cycler9)

#     fig9, ax9 = plt.subplots(2, 3, figsize=(40, 27))
#     gs9 = GridSpec(2,3)
#     # Plot position on x-y plane
#     ax9[0,0].plot(np.linspace(0, N_end*Tperiod, data_001_base[:, 4].shape[0]), np.log10(np.abs((data_001_base[:, 4]-Etot_0)/Etot_0))
#                   , alpha=0.8, label='h=0.001')
#     ax9[0,0].plot(np.linspace(0, N_end*Tperiod, data_0001_base[:, 4].shape[0]), np.log10(np.abs((data_0001_base[:, 4]-Etot_0)/Etot_0))
#                   , alpha=0.8, label='h=0.0001')
#     ax9[0,0].plot(np.linspace(0, N_end*Tperiod, data_00001_base[:, 4].shape[0]), np.log10(np.abs((data_00001_base[:, 4]-Etot_0)/Etot_0))
#                   , alpha=0.8, label='h=0.00001')
#     ax9[0,0].set_title('Euler_base')

#     ax9[0,1].plot(np.linspace(0, N_end*Tperiod, data_001_mod[:, 4].shape[0]), np.log10(np.abs((data_001_mod[:, 4]-Etot_0)/Etot_0))
#                     , alpha=0.8, label='h=0.001')
#     ax9[0,1].plot(np.linspace(0, N_end*Tperiod, data_0001_mod[:, 4].shape[0]), np.log10(np.abs((data_0001_mod[:, 4]-Etot_0)/Etot_0))
#                     , alpha=0.8, label='h=0.0001')
#     ax9[0,1].plot(np.linspace(0, N_end*Tperiod, data_00001_mod[:, 4].shape[0]), np.log10(np.abs((data_00001_mod[:, 4]-Etot_0)/Etot_0))
#                     , alpha=0.8, label='h=0.00001')
#     ax9[0,1].set_title('Euler_modified')

#     ax9[0,2].plot(np.linspace(0, N_end*Tperiod, data_001_rk2[:, 4].shape[0]), np.log10(np.abs((data_001_rk2[:, 4]-Etot_0)/Etot_0))
#                     , alpha=0.8, label='h=0.001')
#     ax9[0,2].plot(np.linspace(0, N_end*Tperiod, data_0001_rk2[:, 4].shape[0]), np.log10(np.abs((data_0001_rk2[:, 4]-Etot_0)/Etot_0))
#                     , alpha=0.8, label='h=0.0001')
#     ax9[0,2].plot(np.linspace(0, N_end*Tperiod, data_00001_rk2[:, 4].shape[0]), np.log10(np.abs((data_00001_rk2[:, 4]-Etot_0)/Etot_0))
#                     , alpha=0.8, label='h=0.00001')
#     ax9[0,2].set_title('RK2-Heun')

#     ax9[1,0].plot(np.linspace(0, N_end*Tperiod, data_001_leap[:, 4].shape[0]), np.log10(np.abs((data_001_leap[:, 4]-Etot_0)/Etot_0))
#                     , alpha=0.8, label='h=0.001')
#     ax9[1,0].plot(np.linspace(0, N_end*Tperiod, data_0001_leap[:, 4].shape[0]), np.log10(np.abs((data_0001_leap[:, 4]-Etot_0)/Etot_0))
#                     , alpha=0.8, label='h=0.0001')
#     ax9[1,0].plot(np.linspace(0, N_end*Tperiod, data_00001_leap[:, 4].shape[0]), np.log10(np.abs((data_00001_leap[:, 4]-Etot_0)/Etot_0))
#                     , alpha=0.8, label='h=0.00001')
#     ax9[1,0].set_title('Leapfrog')

#     ax9[1,1].plot(np.linspace(0, N_end*Tperiod, data_001_rk4[:, 4].shape[0]), np.log10(np.abs((data_001_rk4[:, 4]-Etot_0)/Etot_0))
#                     , alpha=0.8, label='h=0.001')
#     ax9[1,1].plot(np.linspace(0, N_end*Tperiod, data_0001_rk4[:, 4].shape[0]), np.log10(np.abs((data_0001_rk4[:, 4]-Etot_0)/Etot_0))
#                     , alpha=0.8, label='h=0.0001')
#     ax9[1,1].plot(np.linspace(0, N_end*Tperiod, data_00001_rk4[:, 4].shape[0]), np.log10(np.abs((data_00001_rk4[:, 4]-Etot_0)/Etot_0))
#                     , alpha=0.8, label='h=0.00001')
#     ax9[1,1].set_title('RK4')

#     for i in range(2):
#         for j in range(3):
#             ax9[i,j].set_xlabel('absolute time')
#             ax9[i,j].set_ylabel('$log_{10}|\Delta E/E0|$')
#             #ax2[i,j].set_yscale('log')
#             ax9[i,j].set_prop_cycle(custom_cycler9)
#             ax9[i,j].legend()

#     # ax212 = plt.subplot(gs2[1, -1])
#     # ax212.axis('off')  # Turn off the axes for the empty subplot
#     fig9.delaxes(ax9[1,-1])

#     fig9.suptitle('ΔE evolution', fontsize=52, fontweight='600')

#     pdf.savefig(dpi=100)
#     plt.close()

##############################################################################################################

    # # same but w/o euler_base
    # ints = np.array(['Euler_mod', 'RK2', 'Leap', 'RK4'])
    # fig9, ax9 = plt.subplots(1, 3, figsize=(40, 15))
    # gs9 = GridSpec(1,3)

    # ax9[0].boxplot([derr_00001_mod, derr_00001_rk2, derr_00001_leap, derr_00001_rk4], 
    #               labels=ints, notch=True, vert=True, patch_artist=True, boxprops=dict(alpha=0.8))
    # ax9[0].set_title('h=0.00001')
    
    # ax9[1].boxplot([derr_0001_mod, derr_0001_rk2, derr_0001_leap, derr_0001_rk4],
    #               labels=ints, notch=True, vert=True, patch_artist=True, boxprops=dict(alpha=0.8))
    # ax9[1].set_title('h=0.0001')
    
    # ax9[2].boxplot([derr_001_mod, derr_001_rk2, derr_001_leap, derr_001_rk4],
    #               labels=ints, notch=True, vert=True, patch_artist=True, boxprops=dict(alpha=0.8))
    # ax9[2].set_title('h=0.001')

    # fig9.suptitle('Relative Energy errors', fontsize=52, fontweight='600')
    
    # for i in range(3):
    #     # ax8[i,j].set_xticklabels(['0.0001', '0.001', '0.01'])
    #     ax9[i].set_xlabel('Integrator')
    #     ax9[i].set_ylabel('|(E-E0)/E0|')
    #     ax9[i].set_yscale('log')
    #     ax9[i].yaxis.grid(True, which='major', alpha=0.5)
    #     ax9[i].set_prop_cycle(custom_cycler8)

    # pdf.savefig(dpi=100)
    # plt.close()

    # ints = np.array(['RK2', 'Leap', 'RK4'])
    # fig9, ax9 = plt.subplots(1, 3, figsize=(40, 15))
    # gs9 = GridSpec(1,3)

    # ax9[0].boxplot([derr_00001_rk2, derr_00001_leap, derr_00001_rk4], 
    #               labels=ints, notch=True, vert=True, patch_artist=True, boxprops=dict(alpha=0.8))
    # ax9[0].set_title('h=0.00001')
    
    # ax9[1].boxplot([derr_0001_rk2, derr_0001_leap, derr_0001_rk4],
    #               labels=ints, notch=True, vert=True, patch_artist=True, boxprops=dict(alpha=0.8))
    # ax9[1].set_title('h=0.0001')
    
    # ax9[2].boxplot([derr_001_rk2, derr_001_leap, derr_001_rk4],
    #               labels=ints, notch=True, vert=True, patch_artist=True, boxprops=dict(alpha=0.8))
    # ax9[2].set_title('h=0.001')

    # fig9.suptitle('Relative Energy errors', fontsize=52, fontweight='600')
    
    # for i in range(3):
    #     # ax8[i,j].set_xticklabels(['0.0001', '0.001', '0.01'])
    #     ax9[i].set_xlabel('Integrator')
    #     ax9[i].set_ylabel('|(E-E0)/E0|')
    #     ax9[i].set_yscale('log')
    #     ax9[i].yaxis.grid(True, which='major', alpha=0.5)
    #     ax9[i].set_prop_cycle(custom_cycler8)

    # pdf.savefig(dpi=100)
    # plt.close()