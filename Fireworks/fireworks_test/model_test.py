import numpy as np
from fireworks.particles import Particles
import fireworks
import fireworks.ic as fic
import fireworks.nbodylib.dynamics as fd

from typing import Optional, Tuple
import numpy.typing as npt

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from tqdm import tqdm
import time
import gc
import os

try:
    import pyfalcon
    pyfalcon_load=True
except:
    pyfalcon_load=False


def ic_random_uniform(N: int, mass: list[float, float], pos: list[float, float], vel: list[float, float]) -> Particles:

    mass = np.random.uniform(low=mass[0], high=mass[1], size=N) # Generate 1D array of N elements
    pos = np.random.uniform(low=pos[0], high=pos[1], size=3*N).reshape(N,3) # Generate 3xN 1D array and then reshape as a Nx3 array
    vel = np.random.uniform(low=vel[0], high=vel[1], size=3*N).reshape(N,3) # Generate 3xN 1D array and then reshape as a Nx3 array

    return Particles(position=pos, velocity=vel, mass=mass)


def acc_dir_vepe(particles: Particles, softening: float=0.0):

    def acc_2body(i, j):
        #create cartesian component distance
        dx = particles.pos[i, 0] - particles.pos[j, 0]
        dy = particles.pos[i, 1] - particles.pos[j, 1]
        dz = particles.pos[i, 2] - particles.pos[j, 2]

        #modulus of the distance
        r = np.sqrt(dx*dx + dy*dy + dz*dz)

        #Cartesian component of the acceleration
        acc = np.zeros(3)
        acc[0] = particles.mass[i] * dx/(r**3)
        acc[1] = particles.mass[i] * dy/(r**3)
        acc[2] = particles.mass[i] * dz/(r**3)

        return acc

    N =  len(particles.mass)
    acc_body = np.zeros((len(particles), 3))
    for i in range(N):
        for j in range(N):
            if i != j:
                acc_body[j] += acc_2body(i, j)
    jerk = None
    pot = None
    return (acc_body, jerk, pot)

def acc_vect_vepe(particles: Particles, softening: float = 0.0):
    
    pos_x = particles.pos[:, 0] - particles.pos[:, 0].reshape(len(particles), 1)  #broadcasting of (N,) on (N,1) array, obtain distance along x in an (N,N) matrix
    pos_y = particles.pos[:, 1] - particles.pos[:, 1].reshape(len(particles), 1) 
    pos_z = particles.pos[:, 2] - particles.pos[:, 2].reshape(len(particles), 1)

    r = np.sqrt(pos_x**2 + pos_y**2 + pos_z**2)

    r[r==0]=1
    
    acc_x = pos_x/r**3 @ particles.mass #geometrical matrix vector product, returns a (N,) vector 
    acc_y = pos_y/r**3 @ particles.mass
    acc_z = pos_z/r**3 @ particles.mass

    acc = np.zeros((len(particles), 3))
    acc[:, 0] = acc_x
    acc[:, 1] = acc_y
    acc[:, 2] = acc_z

    jerk = None
    pot = None
    return (acc, jerk, pot)

def acc_onearray_vepe(particles: Particles, softening: float = 0.0):
    
    pos_x = particles.pos[:, 0] - particles.pos[:, 0].reshape(len(particles), 1)  #broadcasting of (N,) on (N,1) array, obtain distance along x in an (N,N) matrix
    pos_y = particles.pos[:, 1] - particles.pos[:, 1].reshape(len(particles), 1) 
    pos_z = particles.pos[:, 2] - particles.pos[:, 2].reshape(len(particles), 1)
    
    r = np.sqrt(pos_x**2 + pos_y**2 + pos_z**2)
    r[r==0]=1
    
    pos = np.concatenate((pos_x, pos_y, pos_z)).reshape((3,len(particles),len(particles)))
    acc = (pos/r**3 @ particles.mass).T 

    jerk = None
    pot = None
    return (acc, jerk, pot)

def acc_dir_gia(particles: Particles, jerk_bool = False, pot_bool = False):
    
    N = len(particles) #Extract number of particles
    acc = np.zeros(shape=(N,3)) #Alloc memory for the acceleration array

    if jerk_bool:
        jerk = np.zeros(shape=(N,3)) #Alloc memory for the acceleration array
    else:
        jerk = None

    if pot_bool:
        pot = np.zeros(N) #Alloc memory for the acceleration array
    else:
        pot=None

    ls = np.arange(N)

    for i in ls:
        for j in ls:

            if j != i :
                dr = particles.pos[i] - particles.pos[j] #r_i - r_j, vectors
                r_ij = np.sqrt(np.sum(dr*dr)) #Module of dr
    
                acc[i] -=  particles.mass[j] * (dr) / np.power(r_ij, 3) #Acceleration particle i
    
                if jerk_bool:
                    dv = particles.vel[i] - particles.vel[j] #v_i - v_j, vectors
                    jerk[i] -=  particles.mass[j] * (dv/np.power(r_ij,3) - 3*np.dot(dv, dr)*dr/np.power(r_ij,5))
    
                if pot_bool:
                    pot[i] -= particles.mass[j] / np.power(r_ij,2)

    return acc, jerk, pot

def acc_vect_gia(particles: Particles, jerk_bool = False, pot_bool = False):
    
    N = len(particles) #Extract number of particles

    # if jerk_bool:
    #     jerk = np.zeros(shape=(N,3)) #Alloc memory for the acceleration array
    # else:
    #     jerk = None

    # if pot_bool:
    #     pot = np.zeros(N) #Alloc memory for the acceleration array
    # else:
    #     pot=None

    # xT = np.tile(particles.pos.T.reshape(N,1), N).reshape(3,N,N) #Graph it for a 3x2 matrix -> 2x3x3
    # x =np.tile(x.T, N).reshape(3,N,N)
    # dx = xT - x
    
    # x_ij = np.tile(np.sqrt(np.sum(dx*dx, axis=0)), 3).reshape(3,N,N)
    # x_ij[ x_ij ==0 ] = 1 

    # m = np.tile(particles.mass,3*N).reshape(3,N,N)

    # acc = (-m * dx/x_ij).sum(axis=2).T

    xT , x = np.zeros(shape=(3,N,N)), np.zeros(shape=(3,N,N)) #The data structure is a 3xNxN tensor (k,i,j), where k is the spatial coordinate index
    xT[0], x[0] = np.tile(particles.pos[:,0], N).reshape(N,N).T, np.tile(particles.pos[:,0], N).reshape(N,N) #Forcasting for x coordinate
    xT[1], x[1] = np.tile(particles.pos[:,1], N).reshape(N,N).T, np.tile(particles.pos[:,1], N).reshape(N,N) #Forcasting for y coordinate
    xT[2], x[2] = np.tile(particles.pos[:,2], N).reshape(N,N).T, np.tile(particles.pos[:,2], N).reshape(N,N) #Forcasting for z coordinate

    dx = xT - x  #Compute all the possible difference vectors 
    x_ij = np.sqrt(np.sum(dx*dx, axis=0)) #Compute the all possible distances among particles
    dist = np.ones(shape=(3,N,N)) # 3xNxN tensor which contain the matrix of distances among al particles
    dist[0], dist[1], dist[2] = x_ij, x_ij, x_ij 
    dist[dist==0.]=1 #To avoid division by zero. By construction the only time that wuold happens wuold be when considering the same particle

    m = np.tile(particles.mass,3*N).reshape(3,N,N) #Forcasting of the particles mass into a tensor 3xNxN
    acc = (-m * dx/np.power(dist,3)).sum(axis=2).T #Compute the acceleration by summing over the j index of the tensor

    del xT, x, dx, x_ij

    if jerk_bool:
        vT , v = np.zeros(shape=(3,N,N)), np.zeros(shape=(3,N,N)) #The data structure is a 3xNxN tensor (k,i,j), where k is the velocity coordinate index
        vT[0], v[0] = np.tile(particles.vel[:,0], N).reshape(N,N).T, np.tile(particles.vel[:,0], N).reshape(N,N) #Forcasting for x coordinate
        vT[1], v[1] = np.tile(particles.vel[:,1], N).reshape(N,N).T, np.tile(particles.vel[:,1], N).reshape(N,N) #Forcasting for y coordinate
        vT[2], v[2] = np.tile(particles.vel[:,2], N).reshape(N,N).T, np.tile(particles.vel[:,2], N).reshape(N,N) #Forcasting for z coordinate

        dv = vT - v  #Compute all the possible difference vectors 

        xv = np.zeros(shape=(3,N,N)), #Alloc memory for the forcasting of v_ij * r_ij, tensor 3xNxN
        dot_xv = (dx * dv).sum(axis=0) #Commpute v_ij * r_ij
        xv[0], xv[1], xv[2] = dot_xv, dot_xv, dot_xv 

        jerk =  np.sum(-m * (dv/np.power(dist,3) - 3*xv*dx/np.power(dist,5)), axis=2).T #Compute Jerkby summing over the j index of the tensor

        del vT, v, dv, dor_xv, xv

    else:
        jerk = None

    if pot_bool:
        pot = np.sum(m[0] / np.power(dist[0],2), axis=1)

    else:
        pot = None

    del m, dist
    gc.collect()

    return acc, jerk, pot

def acc_opt_gia(particles: Particles, jerk_bool = False, pot_bool = False):
   
    N = len(particles) #Extract number of particles

    
    xT = np.tile(particles.pos.T.reshape(3*N,1), N).reshape(3,N,N) #Graph it for a 3x2 matrix -> 2x3x3
    x =np.tile(particles.pos.T, N).reshape(3,N,N)
    dx = xT - x  #Compute all the possible difference vectors 
    
    x_ij = np.sqrt(np.sum(dx*dx, axis=0)) #Compute the all possible distances among particles
    x_ij[x_ij==0] = 1 #To avoid division by zero. By construction the only time that would happens wuold be when considering the same particle
    dist = np.tile(x_ij,3).reshape(3,N,N) # 3xNxN tensor which contain the matrix of distances among al particles

    m = np.tile(particles.mass,3*N).reshape(3,N,N) #Forcasting of the particles mass into a tensor 3xNxN
    acc = (-m * dx/np.power(dist,3)).sum(axis=2).T #Compute the acceleration by summing over the j index of the tensor

    del xT, x, dx, x_ij
    gc.collect()

    

    if jerk_bool:
        vT , v = np.zeros(shape=(3,N,N)), np.zeros(shape=(3,N,N)) #The data structure is a 3xNxN tensor (k,i,j), where k is the velocity coordinate index
        vT[0], v[0] = np.tile(particles.vel[:,0], N).reshape(N,N).T, np.tile(particles.vel[:,0], N).reshape(N,N) #Forcasting for x coordinate
        vT[1], v[1] = np.tile(particles.vel[:,1], N).reshape(N,N).T, np.tile(particles.vel[:,1], N).reshape(N,N) #Forcasting for y coordinate
        vT[2], v[2] = np.tile(particles.vel[:,2], N).reshape(N,N).T, np.tile(particles.vel[:,2], N).reshape(N,N) #Forcasting for z coordinate

        dv = vT - v  #Compute all the possible difference vectors 

        xv = np.zeros(shape=(3,N,N)), #Alloc memory for the forcasting of v_ij * r_ij, tensor 3xNxN
        dot_xv = (dx * dv).sum(axis=0) #Commpute v_ij * r_ij
        xv[0], xv[1], xv[2] = dot_xv, dot_xv, dot_xv 

        jerk =  np.sum(-m * (dv/np.power(dist,3) - 3*xv*dx/np.power(dist,5)), axis=2).T #Compute Jerkby summing over the j index of the tensor

        del vT, v, dv, dor_xv, xv
        gc.collect()


    else:
        jerk = None

    if pot_bool:
        pot = np.sum(m[0] / np.power(dist[0],2), axis=1)

    else:
        pot = None

    del m, dist
    gc.collect()
    
    return acc, jerk, pot


# def acceleration_pyfalcon(particles: Particles, softening: float =0.) \
#         -> Tuple[npt.NDArray[np.float64],Optional[npt.NDArray[np.float64]],Optional[npt.NDArray[np.float64]]]:

#     if not pyfalcon_load: return ImportError("Pyfalcon is not available")

#     acc, pot = pyfalcon.gravity(particles.pos,particles.mass,softening)
#     jerk = None

#     return acc, jerk, pot


N_list = [10, 50, 100, 500, 1000]
func_list = [acc_dir_vepe,
             acc_dir_gia,
             acc_vect_vepe,
             acc_vect_gia,
             acc_onearray_vepe,
             acc_opt_gia]#,
            #  acceleration_pyfalcon]
dt=[]


for N in N_list:

    particles = fic.ic_random_uniform(N=N,
                                      pos=[-1.,1.], 
                                      vel=[-1.,1.], 
                                      mass=[0.001,1.])
    for func in func_list:
        
        t1=time.perf_counter()
        _ = func(particles)
        t2=time.perf_counter()
        dt.append(t2-t1) # time elapsed from t1 to t2 in s
        
        # create and write to file
        with open("func_dt.txt", "a") as f:
            f.write(func.__name__ + "\n")
            f.write(str(dt) + "\n")

dt = np.reshape(dt, (5,6)).T



# figs = []
pdf = matplotlib.backends.backend_pdf.PdfPages("model_test_plots.pdf")

# for i in range(dt.shape[0]):
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 15))

for i in range(dt.shape[0]):
    # for j in range(dt.shape[1]):
    ax[0].plot(N_list, dt[i], 'o-', label=func_list[i].__name__)
    ax[1].plot(N_list, dt[i], 'o-', label=func_list[i].__name__)
    # ax[1,0].plot(N_list, dt[i], 'o-', label=func_list[i].__name__)

    ax[0].legend(loc='best')
    ax[0].grid(linestyle='dotted')
    ax[0].set_xlabel('Number of particles')
    ax[0].set_ylabel('Time spent for acceleration estimation (s)')

    ax[1].legend(loc='best')
    ax[1].grid(linestyle='dotted')
    ax[1].set_xlabel('Number of particles')
    ax[1].set_ylabel('Time spent for acceleration estimation (s) [log scale]')
    ax[1].yaxis.tick_right()
    ax[1].set_yscale('log', base=10)

    # ax[1,0].legend(loc='best')
    # ax[1,0].grid(linestyle='dotted')
    # ax[1,0].set_xlabel('Number of particles')
    # ax[1,0].set_ylabel('Time spent for acceleration estimation (s)')
    # ax[1,0].set_ylim(0, 8.)
    # Add an insert in ax[0] with ylim (-2,30)
axins = ax[0].inset_axes([0.3, 0.6, 0.4, 0.4])
axins.plot(N_list, dt[i], 'o-')
axins.set_ylim(-2, 30)
#axins.legend(loc='best')
axins.grid(linestyle='dotted')
    

#fig.suptitle(func_list[i].__name__)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
# figs.append(fig)
pdf.savefig(fig, dpi=300)

pdf.close()
