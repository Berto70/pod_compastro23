import numpy as np
import matplotlib.pyplot as plt
import fireworks.nbodylib.dynamics as fdyn
from fireworks.particles import Particles
from fireworks.ic import ic_random_uniform as ic
import time
from tqdm import tqdm

# overwrite let's you save the results in a file
# 
overwrite= False

def get_time(func,x):
    t1=time.perf_counter()
    func(x)
    t2=time.perf_counter()
    dt=t2-t1 # time elapsed from t1 to t2 in s

    return dt

# Select a grid of values for the number of particles to test.
N   = [2] + [i for i in range(100,10_000,500)]
#N = [1,100,1e3,5e3,1e4]
# For each N, generate random initial conditions using the function you added in fireworks in task A.
position_bound = [-100,200]
velocity_bound = position_bound*2
mass_bound  = [0.1,1e3]

timings_direct     = []
timings_vectorised = []
timings_pyfalcon   = []

for n in tqdm(N,desc="Iterating over N",unit="N") :
   
    # ic returns already an instance of Particles
    part = ic(N=int(n) , position_bound=position_bound, velocity_bound=velocity_bound, mass_bound=mass_bound)

    # estimate the particles acceleration using the three implemented method in the submodule
    # evaluate the time required to complete the acceleration estimate.

    timings_direct.append(get_time(func=fdyn.acceleration_direct,x=part))
    timings_vectorised.append(get_time(func=fdyn.acceleration_direct_vectorised,x=part))
    timings_pyfalcon.append(get_time(func=fdyn.acceleration_pyfalcon,x=part))

# Maybe put a check if this file already exists

if overwrite:

    with open("benchmark_results2.txt","a") as f:
        f.write(f"{timings_direct}\n f{timings_vectorised}\n f{timings_pyfalcon}")


plt.plot(N,timings_direct,marker="o",label="direct")
plt.plot(N,timings_vectorised,marker="o",label="vectorised")
plt.plot(N,timings_pyfalcon,marker="o",label="pyfalcon")

plt.grid()
plt.xlabel("N")
plt.ylabel("log(Time) [s]")
plt.yscale("log")
plt.xticks([2]+[i for i in range(200,1001,200)])


plt.legend()

plt.savefig("benchmark_plot2.png")

plt.show()






