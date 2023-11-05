import numpy as np
import matplotlib.pyplot as plt
import fireworks.nbodylib.dynamics as fdyn
from fireworks.particles import Particles
from fireworks.ic import ic_random_uniform as ic
import time
from tqdm import tqdm

# Select a grid of values for the number of particles to test.
N   = np.arange(1,1e3,100)

# For each N, generate random initial conditions using the function you added in fireworks in task A.
position_bound = [-100,200]
velocity_bound = position_bound*2
mass_bound  = [0.1,1e3]

timings = []
for n in tqdm(N,desc="Iterating over N",unit="N") :
   
    # ic returns already an instance of Particles
    part = ic(N=int(n) , position_bound=position_bound, velocity_bound=velocity_bound, mass_bound=mass_bound)

    print("particle",part)
    # estimate the particles acceleration using the three implemented method in the submodule
    # evaluate the time required to complete the acceleration estimate.

    fdyn.acceleration_direct
    fdyn.acceleration_direct_vectorised
    fdyn.acceleration_pyfalcon
    t1=time.perf_counter()
    func(part)
    t2=time.perf_counter()
    dt=t2-t1 # time elapsed from t1 to t2 in s

    timings.append(dt)

    print(f"Function {func} took {dt} seconds")

    with open("benchmark_results.txt","a") as f:
        f.write(f"Function {func} took {dt} seconds\n")


print("ho finito")






