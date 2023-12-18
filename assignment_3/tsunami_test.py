import numpy as np
from matplotlib import pyplot as plt, animation as animation, ticker as mticker
from tqdm.notebook import tqdm

from fireworks.particles import Particles
import fireworks.ic as ic
import fireworks.nbodylib.dynamics as fnd
import fireworks.nbodylib.integrators as fni

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15), layout='tight')

# for func, integrator, color in zip((fni.integrator_tsunami, fni.integrator_leapfrog), ('TSUNAMI', 'leapfrog'), ('blue', 'red')): 
for func, integrator, color in zip((fni.integrator_tsunami,), ('TSUNAMI',), ('blue',)): 

    for row, e in enumerate([0.5, 0.9, 0.99]):

        rp = 4
        a = rp / (1 - e)
        particles = ic.ic_two_body(mass1=1., mass2=2, e=e, rp=rp)
        Tperiod = 2 * np.pi * np.sqrt(a**3 / (particles.mass[0] + particles.mass[1]))
        tstart=0
        tintermediate=np.linspace(start=0, stop=10*Tperiod, num=100000)
        tcurrent=0

        tstart=0
        tintermediate=np.linspace(start=0, stop=10*Tperiod, num=100000)
        tcurrent=0
        pos_list=[]
        vel_list=[]
        energy_list=[]
        for t in tqdm(tintermediate):
            tstep=t-tcurrent
            if integrator=='TSUNAMI':    
                if tstep <=0: continue # continue means go to the next step (i.e. next t in the array)  
                particles,_,_,_,_= func(particles,tstep)
            else:
                particles,_,_,_,_ = func(particles, tstep, fnd.acceleration_direct_vectorized)
            # Save the particles positions and velocities and energy
            pos_list.append(particles.pos.copy())
            vel_list.append(particles.vel.copy())
            Etot, _, _ = particles.Etot()
            energy_list.append(Etot)
            # Here we can save stuff, plot stuff, etc.
            tcurrent=tcurrent+efftime
        
        pos_x_1, pos_y_1 = np.array(pos_list)[:, 0][:, 0], np.array(pos_list)[:, 0][:, 1] 
        pos_x_2, pos_y_2 = np.array(pos_list)[:, 1][:, 0], np.array(pos_list)[:, 1][:, 1] 
    
        #position plot
        axs[row, 0].plot(pos_x_1, pos_y_1, label=f'{integrator}', color=f'{color}')
        axs[row, 0].plot(pos_x_2, pos_y_2, color=f'{color}')
        axs[row, 0].set_xlabel('X')
        axs[row, 0].set_ylabel('Y')
        axs[row, 0].set_title(f'e={e}')
        axs[row, 0].legend()
        
        #energy plot
        axs[row, 1].plot(energy_list, label=f'{integrator}', color=f'{color}')
        axs[row, 1].set_xlabel('time')
        axs[row, 1].set_ylabel('E')
        axs[row, 1].legend()
        
        #energy error
        energy_array = np.array(energy_list)
        error = np.abs(energy_array - energy_array[0])/energy_array[0]
        axs[row, 2].set_xlabel('time')
        axs[row, 2].plot(error, label=f'{integrator}', color=f'{color}')
        axs[row, 2].set_ylabel(r'$\frac{|E-E_0|}{E_0}$')
        axs[row, 2].legend()