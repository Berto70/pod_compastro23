import pytest
import numpy as np
import fireworks.nbodylib.dynamics as fdyn
from fireworks.particles import Particles
from fireworks.ic import ic_random_normal
import fireworks.nbodylib.integrators as fint



def simple_test(integrator):
    """
    Simple test template to test an integrator
    The idea is very simple, assume an acceleration function that returns always 0
    therefore the after an integration step the velocity remains the same
    and the position is pos+vel*dt where dt is the effective integration timestep

    """

    N = 10
    par = ic_random_normal(N,mass=1)
    tstep = 1

    par_old = par.copy() # Copy the initial paramter
    print(par.vel[0])
    print(par.pos[0])

    # Just return a Nx3 filled with 0 + two None  (Jerk and potential)
    acc_fake = lambda particles, softening: (np.zeros_like(particles.pos), None, None)

    par,teffective,_,_,_=integrator(particles=par,
                                               tstep=tstep,
                                               acceleration_estimator=acc_fake,
                                               softening=0.)

    pos_test = par_old.pos + par_old.vel*teffective

    # Test velocity, we expect no change in velocity (we use a very small number instead of 0)
    # in order to take into account possible round-off errors
    assert np.all( np.abs(par.vel -par_old.vel) <= 1e-10)

    #Test particles
    assert np.all( np.abs(par.pos - pos_test) <= 1e-10)

def test_integrator_template():
    """
    Apply the simple_test the integrator_template

    """

    simple_test(fint.integrator_template)

def test_integrator_velocity_velrlet():
    """
    Apply the simple_test the integrator_template

    """

    simple_test(fint.integrator_velocity_verlet)

