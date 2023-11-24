import pytest
import numpy as np
import fireworks.ic as fic


def test_ic_random_normal():
    """
    Simple test for the method ic_random_normal
    """

    N=100
    mass=10.
    particles = fic.ic_random_normal(N,mass=mass)

    assert len(particles)==N #Test if we create the right amount of particles
    assert np.all(particles.mass==10.) #Test if the mass of the particles is set correctly

def test_ic_random_uniform():
    """
    simple test for the method ic_random_unifirm
    """

    N=100
    low_mass, high_mass = 0.001, 1.0
    low_pos, high_pos   =  -1.0,  1.0
    low_vel, high_vel   =  -1.0,  1.0
    
    particles = fic.ic_random_uniform(N=N, low_mass=low_mass, high_mass=high_mass, low_pos=low_pos, high_pos=high_pos, low_vel=low_vel, high_vel=high_vel)

    assert len(particles)==N #Test if we create the right amount of particles
    assert np.max(particles.mass) <= high_mass and low_mass <= np.min(particles.mass) #test if the mass are withing the boundaries 
    assert np.max(particles.pos) <= high_pos and low_pos <= np.min(particles.pos) #test if the pos are withing the boundaries 
    assert np.max(particles.vel) <= high_pos and low_vel <= np.min(particles.vel) #test if the vel are withing the boundaries 

def test_ic_two_body_circular():
    """
    Simple test for equal mass stars in a circular orbit
    """
    mass1=1.
    mass2=1.
    Mtot=mass1+mass2
    rp=2.
    particles = fic.ic_two_body(mass1,mass2,rp=rp,e=0.)

    assert pytest.approx(particles.vel[0,1],1e-10) == -1./Mtot
    assert pytest.approx(particles.vel[1,1],1e-10) == 1./Mtot

def test_ic_two_body_parabolic():
    """
    Simple test for equal mass stars in a parabolic orbit
    """
    mass1=1.
    mass2=1.
    Mtot=mass1+mass2
    rp=2.
    particles = fic.ic_two_body(mass1,mass2,rp=rp,e=1.)

    assert pytest.approx(particles.vel[0,1],1e-10) == -np.sqrt(2)/Mtot
    assert pytest.approx(particles.vel[1,1],1e-10) == np.sqrt(2)/Mtot

def test_ic_two_body_hyperbolic():
    """
    Simple test for equal mass stars in a hyperbolic orbit
    """
    mass1=1.
    mass2=1.
    Mtot=mass1+mass2
    rp=2.
    particles = fic.ic_two_body(mass1,mass2,rp=rp,e=3.)

    assert pytest.approx(particles.vel[0,1],1e-10) == -2./Mtot
    assert pytest.approx(particles.vel[1,1],1e-10) == 2./Mtot
