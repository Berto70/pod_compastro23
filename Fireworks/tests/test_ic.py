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
