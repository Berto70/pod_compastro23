import pytest
import numpy as np
import fireworks.ic as fic

# def test_ic_random_uniform():
#     """
#     Simple test for the method ic_uniform_normal
#     """
#     N = 100
#     mass_l = 0.1
#     mass_u = 10.
#     pos_l = -10.
#     pos_u = 10.
#     vel_l = -10.
#     vel_u = 10.

#     particles = fic.ic_random_uniform(N, mass_l = mass_l, mass_u = mass_u, pos_l = pos_l, pos_u = pos_u, vel_l = vel_l, vel_u = vel_u)

#     assert len(particles) == N #Test if we create the right amount of particles
#     assert np.min(particles.mass) >= mass_l
#     assert np.max(particles.mass) <= mass_u
#     assert np.min(particles.pos) >= pos_l
#     assert np.max(particles.pos) <= pos_u
#     assert np.min(particles.vel) >= vel_l
#     assert np.min(particles.vel) <= vel_u

def test_ic_random_uniform():
    """
    Simple test for the method ic_uniform_normal
    """
    N = 100
    mass = [0, 10]
    pos = [-10, 10]
    vel = [-10, 10]

    particles = fic.ic_random_uniform(N, mass=mass, pos=pos, vel=vel)

    assert len(particles) == N #Test if we create the right amount of particles
    assert np.min(particles.mass) >= mass[0]
    assert np.max(particles.mass) <= mass[1]
    assert np.min(particles.pos) >= pos[0]
    assert np.max(particles.pos) <= pos[1]
    assert np.min(particles.vel) >= vel[0]
    assert np.min(particles.vel) <= vel[1]


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
