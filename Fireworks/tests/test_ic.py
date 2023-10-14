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