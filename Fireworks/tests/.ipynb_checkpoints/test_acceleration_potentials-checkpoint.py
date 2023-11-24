import pytest
import numpy as np
import fireworks.nbodylib.potentials as fpot
from fireworks.particles import Particles
import fireworks.nbodylib.integrators as fint


def test_point_mass():
    """
    Check that the Point Mass acceleration returns the expected value
    """

    p=Particles([[1,0,0],],[[0,1,0,]],[1e-10,])

    pmpot = fpot.Point_Mass(Mass=1)

    acc_func = pmpot.acceleration

    expected_acc = np.array([-1.,0.,0.])
    acc_estimated,_,_ = acc_func(p)

    assert np.all(np.abs(acc_estimated[0]-expected_acc) <=1e-10)

def test_point_mass_in_integrator():
    """
    Check that the Point Mass acceleration can be called in a in integrator
    """

    p=Particles([[1,0,0],],[[0,1,0,]],[1e-10,])
    pmpot = fpot.Point_Mass(Mass=1)

    tstep=0.01

    #Test that the integration call does not raise an error
    try:
        par,teffective,_,_,_=fint.integrator_template(particles=p,
                                        tstep=tstep,
                                        acceleration_estimator=pmpot.acceleration,
                                        softening=0.)
    except:
        pytest.fail("Unexpected MyError ..")
