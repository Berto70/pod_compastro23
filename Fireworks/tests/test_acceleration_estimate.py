import pytest
import numpy as np
import fireworks.nbodylib.dynamics as fdyn
from fireworks.particles import Particles



def test_acceleration_2body():
    """
    Simple two body case of two bodies alond the x axis (at x=0 and x=1) of mass 1 at a distance 1,
    therfore the acceleration on the first body is  +1 and on the secondy body -1.

    """
    facc_list = [fdyn.acceleration_pyfalcon,fdyn.acceleration_direct,fdyn.acceleration_direct_vectorised]

    pos = np.array([[0.,0.,0.],[1.,0.,0.]])
    vel = np.zeros_like(pos)
    mass = np.ones(len(pos))

    part = Particles(pos,vel,mass)

    true_acc = np.array([[1., 0., 0.],[-1., 0., 0.]])

    for facc in facc_list:
        acc,_,_=facc(part)
        print("2 body test", facc)
        dx = np.abs(acc-true_acc)

        assert np.all(dx<=1e-11)

def test_acceleartion_row():
    """
    Another simple case in which  we have N bodies along the x axis,
    each body is at a distance r from the origin (the first has r=0),
    the mass of each body is equal to r^2.
    Therefore the acceleartion on the body at the centre r=0 is equal to the number of the other bodies

    """


    #facc_list = [fdyn.acceleration_pyfalcon,fdyn.acceleration_direct,fdyn.acceleration_direct_vectorised]
    facc_list = [fdyn.acceleration_pyfalcon,]

    x = np.array(np.arange(11),dtype=float)
    y = np.zeros_like(x) 
    z = np.zeros_like(x)

    pos = np.vstack([x,y,z]).T
    vel = np.zeros_like(pos)
    mass = x*x
    mass[0]=1
    part = Particles(pos,vel,mass)

    acc_true_0 = len(mass)-1

    for facc in facc_list:
        # Diego: aggiungo qualche print
        print(f"facc is {facc}")
        acc,_,_=facc(part)
        dx = acc[0,0]-acc_true_0

        assert pytest.approx(dx, 1e-10) == 0.
        # pytest approx is used to introduce a tollerance in the comparison  (in this case 1e-10)

