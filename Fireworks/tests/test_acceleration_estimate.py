import pytest
import numpy as np
import fireworks.nbodylib.dynamics as fdyn
from fireworks.particles import Particles



def test_acceleration_2body():
    """
    Simple two body case of two bodies alond the x axis (at x=0 and x=1) of mass 1 at a distance 1,
    therfore the acceleration on the first body is  +1 and on the secondy body -1.

    """
    facc_list = [fdyn.acceleration_direct, fdyn.acceleration_direct_vectorized, fdyn.acceleration_pyfalcon,]

    pos = np.array([[0.,0.,0.],[1.,0.,0.]])
    vel = np.zeros_like(pos)
    mass = np.ones(len(pos))

    part = Particles(pos,vel,mass)

    true_acc = np.array([[1., 0., 0.],[-1., 0., 0.]])

    for facc in facc_list:
        acc,_,_=facc(part)

        dx = np.abs(acc-true_acc)

        assert np.all(dx<=1e-11)

def test_acceleartion_row():
    """
    Another simple case in which  we have N bodies along the x axis,
    each body is at a distance r from the origin (the first has r=0),
    the mass of each body is equal to r^2.
    Therefore the acceleartion on the body at the centre r=0 is equal to the number of the other bodies

    """


    facc_list = [fdyn.acceleration_direct, fdyn.acceleration_direct_vectorized, fdyn.acceleration_pyfalcon,]

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
        acc,_,_=facc(part)
        dx = acc[0,0]-acc_true_0

        assert pytest.approx(dx, 1e-10) == 0.
        # pytest approx is used to introduc a tollerance in the comparison  (in this case 1e-10)

def test_acceleration_2body_jerk():
    """
    Simple 2 body problem. The two body are alligned on the x-axis, with position x_0=0 and x_1=1,
    the both have mass m=1 and the their velocity (also on the x_axis) are v_x0=0 and v_x1=-1. 
    The expected acceleration are a_0 = (1, 0, 0) and a_1 = (-1, 0, 0)   
    The expected jerk are j_0=(-2, 0, 0) and j_1=(2, 0, 0)
    """
    facc_list = [fdyn.acceleration_direct_vectorized]

    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    vel = np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    mass= np.ones(len(pos))
    part = Particles(pos, vel, mass)

    acc_true = np.array([[1.0, 0.0, 0.0],[-1.0, 0.0, 0.0]])
    jerk_true = np.array([[2.0, 0.0, 0.0],[-2.0, 0.0, 0.0]])

    for facc in facc_list:
        acc,jerk,_ = facc(part, return_jerk=True)

        d_ax = np.abs(acc - acc_true)
        d_jx = np.abs(jerk - jerk_true)

        assert np.all(d_ax<=1e-11)
        assert np.all(d_jx<=1e-11)

def test_acceleration_2body_circular_orbit_jerk():
    """
    Simple 2 body problem, where the second object is less massive and orbit around the first more massive one. 
    The two body are alligned on the y=x line, with:
    position: r_0 = (0, 0, 0) and r_1 = (1, 1, 0)
    velocity: v_0=(0, 0, 0), v_1=(-1, 1, 0) 
    mass:     m_0=100 and m_1=1
    
    The expected acceleration:  a_0 = (1, 1, 0)*(2**(-3/2)), a_1 = (-100, -100, 0)*(2**(-3/2)) 
    The expected jerk:          j_0 = (1, -1, 0)*(2**(-3/2)), j_1 = (-100, 100, 0)*(2**(-3/2))
    """
    facc_list = [fdyn.acceleration_direct_vectorized]

    pos = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    vel = np.array([[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0]])
    mass= np.array([100.0, 1.0])
    part = Particles(pos, vel, mass)

    acc_true = np.array([[1.0, 1.0, 0.0],[-100.0, -100.0, 0.0]])*(2**(-3/2))
    jerk_true = np.array([[-1.0, 1.0, 0.0],[100.0, -100.0, 0.0]])*(2**(-3/2))

    for facc in facc_list:
        acc,jerk,_ = facc(part,  return_jerk=True)

        d_ax = np.abs(acc - acc_true)
        d_jx = np.abs(jerk - jerk_true)

        assert np.all(d_ax<=1e-11)
        assert np.all(d_jx<=1e-11)


