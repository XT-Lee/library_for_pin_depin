import numpy as np
def gen_a_line():
    x = np.linspace(0, 1, 10)
    y = -x
    y2 = np.sin(x)
    z = np.stack((x, y, y2), axis=0)
    #np.concatenate((x, y))
    print(f"z: {z}\n")
    print(f"z.shape: {z.shape}\n")
    return z.T

def gen_a_trajectory_with_oscillation_and_jump():
    x = np.random.rand(2,100)
    """x-=0.5
    x*=10"""
    x[0,33:]+=5
    x[0,66:]+=5
    x[1,33:]-=5
    x[1,66:]-=5
    #print(f"x: {x}\n")
    return x.T

def gen_a_trajectory_with_period():
    x = np.linspace(0, 10, 100)
    y = 5*np.sin(2*x)
    y2 = 5*np.cos(2*x)
    z = np.stack((y, y2), axis=0)
    return z.T