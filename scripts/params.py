import jax.numpy as np

def init(*args):
    return np.array([0])

def update(k, *args):
    print(k*3,'hello world')
    return np.array([k])

