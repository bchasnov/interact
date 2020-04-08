import jax.numpy as np


def init(*args):
    return 0, None

def update(state, *args):
    k, win = state

    print(k)
    return k+1, win

