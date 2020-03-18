import numpy as np

def block(M):
    return np.vstack([np.hstack(m) for m in M])
