import numpy as np
import jax.numpy as jnp
from jax import grad, vmap, jit, pmap

class colors():
    GREENS = [(143,255,140),
            (40,238,40),
            (26,192,22),
            (0,157,0),
            (0,126,0),
            (19,91,0)]
    BLUES = [(180,229,241),
            (96,189,224),
            (34,161,214),
            (34,133,166),
            (31,103,131),
            (26,79,94)]
    GREEN_BLUE = [(165,255,70), #(40,238,40),
          (0,157,0),
          (15,15,15),
          (0,72,127),
          (0,26,80)]
    GREEN_WHITE = [ 
        GREENS[1],
        GREENS[1],
        (15,15,15),
        (255,255,255),
        (255,255,255)]
    BLUE_WHITE = [
            BLUES[0],
            BLUES[0],
          (15,15,15),
          (255,255,255),
          (255,255,255)]
    
def block(M, np=np):
    return np.vstack([np.hstack(m) for m in M])

def rotation2(theta, np=np):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

def datehour():
    return 'datehour'


def grid(xlim, ylim, xnum, ynum, np=np):
    x = np.linspace(*xlim, xnum)
    y = np.linspace(*ylim, ynum)
    out = np.array(np.meshgrid(x,y))
    out = out.reshape(2, xnum*ynum).T
    return out, (xnum, ynum)

""" Numerical range of matrix M = [[A,B],[C,D]] """
def numerical_range(A, N=int(1e5)):
    xs = sphere(N, A.shape[0])
    W = [(A @ x) @ jnp.conj(x) for x in xs]
    return W

def block_to_2x2(J, x, y):
    A,B,C,D = J
    quad = lambda x,A,y: jnp.conj(x) @ A @ y
    M = jnp.asarray([[quad(x, A, x), quad(x, B, y)], \
                   [quad(y, C, x), quad(y, D, y)]])
    return M

@jit
def qnum(J,x,y):
    M = block_to_2x2(J,x,y)
    return eig2x2(M)
qnum = jit(vmap(qnum, (None, 0, 0)))

def eig2x2(A):
    (a,b),(c,d) = A
    root = lambda tr, det: (tr/2 + jnp.sqrt(tr**2 - 4*det)/2, 
                            tr/2 - jnp.sqrt(tr**2 - 4*det)/2 )
    return root(a+d, a*d-b*c)

def sphere(num_samples, n):
    x = np.random.randn(num_samples, n) \
        + np.random.randn(num_samples, n)*1j
    x /= jnp.linalg.norm(x, axis=1)[:,jnp.newaxis]
    return x

def quadratic_numerical_range(M, num=int(1e4)):
    (A,B),(C,D) = M
    xs = sphere(num, A.shape[0])
    ys = sphere(num, D.shape[0])
    return jnp.hstack(qnum((A,B,C,D), xs, ys))

def numrange(M, num=int(1e4)):
    return quadratic_numerical_range(M)

