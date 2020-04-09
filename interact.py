





"""
Interact with an optimization algorithm in real-time.

There are two layers:
    stochastic (init):    
     samples initial condition of the algorithm, 
     either initial state, or sets the seed 
     for the iid noise of the whole trial. 
    deterministic (update):
     uses initial condition to run the algorithm

Within the deterministic update, 
we have various 


There is a strong emphasis on visualising and 
communicating the progress and outcome to the user.

The user must be allowed to interact with the code
in response to the progress of the algorithm, 
as well changing its initializations.

"""



""" As an example, we consider a scalar
polynomial game from the first example in Section 5
of (Chasnov 2020).
"""

defaultConfig = dict(a=-1.1,d=1.2,gamma1=0.01,gamma2=0.02)

def init(a, d, gamma1, gamma2, np=np):

    state = (1,1)

    def update(k, state):
        x, y = state
        next_x = x + gamma1*(a*x - y + x**3)
        next_y = y + gamma2*(d*y + 2*x)
    
        return (next_x, next_y)

    def info(state):
        x,y = state
        J = np.array([[a+1/3*x**2, -1],[2,d]])
        g = np.array([a*x - y + x**3, d*y + 2*x])
        eigs = np.linalg.eigvals(J)
        return g, J, eigs

    return state, update, info

global _modified
def load(f):
    """ Move to utils """
    global _modified
    _m = _modified
    _modified = os.time.something()
    if _modified:
        spec = importlib.util.spec_from_file_location("params",f)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    # Not modified
    return False

def instance(filename, config):
    module = load(filename)

    def tick(k):
        module_updated = load(filename)
        if module_updated:
            state, update, info = module.init(**config)
        else:
            state = module.update(state)

    return 0, tick

instance("params.py", defaultConfig)


    
    


