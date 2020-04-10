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
import numpy as np

""" As an example, we consider a scalar
polynomial game from the first example in Section 5
of (Chasnov 2020).
"""

defaultConfig = dict(a=-1.1,d=1.2,gamma1=0.01,gamma2=0.02, initial_state=[1,1])

def init(a, d, gamma1, gamma2, initial_state, np=np):

    def update(k, state):
        x, y = state
        next_x = x - gamma1*(a*x - y + x*x*x)
        next_y = y - gamma2*(d*y + 2*x)
        return (next_x, next_y)

    def info(state):
        x,y = state
        J = np.array([[a+1/3*x*x, -1],[2,d]])
        g = np.array([a*x - y + x**3, d*y + 2*x])
        eigs = np.linalg.eigvals(J)
        return g, J, eigs

    return initial_state, update, info

if __name__ == '__main__':
    num_iter = int(1e3)
    state, update, info = init(**defaultConfig)
    for k in range(num_iter):
        state = update(k, state)
        if k%100 == 0:
            print(f"Iteration {k}")
            print(info(state))

    
    


