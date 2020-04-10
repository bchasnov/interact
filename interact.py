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
from refresh import instance
import importlib
import importlib.util

print(importlib.util.spec_from_file_location("modules/quartic.py"))

k, tick = instance(r"modules/quartic.py")

if __name__ == '__main__':
    num_iter = int(1e3)
    while True:
        k, info = tick(k)
        if k % 100 == 0:
            print(f"Iteration {k}")
            print(info)
    
