#!/usr/bin/python
import os.path, time
import sys
import importlib.util
from importlib import reload

def upload(f):
    spec = importlib.util.spec_from_file_location("module.name",f)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

import params as module

if __name__ == "__main__":
    prev = 0
    f = sys.argv[1] if len(sys.argv) > 1 else '.'
#    module = upload(f)
    print('Built')
    
    state = module.init()
    while True:
        t = os.path.getmtime(f)
        if prev != t:
            prev = t
            reload(module)
            state = module.init()
        time.sleep(0.2)
        state = module.update(state)
