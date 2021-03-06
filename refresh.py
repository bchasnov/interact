import importlib
import importlib.util
import os.path

def load(f):
    """ Move to utils """
    def refresh(status):
        m = os.path.getmtime(f)

        # If file did not change
        if m == status: 
            return m, False

        spec = importlib.util.spec_from_file_location("module", f)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return m, module

    return 0, refresh

def instance(filename):
    status, refresh = load(filename)
    status, module = refresh(status)
    config = module.defaultConfig

    def tick(k):
        global status
        status, _module= refresh(status)
        if _module:
            module = _module
            state, update, info = module.init(**config)
            k = 0
        else:
            state = module.update(state)
            k += 1

        return k, module.info(state)

    return 0, tick

