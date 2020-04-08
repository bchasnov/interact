#!/usr/bin/python
import os.path, time
import sys
import importlib.util
from importlib import reload

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

def upload(f):
    spec = importlib.util.spec_from_file_location("params",f)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

#import params as module
module = upload("params.py")
print(sys.modules)
state = None
modified = 0
def tick(doc):
    global state, modified, module
    current = os.path.getmtime(doc)
    if modified != current:
        modified = current
        #reload(module)
        module = upload(doc)
        state = module.init()
    else:
        state = module.update(state)

app = QtGui.QApplication([])
w = pg.GraphicsLayoutWidget(show=True, border=0.5)
w.show()

doc = sys.argv[1] if len(sys.argv) > 1 else '.'
timer = QtCore.QTimer()
timer.timeout.connect(lambda: tick(doc))
timer.start(200)

if __name__ == "__main__":
    QtGui.QApplication.instance().exec_()
    tick(doc)
    
