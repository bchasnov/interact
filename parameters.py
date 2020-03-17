import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

app = QtGui.QApplication([])
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree

class BlockMatrixParameter(pTypes.GroupParameter):
    def __init__(self, callback, **opts):
        opts['type'] = 'bool'
        opts['value'] = True
        pTypes.GroupParameter.__init__(self, **opts)
        self.params = []
        self.callback = callback
        
        np.random.seed(0)
        m = 2
        M = [[-np.eye(m), np.random.randn(m,m)], \
             [np.random.randn(m,m), -np.eye(m)]]
        (A,B),(C,D) = M

        self.addChild({'name': 'seed', 'type': 'int', 'value': 0})
        self.seed = self.param('seed')
        self.seed.sigValueChanging.connect(self.changed)

        for e,E in zip(['a','d','b','c'], (A,D,B,C)):
            (e11,e12),(e21,e22) = E
            self.addChild({'name': e+'11', 'type':'float', 'value': e11})
            self.addChild({'name': e+'12', 'type':'float', 'value': e12})
            self.addChild({'name': e+'21', 'type':'float', 'value': e21})
            self.addChild({'name': e+'22', 'type':'float', 'value': e22})
            self.params.append(self.param(e+'11'))
            self.params.append(self.param(e+'12'))
            self.params.append(self.param(e+'21'))
            self.params.append(self.param(e+'22'))
        
        for p in self.params:
            p.sigValueChanging.connect(self.changed)

    def changed(self):
        self.callback([p.value for p in params])

def calc():
    pass

params = [BlockMatrixParameter(calc, name='M')]
p = Parameter.create(name='params', type='group', children=params)

t = ParameterTree()
t.setParameters(p, showTop=False)

win = QtGui.QWidget()
layout = QtGui.QGridLayout()
win.setLayout(layout)
layout.addWidget(t, 1, 0, 1, 1)
win.show()
win.resize(800,800)

if __name__ == '__main__':
    QtGui.QApplication.instance().exec_()
