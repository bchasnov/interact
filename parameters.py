import numpy as np
from constants import *
import benpy as bp

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
        self.values = {}
        self.callback = callback
        self.initialize(0)

    def initialize(self, seed):
        np.random.seed(seed)
        m = 2
        M = [[-np.eye(m), np.random.randn(m,m)], \
             [np.random.randn(m,m), -np.eye(m)]]
        (A,B),(C,D) = M

        self.addChild({'name': 'seed', 'type': 'int', 'value': 0})
        self.seed = self.param('seed')
        self.seed.sigValueChanging.connect(self.changed)

        # TODO: make it support more than 2x2
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
            self.values[e+'11'] = e11;
            self.values[e+'12'] = e12;
            self.values[e+'21'] = e21;
            self.values[e+'22'] = e22;
        
        for p in self.params:
            #p.sigValueChanged.connect(self.changed)
            p.sigValueChanging.connect(self.changing)

    def changed(self):
        self.callback([p.value() for p in self.params])

    def changing(self, param, value):
        #TODO: change the seed
        #if param.name() is 'seed': initialize(value)
        self.values[param.name()] = value;
        
        v = self.values
        A = [[v['a11'], v['a12']], 
             [v['a21'], v['a22']]]
        B = [[v['b11'], v['b12']], 
             [v['b21'], v['b22']]]
        C = [[v['c11'], v['c12']], 
             [v['c21'], v['c22']]]
        D = [[v['d11'], v['d12']], 
             [v['d21'], v['d22']]]
        M = ((A,B),(C,D))

        return self.callback(M)

w = pg.GraphicsLayoutWidget(show=True, border=1)
_p1 = w.addPlot(row=0,col=0)
p1 = _p1.plot()
p2 = w.addPlot(row=1,col=0)
s1 = pg.ScatterPlotItem()
eigs = [1+1j, 1-1j, 2+2j, 2-2j]
spots = {}
p2.addItem(s1)

import numpy.linalg as la

def calc(M):
    J = bp.block(M)
    eigs = la.eigvals(J)
    s1.clear()
    spots = [{'pos': [np.real(z), np.imag(z)], 'data': i} for i, z in enumerate(eigs)]
    s1.addPoints(spots)
    #p1.setData([np.real(eigs), np.imag(eigs)])
    p1.setData(np.real(eigs))
   # [np.real(eigs), np.imag(eigs)])

params = [BlockMatrixParameter(calc, name='M')]
p = Parameter.create(name='params', type='group', children=params)
t = ParameterTree()
t.setParameters(p, showTop=False)

app = pg.QtGui.QApplication([])

win = QtGui.QWidget()
layout = QtGui.QGridLayout()
win.setLayout(layout)
layout.addWidget(t, 1, 0, 1, 1)
layout.addWidget(w, 1, 1, 1, 1);
win.show()
win.resize(800,800)


if __name__ == '__main__':
    QtGui.QApplication.instance().exec_()
