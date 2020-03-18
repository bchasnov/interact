import numpy as np
from constants import *
import benpy as bp
from parameters import InteractParameter

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

app = QtGui.QApplication([])
from pyqtgraph.parametertree import Parameter, ParameterTree

import numpy.linalg as la

config = [('seed', 0), 
        ('nash', False), ('stable', False), 
        ('diagonalize', True),
        ('dim1', 2), ('dim2', 2),
        ('Amin', -1.), ('Amax', 1.), 
        ('Dmin', -1.), ('Dmax', 1.), 
        ('Pnorm', 1.), ('Knorm', 1.)]

def init(p):
    np.random.seed(int(p['seed']))
    m,n = int(p['dim1']), int(p['dim2'])
    if p['diagonalize']:
        A = np.diag(np.random.uniform(p['Amin'], p['Amax'], m))
        D = np.diag(np.random.uniform(p['Dmin'], p['Dmax'], n))
    else: 
        raise NotImplementedError

    P = np.random.rand(m,n)
    P *= p['Pnorm']/la.norm(P)

    K = np.random.rand(m,n)
    K *= p['Knorm']/la.norm(K)
    
    return (A, P-K),(P.T+K.T, D)

def calc(M):
    J = bp.block(M)
    eigs = la.eigvals(J)
    return eigs

def plot(eigs):
    s1.clear()
    spots = [{'pos': [np.real(z), np.imag(z)], 'data': i} for i, z in enumerate(eigs)]
    s1.addPoints(spots)

def preview(params):
    M = init(params)
    out = calc(M)
    plot(out)

def run(params):
    preview(params)

w = pg.GraphicsLayoutWidget(show=True, border=1)
p1 = w.addPlot(row=0,col=0)
s1 = pg.ScatterPlotItem()
p1.addItem(s1)

p2 = w.addPlot(row=1,col=0)
s2 = pg.ScatterPlotItem()
p2.addItem(s2)

params = [InteractParameter(params=config, changing=preview, changed=run, name='M')]
p = Parameter.create(name='params', type='group', children=params)
t = ParameterTree()
t.setParameters(p, showTop=False)

win = QtGui.QWidget()
layout = QtGui.QGridLayout()
win.setLayout(layout)
layout.addWidget(t, 1, 0, 1, 1)
layout.addWidget(w, 1, 1, 1, 1);
win.show()
win.resize(800,800)

if __name__ == '__main__':
    QtGui.QApplication.instance().exec_()
