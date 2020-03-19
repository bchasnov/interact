import numpy as np
from constants import *
import benpy as bp
from parameters import InteractParameter
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

app = QtGui.QApplication([])
from pyqtgraph.parametertree import Parameter, ParameterTree

import numpy.linalg as la

config = [('seed', 1), 
        ('nash', False), ('nonnash', False), 
        ('stable', False), ('unstable', False), 
        ('diagonalize', True),
        ('dim1', 2), ('dim2', 2),
        ('Amin', -1.), ('Amax', 1.), 
        ('Dmin', -1.), ('Dmax', 1.), 
        ('Pnorm', 1.), ('Knorm', 1.),
        ('max_samples', 1e2)]

def init(params):
    max_samples = int(params['max_samples'])
    for seed in range(max_samples):
        M = sample(params, seed)
        print(params)
        if not params['nash'] \
            and not params['nonnash'] \
            and not params['stable'] \
            and not params['unstable']:
            break

        Aeigs = la.eigvals(M[0][0])
        Deigs = la.eigvals(M[1][1])

        if params['nash'] and \
            np.all(np.real(Aeigs) < 0) and \
            np.all(np.real(Deigs) < 0): break
        if params['nonnash'] and \
            (np.any(np.real(Aeigs) >= 0) or \
            np.any(np.real(Deigs) >= 0)): break

        J = bp.block(M)
        eigs = la.eigvals(J)
        if params['stable'] and np.all(np.real(eigs)<0):
            break
        if params['unstable'] and np.any(np.real(eigs)>=0):
            break

    else:
        print("couldn't find a valid matrix in {} samples".format(max_samples))

    return M

def sample(p, seed):
    np.random.seed((1+seed)*int(p['seed']))
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

def plot_eigs(eigs):
    p1_eigs.setData(np.real(eigs), np.imag(eigs))

def plot_numrange(eigs):
    p1_qnr.setData(np.real(eigs), np.imag(eigs))

def preview(params):
    M = init(params)
    J = bp.block(M)
    eigs = la.eigvals(J)
    numrange = bp.numrange(M)

    plot_eigs(eigs)
    plot_numrange(numrange)
    return M

def run(params):
    M = preview(params)

w = pg.GraphicsLayoutWidget(show=True, border=1)
p1 = w.addPlot(row=0,col=0,lockAspect=1)
p1_eigs = pg.PlotDataItem(pen=None, symbol='+', lockAspect=1)
p1_qnr = pg.PlotDataItem(pen=None, symbol='o', lockAspect=1)
p1.addItem(p1_eigs)
p1.addItem(p1_qnr)

p2 = w.addPlot(row=1,col=0)
s2 = pg.ScatterPlotItem()
p2.addItem(p1_qnr)

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
