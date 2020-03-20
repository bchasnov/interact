import numpy as np
from constants import *
import benpy as bp
from parameters import InteractParameter
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import jax.numpy as jnp
import jax

app = QtGui.QApplication([])
from pyqtgraph.parametertree import Parameter, ParameterTree

import numpy.linalg as la

zero = float(0)
config = [('a1', -1.),
          ('a2', -.8),
          ('theta_a', zero),
          ('d1', -.6),
          ('d2', -.4),
          ('theta_d', zero),
          ('p1', zero),
          ('p2', zero),
          ('theta_p', zero),
          ('k1', zero),
          ('k2', zero),
          ('theta_k', zero)]

def init(a1,a2,theta_a, d1,d2,theta_d,
         p1,p2,theta_p, k1,k2,theta_k,**kwargs):
    np = jnp
    rot = lambda t: np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    diag = lambda x,y: np.array([[x,0],[0,y]])
    
    Ra = rot(theta_a)
    Rd = rot(theta_d)
    Rp = rot(theta_p)
    Rk = rot(theta_k)
    
    Sa = diag(a1,a2)
    Sd = diag(d1,d2)
    Sk = diag(k1,k2)
    Sp = diag(p1,p2)
    
    A = Ra@Sa@Ra.T
    D = Rd@Sd@Rd.T
    P = Rp@Sp@Rp.T
    K = Rk@Sk@Rk.T
    
    return ((A,P-K), (P.T+K.T, D))

def run(params):
    M = init(**params)
    J = bp.block(M, np=jnp)
    eigs = np.linalg.eigvals(J)
    qnum = bp.numrange(M, num=1e2)
    return eigs, qnum 

def preview(params):
    eigs, qnum = run(params)
    c = np.max(np.abs(np.real(qnum)))
    p_eigs.setRange(xRange=(-c,c), yRange=(-c,c))

    plt_eigs.setData(np.real(eigs), np.imag(eigs))
    plt_qnum.setData(np.real(qnum), np.imag(qnum))
    plt_diags1.setData([params['a1'], params['a2']],[0,0])
    plt_diags2.setData([params['d1'], params['d2']],[0,0])

    M = init(**params)
    (A,B),(C,D) = M
    J = bp.block(M)
    l_matrix.setText("M=<br>" + "<br>".join([
        ' '.join(['{: 2.2f}'.format(m)
                        for m in row]) 
        for row in J]))

    return M

def update(params):
    M = preview(params)

    xlim = [0, 2*np.pi]
    ylim = [0, 2*np.pi]
    grid, shape = bp.grid(xlim=xlim, ylim=ylim, xnum=64, ynum=64)

    def scan(s):
        myparams = dict()
        myparams.update(params)
        [myparams.pop(_) for _ in s]
        def sample(thetas, np=jnp):
            t1,t2 = thetas
            p = {_:t for _,t in zip(s,thetas)}
            M = init(**p, **myparams)
            J = bp.block(M, np=jnp)
            return np.max(np.real(np.linalg.eigvals(J)))
        return sample

    s = [['theta_a','theta_d'],
        ['theta_p','theta_k'],
        ['theta_a','theta_k'],
        ['theta_d','theta_k'],
        ['theta_a','theta_p'],
        ['theta_d','theta_p']]

    data_min = np.inf
    data_max = -np.inf
    for i,s in enumerate(s):
        sample = scan(s)
        data = np.array(jax.vmap(sample)(grid).reshape(shape))
        data_min = np.minimum(np.min(data), data_min)
        data_max = np.maximum(np.max(data), data_max)
        plt_imv[i].setImage(data)
        plt_imv[i].setRect(QtCore.QRectF(xlim[0], ylim[0], xlim[1], ylim[1]))#/np.pi, ylim[1]/np.pi))
        p_rot[i].setLabels(bottom=s[0], left=s[1])
        plt_pair[i].setData([params[s[0]]], [params[s[1]]])

    for p in plt_imv:
        p.setLevels([data_min, data_max])

    return M
    
pg.setConfigOptions(background=(255,255,255))
w = pg.GraphicsLayoutWidget()
w.show()

color1 = "#F5E800"
color2 = "#1ACDEA"
color3 = "#E1851A"
color4 = "#8C0053"

l_matrix = w.addLabel('matrix')
p_eigs = w.addPlot()
plt_qnum = pg.PlotDataItem(symbolSize=2, symbol='o', pen=None, symbolPen=None, symbolBrush=color1)
plt_eigs = pg.PlotDataItem(symbolSize=10, symbol='x', pen=None, symbolPen=color2, symbolBrush=color2)
plt_diags1 = pg.PlotDataItem(symbolSize=6, pen=None, symbol='o', symbolPen=color3, symbolBrush=color3)
plt_diags2 = pg.PlotDataItem(symbolSize=6, pen=None, symbol='o', symbolPen=color4, symbolBrush=color4)
p_eigs.addItem(plt_qnum)
p_eigs.addItem(plt_eigs)
p_eigs.addItem(plt_diags1)
p_eigs.addItem(plt_diags2)
p_eigs.setLabels(bottom='Real', left='Imaginary')

p_rot = []
plt_imv = []
plt_pair = []

for _ in range(6):
    p_rot.append(w.addPlot(row=_//2+1, col=_%2))
    plt_imv.append(pg.ImageItem(scale=(0.1,0.1)))
    plt_pair.append(pg.PlotDataItem(symbolSize=5, symbol='s'))
    plt_imv[-1].show() 
    p_rot[-1].addItem(plt_imv[-1])
    p_rot[-1].addItem(plt_pair[-1])

params = [InteractParameter(params=config, changing=preview, changed=update, name='M')]
p = Parameter.create(name='params', type='group', children=params)
t = ParameterTree()
t.setParameters(p, showTop=False)

win = QtGui.QWidget()
layout = QtGui.QGridLayout()
win.setLayout(layout)
layout.addWidget(t, 1, 0, 1, 1)
layout.addWidget(w, 1, 1, 1, 1);
win.show()
win.resize(1200,800)

if __name__ == '__main__':
    QtGui.QApplication.instance().exec_()
