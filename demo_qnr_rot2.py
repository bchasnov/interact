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
config = [('ga', -.8),
          ('fa', .1),
          ('gd', -.2),
          ('fd', .1),
          ('gp', zero),
          ('fp', .1),
          ('gk', zero),
          ('fk', .1),
          ('theta_a', zero),
          ('theta_d', zero),
          ('theta_r_p', zero),
          ('theta_s_p', zero),
          ('theta_r_k', zero),
          ('theta_s_k', zero)]

config_plot = [('eps', 0.01),
              ('half', 0.3)]
config += config_plot

def init(ga,fa,theta_a, 
         gd,fd,theta_d,
         gp,fp,theta_r_p, theta_s_p,
         gk,fk,theta_r_k, theta_s_k,
         **kwargs):
    np = jnp
    diag = lambda x,y: np.array([[x,0],[0,y]])
    rot = lambda t: np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    
    Sa = diag(ga-fa, ga+fa)
    Sd = diag(gd-fd, gd+fd)
    Sk = diag(gk-fk, gk+fk)
    Sp = diag(gp-fp, gp+fp)
    
    A = rot(theta_a) @ Sa @ rot(theta_a).T
    D = rot(theta_d) @ Sd @ rot(theta_a).T
    P = rot(theta_s_p) @ Sp @ rot(theta_r_p).T
    K = rot(theta_s_k) @ Sk @ rot(theta_r_k).T
    
    return ((A,P-K), (P.T+K.T, D))

def run(params):
    M = init(**params)
    J = bp.block(M, np=jnp)
    eigs = np.linalg.eigvals(J)
    qnum = bp.numrange(M, num=1e2)
    return eigs, qnum 

def preview(params):
    """ Fast calculations to preview current parameters """
    eigs, qnum = run(params)
    c = np.max(np.abs(np.real(qnum)))
    p_eigs.setRange(xRange=(-c,c), yRange=(-c,c))

    plt_eigs.setData(np.real(eigs), np.imag(eigs))
    plt_qnum.setData(np.real(qnum), np.imag(qnum))
    plt_diags1.setData([params['ga']-params['fa'], params['ga']+params['fa']],[0,0])
    plt_diags2.setData([params['gd']-params['fd'], params['gd']+params['fd']],[0,0])

    M = init(**params)
    (A,B),(C,D) = M
    J = bp.block(M)
    l_matrix.setText("M=<br>" + "<br>".join([
        ' '.join(['{: 2.2f}'.format(m)
                        for m in row]) 
        for row in J]))

    return M

params_plot = {k:v for k,v in config_plot}
def update(params):
    """ Fills out the plots using current parameters """
    M = preview(params)

    xlim = [0, 2*np.pi]
    ylim = [0, 2*np.pi]
    grid, shape = bp.grid(xlim=xlim, ylim=ylim, xnum=64, ynum=64)

    def scan(s):
        myparams = dict()
        myparams.update(params)
        for _ in s:
            print(_)
            myparams.pop(_)
        def sample(thetas, np=jnp):
            t1,t2 = thetas
            p = {_:t for _,t in zip(s,thetas)}
            M = init(**p, **myparams)
            J = bp.block(M, np=jnp)
            return np.max(np.real(np.linalg.eigvals(J)))
        return sample

    ss = [['theta_a','theta_d'],
        ['theta_r_p','theta_r_k'],
        ['theta_a','theta_r_k'],
        ['theta_d','theta_r_k'],
        ['theta_a','theta_r_k'],
        ['theta_d','theta_r_k'],
        ['theta_s_k','theta_r_k'],
        ['theta_r_p','theta_r_k'],
        ['theta_s_k','theta_s_p'],
        ['theta_r_p','theta_s_p']]

    colors = [(165,255,70), #(40,238,40),
          (0,157,0),
          (92,96,88),
          (15,15,15),
          (92,96,88),
          (0,72,127),
          (0,26,80)]

    eps = params['eps']
    half = params['half'] 

    pos = [0,  half, .5-eps, .5, .5+eps, 1.-half, 1.]
    cmap = pg.ColorMap(color=colors, pos=pos, mode=pg.ColorMap.HSV_POS)
    lut = cmap.getLookupTable(start=0., stop=1., nPts=512, alpha=1.)

    data_min = np.inf
    data_max = -np.inf
    for i,s in enumerate(ss):
        sample = scan(s)
        data = np.array(jax.vmap(sample)(grid).reshape(shape))
        data_min = np.minimum(np.min(data), data_min)
        data_max = np.maximum(np.max(data), data_max)
        plt_imv[i].setImage(data)
        plt_imv[i].setRect(QtCore.QRectF(xlim[0], ylim[0], xlim[1], ylim[1]))#/np.pi, ylim[1]/np.pi))
        plt_imv[i].setLookupTable(lut)
        p_rot[i].setLabels(bottom=s[0], left=s[1])
        plt_pair[i].setData([params[s[0]]], [params[s[1]]])

    level = np.max(np.abs([data_min, data_max]))
    for p in plt_imv:
        #p.setLevels([data_min, data_max])
        p.setLevels([-level, level])

    return M
    
def savePlots(plts, directory, width=100):
    for plt in plts:
        exporter = pg.exporters.ImageExporter(plt.plotItem)
        exporter.parameters()['width'] = width
        #filename = plt.getTitle()
        filename = 'test'
        exporter.export(os.path.join(directory, filename))

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

for _ in range(10):
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
