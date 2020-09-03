import sys
sys.path.append('..')
import numpy as np
import subprocess
import pickle
from constants import *
import benpy as bp
from parameters import InteractParameter
import pyqtgraph as pg
import pyqtgraph.exporters
from pyqtgraph.Qt import QtCore, QtGui
import jax.numpy as jnp
import jax
import os

app = QtGui.QApplication([])
from pyqtgraph.parametertree import Parameter, ParameterTree

import numpy.linalg as la

class ComputeJax():
    def __init__(self, param_dict, param_pair):
        """ param_dict: dictionary of parameters 
            param_pair: list of two parammeters to scan
            samples: number of points (width and height)
            lim: range of params [[xmin,xmax],[ymin,ymax]]"""
        self.param_dict = dict()
        self.param_dict.update(param_dict)
        self.param_pair = param_pair

        def sample(pairs, params, np=jnp):
            p = {k:v for k,v in zip(param_pair, pairs)}
            M = init(**p, **params)
            J = bp.block(M, np=jnp)
    
            (A,_),(_,D) = M
            Aeigs = np.linalg.eigvals(A)
            Deigs = np.linalg.eigvals(D)
            stable = np.max(np.real(np.linalg.eigvals(J)))
            opt1 = np.all(np.real(Aeigs)<0)
            opt2 = np.all(np.real(Deigs)<0)
            nash = opt1*opt2*2.-1
            return stable, nash

        self.sample = jax.vmap(sample, (0,None))

    def compute(self, param_dict, lim, num):
        xlim, ylim = lim
        xnum, ynum = num

        grid, shape = bp.grid(xlim=xlim, ylim=ylim, 
                              xnum=xnum, ynum=ynum)
        params = dict()
        params.update(param_dict)
        for k in self.param_pair:
            params.pop(k)
        stable, nash = self.sample(grid, params)
        stable = np.array(stable.reshape(shape)).T
        nash = np.array(nash.reshape(shape)).T

        return stable, nash




zero = float(0)
config = [('m', -5),
          ('h', 3),
          ('p', zero),
          ('z', 2),
          ('h1', 3),
          ('h2', 4),
          ('zp', zero),
          ('zz', 5),
          ('t1', zero),
          ('t2', zero),
          ('tp', zero),
          ('tz', zero)]

config_plot = [('res', 32),
        ('eps', 0.0001)]

config += config_plot

lim = [[-10,10],[-10,10]]
pis = [[-np.pi, np.pi],[-np.pi, np.pi]]
params_plot = [('m','z', lim),
               ('h','p',lim), 
               ('h1','h2', np.array(lim)*1.2),
               ('zp','zz', np.array(lim)*1),
               ('t1','t2', pis),
               ('tp','tz', pis) ]

def init(m,h,p,z,h1,h2,zp,zz,t1,t2,tp,tz,
        np=jnp, **kwds):
    """ Input parameters and output matrix """
    diag = lambda x,y: np.array([[x,0],[0,y]])
    rot = lambda t: np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    z1=zp
    z2=zz
    I = np.eye(2)
    L = np.eye(2)*np.array([1,-1])
    P = rot(t1-tp).T @ ((p)*I + z1*L) @ rot(t2-tp).T
    Z = rot(t1-tz).T @ ((z)*I + z2*L) @ rot(t2-tz)
    A = (m+h)*I + h1*L
    B = (P-Z)
    C = (P.T+Z.T)
    D = (m-h)*I + h2*L
    
    return ((A,B), (C, D))

def run(params):
    M = init(**params)
    J = bp.block(M, np=jnp)
    eigs = np.linalg.eigvals(J)
    qnum = bp.numrange(M, num=1e2)
    return eigs, qnum 

def preview(params):
    """ Fast calculations to preview current parameters """
    eigs, qnum = run(params)

    m, h = params['m'], params['h']

    a = m+h
    d = m-h
    a1 = a + params['h1']
    a2 = a - params['h1']
    d1 = d + params['h2']
    d2 = d - params['h2']

    c = np.max(np.abs(np.real(qnum)))
    c = max(np.max(np.abs(np.imag(qnum))),c)
    p_eigs.setRange(xRange=(-c,c), yRange=(-c,c))
    plt_eigs.setData(np.real(eigs), np.imag(eigs))
    plt_qnum.setData(np.real(qnum), np.imag(qnum))
    plt_diags1.setData([a1, a2],[0,0])
    plt_diags2.setData([d1, d2],[0,0])

    M = init(**params)
    (A,B),(C,D) = M
    J = bp.block(M)
    text = "M=<br>" + "<br>".join([
        ' '.join(['{: 2.2f}'.format(m)
                        for m in row]) 
        for row in J])

    text += '<br>eigs=<br>'
    def imag2st(imag, fm):
        g,k = np.real(imag), np.imag(imag)
        if np.isclose(k,0):
            return fm.format(g)
        st = fm.format(g)
        if k > 0: st += "+"
        return st + fm.format(k) + 'i'

    text += '<br>'.join([imag2st(e,'{:.2f}') for e in np.sort(eigs)])
    l_matrix.setText(text)

    return M


def update(params, plot=params_plot):
    """ Fills out the plots using current parameters """
    M = preview(params)

    #TODO: move to something that does graphics..
    eps = params['eps']
    pos = [0,  .5-eps, .5, .5+eps, 1.]
    cmap = pg.ColorMap(color=bp.colors.GREEN_WHITE, pos=pos, mode=pg.ColorMap.HSV_POS)
    lut = cmap.getLookupTable(start=0., stop=1., nPts=512, alpha=1.)
    for plt in plt_imv:
        plt.setLookupTable(lut)
    cmap2 = pg.ColorMap(color=bp.colors.BLUE_WHITE,
            pos=pos, mode=pg.ColorMap.RGB)
            #[bp.colors.BLUE_WHITE[0], (255,255,255)], pos=[0,1], mode=pg.ColorMap.HSV_POS)
    lut2 = cmap2.getLookupTable(start=1., stop=0., nPts=512, alpha=1.)
    for plt in plt_imv2:
        plt.setLookupTable(lut2)
    #end TODO

    data_min, data_max = np.inf, -np.inf
    for i,(p1,p2,(xlim, ylim)) in enumerate(plot):
        rect = QtCore.QRectF(xlim[0], ylim[0], xlim[1]-xlim[0], ylim[1]-ylim[0])
        num = params['res'], params['res']
        lim = (xlim, ylim)
        data, nash = computes[i].compute(params, lim=lim, num=num)
        data_min = np.minimum(np.min(data), data_min)
        data_max = np.maximum(np.max(data), data_max)
        plt_imv[i].setImage(data)
        plt_imv[i].setRect(rect)
        plt_imv2[i].setImage(nash)
        plt_imv2[i].setRect(rect)
        p_rot[i].setLabels(bottom=p1, left=p2)
        plt_pair[i].setData([params[p1]], [params[p2]])

    level = np.max(np.abs([data_min, data_max]))
    for p in plt_imv:
        p.setLevels([-level, level])
    for p in plt_imv2:
        p.setLevels([-1.,1.])

    return M
    
def savePlot(plt, axes, lims, directory='', name=''):
    assert len(axes) == len(lims)
    assert len(axes) == 2

    filename = "{}_{}_{}".format(axes[0], axes[1], name)

    filename = os.path.join(directory, '_' + filename+ '.png')
    print('Saving', filename)
    #img = QtGui.QImage()
    #pix = QtGui.QPixmap(img)
    #painter = QtGui.QPainter(pix)
    #painter.drawImage(0,0,plt_stable.qimage.mirrored(vertical=True))
    #painter.setCompositionMode(QtGui.QPainter.CompositionMode_Multiply)
    #painter.drawImage(0,0,plt_nash.qimage.mirrored(vertical=True))
    #status = img.save(filename)
    status = plt.qimage.mirrored(vertical=True).save(filename)
    print('Status:', status)
    return filename
    #exporter.export(os.path.join(directory, filename, '.png'))

pg.setConfigOptions(background=(255,255,255))
w = pg.GraphicsLayoutWidget()
w.show()

color1 = "#F5E800"
color2 = "#1ACDEA"
color3 = "#E1851A"
color4 = "#8C0053"

#TODO: dynamically allocate plots
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
plt_imv2 = []
plt_pair = []
computes = []

interact_parameters = InteractParameter(params=config, changing=preview, changed=update, name='M')
num = (32,32)
for i, (p1,p2,(xlim,ylim)) in enumerate(params_plot):
    computes.append(ComputeJax(param_dict=config,
        param_pair=(p1,p2)))
    p_rot.append(w.addPlot(row=i//2+1, col=i%2))
    plt_imv.append(pg.ImageItem())
    plt_imv2.append(pg.ImageItem())
    plt_pair.append(pg.PlotDataItem(symbolSize=5, symbol='+', symbolBrush=(255,255,255)))

    def click(e, imv=plt_imv[-1], xlim=xlim, ylim=ylim, pp=(p1,p2)):
        h, w = imv.height(), imv.width()
        h = p.param('M', 'res').value()
        w = h
        x,y = e.pos()
        x = (x/w)*(xlim[1]-xlim[0])+xlim[0]
        y = (y/h)*(ylim[1]-ylim[0])+ylim[0]
        p.param('M', pp[0]).setValue(np.round(x,1))
        p.param('M', pp[1]).setValue(np.round(y,1))

    plt_imv[-1].mouseClickEvent = click
    plt_imv[-1].show() 
    plt_imv2[-1].show() 
    plt_imv[-1].setCompositionMode(QtGui.QPainter.CompositionMode_Multiply)
    plt_imv2[-1].setCompositionMode(QtGui.QPainter.CompositionMode_Multiply)

    p_rot[-1].addItem(plt_imv[-1])
    p_rot[-1].addItem(plt_imv2[-1])
    p_rot[-1].addItem(plt_pair[-1])
    p_rot[-1].setAspectLocked()

params = [interact_parameters, 
        {'name': 'save dir', 'type':'str', 'value':'figs_{}'.format(bp.datehour())},
        {'name': 'Save figures', 'type': 'action'},
    {'name': 'Save/Restore', 'type': 'group', 'children': [
        {'name': 'Save State', 'type': 'action'},
        {'name': 'Restore State', 'type': 'action', 'children': [
            {'name': 'Add missing items', 'type': 'bool', 'value': True},
            {'name': 'Remove extra items', 'type': 'bool', 'value': True},
        ]},
    ]}]

p = Parameter.create(name='params', type='group', children=params)

def savePlots():
    directory = p.param('save dir').value()
    try:
        os.mkdir(directory)
    except FileExistsError:
        pass
    except:
        print('error!')

    for plt1,plt2,(p1,p2,lims) in zip(plt_imv,plt_imv2,params_plot):
        f1 = savePlot(plt1, (p1,p2), lims,
                directory=directory, name='stable')
        f2 = savePlot(plt2, (p1,p2), lims,
                directory=directory, name='nash')

        cmd = str(f1) + ' -compose Multiply '
        cmd += str(f2) + ' '
        cmd += os.path.join(directory,'{}_{}.png'.format(p1,p2))
        #subprocess.call(['composite', cmd])
        #print(cmd)
        os.system('pwd')
        os.system('composite '+cmd)

    #composite $1_stable.png -compose Multiply $1_nash.png $1.png

    state = p.saveState()
    state = interact_parameters.values;
    f = open(os.path.join(directory, 'params.txt'), "w")
    f.write(str(state))
    f.close()
    #param('M')
    print("save state is")
    print(state)
p.param('Save figures').sigActivated.connect(savePlots)

def save():
    state = p.saveState()
    directory = p.param('save dir').value()
    try: os.mkdir(directory)
    except FileExistsError: print('file exists')
    except: print('error!')

    with open(os.path.join(directory, 'state.pkl'), 'wb') as f:
        pickle.dump(state, f )
    
def restore():
    directory = p.param('save dir').value()
    with open(os.path.join(directory, 'state.pkl'), 'rb') as f:
        state = pickle.load(f)
    add = p['Save/Restore', 'Restore State', 'Add missing items']
    rem = p['Save/Restore', 'Restore State', 'Remove extra items']
    p.restoreState(state, addChildren=add, removeChildren=rem)
p.param('Save/Restore', 'Save State').sigActivated.connect(save)
p.param('Save/Restore', 'Restore State').sigActivated.connect(restore)

t = ParameterTree()
t.setParameters(p, showTop=False)

win = QtGui.QWidget()
layout = QtGui.QGridLayout()
win.setLayout(layout)
layout.addWidget(t, 1, 0, 1, 1)
layout.addWidget(w, 1, 1, 1, 1);
win.show()
win.resize(1000,700)

timer = QtCore.QTimer()
timer.timeout.connect(lambda: update(interact_parameters.values) and timer.stop())
timer.start(100)


if __name__ == '__main__':
    QtGui.QApplication.instance().exec_()
