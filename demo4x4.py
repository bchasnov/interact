import numpy as np
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

zero = float(0)
config = [('g', -.4),
          ('f', .3),
          ('p', .2),
          ('k', zero),
          ('da', zero),
          ('dd', zero),
          ('dp', .1),
          ('dk', zero),
          ('theta1', zero),
          ('theta2', zero),
          ('theta_a', zero),
          ('theta_d', zero),
          ('dtheta_a', zero),
          ('dtheta_d', zero)]

config_plot = [('res', 32),
        ('eps', 0.0001)]

config += config_plot

ones = [[-1,1],[-1,1]]
pis = [[-np.pi, np.pi],[-np.pi, np.pi]]
params_plot = [('g','k', ones),
               ('f','p',ones), 
               ('da','dd', ones),
               ('dp','dk', ones),
               ('da','p', ones),
               ('da','k', ones),
               ('dk','k', ones),
               ('dp','p', ones),
               ('g', 'theta_a', (ones[0], pis[1])),
               ('f', 'dtheta_a', (ones[0], pis[1])),
               ('theta_a','dtheta_a', pis),
               ('theta_d','dtheta_a', pis),
               ('dtheta_a','dtheta_d', pis),
               ('theta_a','theta_d', pis),]
#               ('theta1','theta2', pis),
#               ('dtheta_a','dtheta_d', pis)]

def init(g,f,p,k, da,dd,dp,dk,
        theta1,theta2, theta_a,theta_d,
        dtheta_a,dtheta_d, 
        np=jnp, **kwds):
    """ Input parameters and output matrix """
    diag = lambda x,y: np.array([[x,0],[0,y]])
    rot = lambda t: np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    
    a = g-f
    d = g+f

    Sa = diag(a-da, a+da)
    Sd = diag(d-dd, d+dd)
    Sp = diag(p-dp, p+dp)
    Sk = diag(k-dk, k+dk)
    
    A = rot(theta1) @ Sa @ rot(theta1).T
    D = rot(theta2) @ Sd @ rot(theta2).T
    P = rot(theta_a + dtheta_a) @ Sp @ rot(theta_d + dtheta_a).T
    K = rot(theta_a - dtheta_a) @ Sk @ rot(theta_d - dtheta_d).T
    
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

    g, f = params['g'], params['f']

    a = g-f
    d = g+f
    a1 = a + params['da']
    a2 = a - params['da']
    d1 = d + params['dd']
    d2 = d - params['dd']

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

    def scan(s):
        #TODO: how to scan parameters of a dictionary 
        #efficiently without needing to recompile?
        myparams = dict()
        myparams.update(params)
        for _ in s:
            myparams.pop(_)
        def sample(thetas, np=jnp):
            t1,t2 = thetas
            p = {_:t for _,t in zip(s,thetas)}
            M = init(**p, **myparams)
            J = bp.block(M, np=jnp)
            (A,_),(_,D) = M
            Aeigs = np.linalg.eigvals(A)
            Deigs = np.linalg.eigvals(D)
            return np.max(np.real(np.linalg.eigvals(J))), \
                    np.all(np.real(Aeigs)<0)*np.all(np.real(Deigs)<0)*2.-1.
        return sample

    def sample(p, vs):
        fn = scan(p)
        return jax.vmap(fn)(vs)
    #end TODO

    data_min, data_max = np.inf, -np.inf
    for i,(p1,p2,(xlim, ylim)) in enumerate(plot):
        sample = scan((p1,p2))
        rect = QtCore.QRectF(xlim[0], ylim[0], xlim[1]-xlim[0], ylim[1]-ylim[0])

        grid, shape = bp.grid(xlim=xlim, ylim=ylim, xnum=params['res'], ynum=params['res'])
        data, nash = jax.vmap(sample)(grid)
        data = np.array(data.reshape(shape)).T
        nash = np.array(nash.reshape(shape)).T*1.
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

    filename = os.path.join(directory, filename+ '.png')
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

interact_parameters = InteractParameter(params=config, changing=preview, changed=update, name='M')
for i, (p1,p2,(xlim,ylim)) in enumerate(params_plot):

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

params = [interact_parameters, 
        {'name': 'save dir', 'type':'str', 'value':'figs_{}'.format(bp.datehour())},
        {'name': 'Save figures', 'type': 'action'}]

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
        savePlot(plt1, (p1,p2), lims,directory=directory, name='stable')
        savePlot(plt2, (p1,p2), lims,directory=directory, name='nash')

p.param('Save figures').sigActivated.connect(savePlots)

t = ParameterTree()
t.setParameters(p, showTop=False)

win = QtGui.QWidget()
layout = QtGui.QGridLayout()
win.setLayout(layout)
layout.addWidget(t, 1, 0, 1, 1)
layout.addWidget(w, 1, 1, 1, 1);
win.show()
win.resize(800,1200)

timer = QtCore.QTimer()
timer.timeout.connect(lambda: update(interact_parameters.values) and timer.stop())
timer.start(2000)


if __name__ == '__main__':
    QtGui.QApplication.instance().exec_()
