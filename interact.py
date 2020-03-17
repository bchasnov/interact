""" main python doc """

from interact import *
from constants import *
from benpy import bp

import pyqtgraph as pg
import numpy as np
import numpy.linalg as la
#import jax.numpy as jp

class ParamObj(object):
    # Just a helper for tracking parameters and responding to changes
    def __init__(self):
        self.__params = {}

    def __setitem__(self, item, val):
        self.setParam(item, val)

    def setParam(self, param, val):
        self.setParams(**{param:val})

    def setParams(self, **params):
        """Set parameters for this optic. This is a good function to override for subclasses."""
        self.__params.update(params)
        self.paramStateChanged()

    def paramStateChanged(self):
        pass

    def __getitem__(self, item):
        # bug in pyside 1.2.2 causes getitem to be called inside QGraphicsObject.parentItem:
        return self.getParam(item)  # PySide bug: https://bugreports.qt.io/browse/PYSIDE-671

    def __len__(self):
        # Workaround for PySide bug: https://bugreports.qt.io/browse/PYSIDE-671
        return 0

    def getParam(self, param):
        return self.__params[param];

    def __eq__(self, other):
        raise NotImplementedError


class ComplexPlane(pg.GraphicsObject, ParamObj):
    pass

class Array(pg.GraphicsObject, ParamObj):
    pass




app = pg.QtGui.QApplication([])

w = pg.GraphicsLayoutWidget(show=True, border=1)
w.resize(WINDOW_SIZE*4*16,WINDOW_SIZE*4*10)
win.setWindowTitle(WINDOW_TITLE)
w.show()

view_items = []

""" Plot Eigenvalues, Quadratic Numerical Range, and Gerschgorin circles """
eigenvalues = ComplexPlane()
numerical_range = ComplexPlane()
gerschgorin = Circles()

""" Plot (g,k),(f,p) axes """
abcd = Array()
gkfp = Array()
normM = Array()
normG = Array()

""" Collect all views """
view_gkfp = [abcd, gkfp]
view_complex = [eigenvalues, numerical_range, gerschgorin]
views = [view_complex, view_gkfp]

""" Calculate Function """
def calc():
    M = parameters.get('matrix')
    (A,B),(C,D) = M
    eigs = la.eigvals(bp.block(M))
    qnr = bp.nrange.quadratic(M)
    decomp = bp.transform.helmholtz(M)

    eigenvalues == eigs
    numerical_range == qnr
    gerschgorin == gersh
    abcd == M
    gkfp == decomp
    normM == bp.map(la.norm, M)

""" Add all views """
for view in views:
    v = w.addViewBox()
    v.setRange(pg.QtCore.QRectF(-100,-100,100,100))
    v.setAspectLocked()

    for item in view:
        v.addItem(item)

count = 0
def callback():
    global count
    print("{05d}".format(count))
    count += 1

block_matrix = BlockMatrixParameter(callback, name='M') 
params = [block_matrix]

p = Parameter.create(name='params', type='group', children=params)
t = ParameterTree()
t.setParameters(p, showTop=False)

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1):
        QtGui.QApplication.instance().exec_()
