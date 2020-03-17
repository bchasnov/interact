

""" main python doc """

from interact import *
from constants import *
from benpy import bp

import pyqtgraph as pg
import numpy as np
import numpy.linalg as la
#import jax.numpy as jp

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

w.addPlot(







if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1):
        QtGui.QApplication.instance().exec_()
