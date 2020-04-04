from collections import OrderedDict

class TwoPlayer(SystemSolver):
    """
    Provides an interface for scaling and rotating a 
    block 2 x 2 matrix.

    """
    defaultState = OrderedDict([
        ('m1', [None, float, None, 'nf']),
        ('m2', [None, float, None, 'nf']),
        ('h1', [None, float, None, 'nf']),
        ('h2', [None, float, None, 'nf']),
        ('z1', [None, float, None, 'nf']),
        ('z2', [None, float, None, 'nf']),
        ('p1', [None, float, None, 'nf']),
        ('p2', [None, float, None, 'nf']),
        ('theta1', [None, float, None, 'nf']),
        ('theta2', [None, float, None, 'nf']),
        ('thetaz', [None, float, None, 'nf']),
        ('thetap', [None, float, None, 'nf']),
        ('a11', [None, float, None, 'nf']),
        ('a12', [None, float, None, 'nf']),
        ('a21', [None, float, None, 'nf']),
        ('a22', [None, float, None, 'nf']),
        ('b11', [None, float, None, 'nf']),
        ('b12', [None, float, None, 'nf']),
        ('b21', [None, float, None, 'nf']),
        ('b22', [None, float, None, 'nf']),
        ('c11', [None, float, None, 'nf']),
        ('c12', [None, float, None, 'nf']),
        ('c21', [None, float, None, 'nf']),
        ('c22', [None, float, None, 'nf']),
        ('d11', [None, float, None, 'nf']),
        ('d12', [None, float, None, 'nf']),
        ('d21', [None, float, None, 'nf']),
        ('d22', [None, float, None, 'nf']),
        ('eig1_real', [None, float, None, 'nf']),
        ('eig1_imag', [None, float, None, 'nf']),
        ('eig2_real', [None, float, None, 'nf']),
        ('eig2_imag', [None, float, None, 'nf']),
        ('eig3_real', [None, float, None, 'nf']),
        ('eig3_imag', [None, float, None, 'nf']),
        ('eig4_real', [None, float, None, 'nf']),
        ('eig4_imag', [None, float, None, 'nf']),
        ])


    def M(self):
        return self.matrix
    def _a11(self): return self.M()[0][0]
    def _a12(self): return self.M()[0][1]
    def _a21(self): return self.M()[1][0]
    def _a22(self): return self.M()[1][1]
    def _b11(self): return self.M()[0][2]
    def _b12(self): return self.M()[0][3]
    def _b21(self): return self.M()[1][2]
    def _b22(self): return self.M()[1][3]
    def _c11(self): return self.M()[2][0]
    def _c12(self): return self.M()[2][1]
    def _c21(self): return self.M()[3][0]
    def _c22(self): return self.M()[3][1]
    def _d11(self): return self.M()[2][2]
    def _d12(self): return self.M()[2][3]
    def _d21(self): return self.M()[3][2]
    def _d22(self): return self.M()[3][3]

