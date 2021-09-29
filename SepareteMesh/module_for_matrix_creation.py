from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
import numpy as np
import multiprocessing as mp

class Matrix_Creation:
    def __init__(self, npoints, ne, X, Y, IEN, vx, vy, dt, nu):
        self.ne = ne
        self.X = X
        self.Y = Y
        self.IEN = IEN
        self.vx = vx
        self.vy = vy
        self.dt = dt
        self.nu = nu
        self.npoints = npoints

    def matrix_mouting(self, inicial, final):
        K = lil_matrix(     (self.npoints,self.npoints), dtype='float')
        M = lil_matrix(     (self.npoints,self.npoints), dtype='float')
        Gx = lil_matrix(    (self.npoints,self.npoints), dtype='float')
        Gy = lil_matrix(    (self.npoints,self.npoints), dtype='float')
        Kest = lil_matrix(  (self.npoints,self.npoints), dtype='float')
        Kx = lil_matrix(    (self.npoints,self.npoints), dtype='float')
        Ky = lil_matrix(    (self.npoints,self.npoints), dtype='float')
        Kxy = lil_matrix(   (self.npoints,self.npoints), dtype='float')

        for e in range(inicial, final):
            v = self.IEN[e]
            # area do elemento
            det = self.X[v[0]]*( self.Y[v[1]]-self.Y[v[2]]) \
                  + self.X[v[1]]*( self.Y[v[2]]-self.Y[v[0]]) \
                  + self.X[v[2]]*( self.Y[v[0]]-self.Y[v[1]])

            area = det/2.0        
            m = (area/12.0) * np.array([ [2.0, 1.0, 1.0],
                                      [1.0, 2.0, 1.0],
                                      [1.0, 1.0, 2.0] ])
            # formula do k do elementro triangular linear
            b1 = self.Y[v[1]]-self.Y[v[2]]
            b2 = self.Y[v[2]]-self.Y[v[0]]
            b3 = self.Y[v[0]]-self.Y[v[1]]
            c1 = self.X[v[2]]-self.X[v[1]]
            c2 = self.X[v[0]]-self.X[v[2]]
            c3 = self.X[v[1]]-self.X[v[0]]
            VX = (self.vx[v[0]]+self.vx[v[1]]+self.vx[v[2]])/3
            VY = (self.vy[v[0]]+self.vy[v[1]]+self.vy[v[2]])/3


            gxele = (1.0/6.0)*np.array([ [b1, b2, b3],
                                      [b1, b2, b3],
                                      [b1, b2, b3] ])
            gyele = (1.0/6.0)*np.array([ [c1, c2, c3],
                                      [c1, c2, c3],
                                      [c1, c2, c3] ])

            kxele=(1.0/(4.0*area))*np.array([ [b1**2, b1*b2, b1*b3],
                                      [b2*b1, b2**2, b2*b3],
                                      [b3*b1, b3*b2, b3**2] ])
            kyele=(1.0/(4.0*area))*np.array([ [c1**2, c1*c2, c1*c3],
                                      [c2*c1, c2**2, c2*c3],
                                      [c3*c1, c3*c2, c3**2] ])
            kxyele=(1.0/(4.0*area))*np.array([ [b1*c1, b1*c2, b1*c3],
                                      [b2*c1, b2*c2, b2*c3],
                                      [b3*c1, b3*c2, b3*c3] ])
            kest = VX*self.dt/2*(VX*kxele+VY*kxyele)+VY*self.dt/2*(VX*kxyele+VY*kyele)
            kele = kxele+kyele

            for i in range(0,3):
                ii = self.IEN[e,i]
                for j in range(0,3):
                    jj = self.IEN[e,j]
                    # montagem (assembling) das matrizes K e M

                    K[ii,jj] += self.nu * kele[i,j]
                    M[ii,jj] +=  m[i,j]
                    Gx[ii,jj] +=  gxele[i,j]
                    Gy[ii,jj] +=  gyele[i,j]
                    Kest[ii,jj] +=  kest[i,j]
                    Kx[ii,jj] += kxele[i,j]
                    Ky[ii,jj] += kyele[i,j]
                    Kxy[ii,jj]+= kxyele[i,j]
        return K, M, Gx, Gy, Kest, Kx, Ky, Kxy

def csr_zero_rows(csr, rows_to_zero):
    rows, cols = csr.shape
    mask = np.ones((rows,), dtype=np.bool)
    mask[rows_to_zero] = False
    nnz_per_row = np.diff(csr.indptr)
    mask = np.repeat(mask, nnz_per_row)
    nnz_per_row[rows_to_zero] = 0
    csr.data = csr.data[mask]
    csr.indices = csr.indices[mask]
    csr.indptr[1:] = np.cumsum(nnz_per_row)
