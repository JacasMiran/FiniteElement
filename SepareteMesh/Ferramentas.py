from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
import numpy as np

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

def matrix_cretation(npoints,ne,X,Y,IEN,vx,vy,dt,nu):
    K = lil_matrix( (npoints,npoints), dtype='float')
    M = lil_matrix( (npoints,npoints), dtype='float')
    Gx = lil_matrix( (npoints,npoints), dtype='float')
    Gy = lil_matrix( (npoints,npoints), dtype='float')
    Kest = lil_matrix( (npoints,npoints), dtype='float')
    Kx = lil_matrix( (npoints,npoints), dtype='float')
    Ky = lil_matrix( (npoints,npoints), dtype='float')
    Kxy = lil_matrix( (npoints,npoints), dtype='float')


    for e in range(0,ne):
        v = IEN[e]
        # area do elemento
        det = X[v[2]]*( Y[v[0]]-Y[v[1]]) \
              + X[v[0]]*( Y[v[1]]-Y[v[2]]) \
              + X[v[1]]*(-Y[v[0]]+Y[v[2]])
        area = abs(det)/2.0
        m = (area/12.0) * np.array([ [2.0, 1.0, 1.0],
                                  [1.0, 2.0, 1.0],
                                  [1.0, 1.0, 2.0] ])
        # formula do k do elementro triangular linear
        b1 = Y[v[1]]-Y[v[2]]
        b2 = Y[v[2]]-Y[v[0]]
        b3 = Y[v[0]]-Y[v[1]]
        c1 = X[v[2]]-X[v[1]]
        c2 = X[v[0]]-X[v[2]]
        c3 = X[v[1]]-X[v[0]]
        VX = (vx[v[0]]+vx[v[1]]+vx[v[2]])/3
        VY = (vy[v[0]]+vy[v[1]]+vy[v[2]])/3


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
        kest = VX*dt/2*(VX*kxele+VY*kxyele)+VY*dt/2*(VX*kxyele+VY*kyele)
        kele = kxele+kyele

        for i in range(0,3):
            ii = IEN[e,i]
            for j in range(0,3):
                jj = IEN[e,j]
                # montagem (assembling) das matrizes K e M
                K[ii,jj] += nu*kele[i,j]
                M[ii,jj] +=  m[i,j]
                Gx[ii,jj] +=  gxele[i,j]
                Gy[ii,jj] +=  gyele[i,j]
                Kest[ii,jj] +=  kest[i,j]
                Kx[ii,jj] += kxele[i,j]
                Ky[ii,jj] += kyele[i,j]
                Kxy[ii,jj]+= kxyele[i,j]

    return K, M, Gx, Gy, Kest, Kx, Ky, Kxy
