import numpy as np
import multiprocessing as mp
from Ferramentas import *

from scipy.sparse.linalg import cgs
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import spsolve
import meshio
from module_for_matrix_creation import Matrix_Creation
import time
inicio = time.time()
# definicao das constantes do problema
nu = 1.0


velini = 1.0 #velocidade inicial
dt = 0.0001
msh1 = meshio.gmsh.main.read("Cavidade.msh")
msh = meshio.read("Cavidade.msh")
#print(msh.cell_data['gmsh:physical'][0])

test = meshio.read("Cavidade.msh")
IEN = msh.cells_dict["triangle"]
cells=[('triangle',IEN)]

X = msh.points[:,0]
Y = msh.points[:,1]
Ystatic = test.points[:,1]
Xstatic = test.points[:,0]
xyz = msh.points
#print(xyz)
npoints = len(X)

ne = len(IEN)

vx = np.zeros( (npoints,1),dtype='float')
vy = np.zeros( (npoints,1),dtype='float')

cpu_cores = mp.cpu_count()
Matrices_Object = Matrix_Creation(npoints, ne, X, Y, IEN, vx, vy, dt, nu)

start_par = time.time() #Inicio do tempo para computação paralela

cpu_cores = mp.cpu_count()
list_cores = []
tuple_int = ()


for i in range(0, cpu_cores):
    if i < cpu_cores-1:
        tuple_int = ((i * (ne // cpu_cores)), (i+1) * (ne // cpu_cores))
        list_cores.append(tuple_int)
    else:
        tuple_int = (i * (ne // cpu_cores) ,(i+1) * (ne // cpu_cores) + ne%cpu_cores)
        list_cores.append(tuple_int)

def MatricesInParallel():
    with mp.Pool(cpu_cores) as p:
        a = p.starmap(Matrices_Object.matrix_mouting, list_cores)
    return a

a = MatricesInParallel()

K    = lil_matrix( (npoints,npoints), dtype='float')
M    = lil_matrix( (npoints,npoints), dtype='float')
Gx   = lil_matrix( (npoints,npoints), dtype='float')
Gy   = lil_matrix( (npoints,npoints), dtype='float')
Kest = lil_matrix( (npoints,npoints), dtype='float')
Kx   = lil_matrix( (npoints,npoints), dtype='float')
Ky   = lil_matrix( (npoints,npoints), dtype='float')
Kxy  = lil_matrix( (npoints,npoints), dtype='float')

for i in a:
    K    = np.add(K,i[0])
    M    = np.add(M,i[1])
    Gx   = np.add(Gx,i[2])
    Gy   = np.add(Gy,i[3])
    Kest = np.add(Kest,i[4])
    Kx   = np.add(Kx,i[5])
    Ky   = np.add(Ky,i[6])
    Kxy  = np.add(Kxy,i[7])




# cc is the boundary condition indices
cc = msh.cells_dict["line"]
cc_data = msh.cell_data_dict["gmsh:physical"]["line"]

cc2 = msh.cells_dict["line"]
cc2.reshape((cc.size)) # reshape for line vector
cc2.sort()
cc2 = np.unique(cc2)

inner = [x for x in range(0,npoints) if x not in cc]

for i in range(0, len(cc)):
    if cc_data[i] == 3:
        vx[cc[i][0]] = velini
        vx[cc[i][1]] = velini
        vy[cc[i][0]] = 0
        vy[cc[i][1]] = 0
    else:
        vx[cc[i][0]] = 0
        vx[cc[i][1]] = 0
        vy[cc[i][0]] = 0
        vy[cc[i][1]] = 0
#print (msh.cells_dict)

# inicializar os campos vx e vy
vx = np.zeros( (npoints,1),dtype='float')
vy = np.zeros( (npoints,1),dtype='float')
mt = np.ones( (npoints,1),dtype='float')

# loop dos elementos da malha



eps = 1e-6

fim = time.time()


print('Tempo de montagem',(fim - inicio))
y_coordinate = 0
velocidade = 0

#############################################################################
for INTE in range(0,10000):
    inicio = time.time()

    ips=1e-6
    b = Gx.dot(vy) - Gy.dot(vx)
    omega, error = cgs(M,b,tol=ips)
    omega = (mt.transpose()*omega).transpose()
   # calculo do omega para inclusao no contorno
    omegacc = omega.copy()
# # zerando os valores de omegacc no interior da malha
    for i in inner:
        omegacc[i] = 0.0
         # v \dot \nabla \omega

    vgo = Gx.multiply(vx) + Gy.multiply(vy)

    A = (1.0/dt) * M + vgo + K

    b_1 = ((1.0/dt)*M.dot(omega))# - (vgo + K + Kest).dot(omega)

    A_2 = K.tocsr()

    psicc =  np.zeros( (npoints,1), dtype='float')
    A = A.tocsr()
    for i in range(0, len(cc)):
        csr_zero_rows(A,cc[i][0])#zerando
        csr_zero_rows(A,cc[i][1])#zerando

        csr_zero_rows(A_2,cc[i][0])#zerando
        csr_zero_rows(A_2,cc[i][1])#zerando

    A = lil_matrix(A)
    A_2 = lil_matrix(A_2)

    for i in range(0, len(cc)):
        A[cc[i][0],cc[i][0]] = 1.0
        A[cc[i][1],cc[i][1]] = 1.0 # impondo 1 na diagonal
        A_2[cc[i][0],cc[i][0]] = 1.0
        A_2[cc[i][1],cc[i][1]] = 1.0 # impondo 1 na diagonal psi
        b_1[cc[i][0]] = omegacc[cc[i][0]]
        b_1[cc[i][1]] = omegacc[cc[i][1]]

    A_2 = A_2.tocsc()
    A = A.tocsc()
    M_x = lambda x: spsolve(A,b_1)
    pre = LinearOperator((npoints, npoints), M_x)
# resolve eq. transporte solve
    omega, error = cgs(A,b_1,tol=ips,M=pre)
    omegap = (mt.transpose()*omega).transpose()

    b_2 = M.dot(omega)

    for i in range(0, len(cc)):
        b_2[cc[i][0]] = psicc[cc[i][0]] #psi
        b_2[cc[i][1]] = psicc[cc[i][1]]

    M_x1 = lambda x: spsolve(A_2,b_2)
    pre1 = LinearOperator((npoints, npoints), M_x1)
    psi, error = cgs(A_2,b_2,tol=ips,M=pre1)
    psip = (mt.transpose()*psi).transpose()

# encontrar velocidades M vx = Gy \psi, M vy = -Gx \psi
    M = M.tocsc()

    b_3 = (mt.transpose()*Gy.dot(psi)).transpose()
    M_x = lambda x: spsolve(M,b_3)
    pre = LinearOperator((npoints, npoints), M_x)
    vxT, error = cgs(M,b_3,tol=ips,M=pre)
    vx  = (mt.transpose()*vxT).transpose()

    b_4 = (mt.transpose()*Gx.dot(psi)).transpose()
    M_x = lambda x: spsolve(M,b_4)
    pre = LinearOperator((npoints, npoints), M_x)
    vyT= -1*cgs(M,b_4,tol=ips,M=pre)[0]
    vy  = (mt.transpose()*vyT).transpose()


 #impor cc de velocidade nos vetores vx e vy
 #impondo cc para vx e vy

    for i in range(0, len(cc)):
        if cc_data[i] == 3:
            vx[cc[i][0]] = 1
            vx[cc[i][1]] = 1
            vy[cc[i][0]] = 0
            vy[cc[i][1]] = 0
        else:
            vx[cc[i][0]] = 0
            vx[cc[i][1]] = 0
            vy[cc[i][0]] = 0
            vy[cc[i][1]] = 0

    if INTE%10==0:
        meshio.write_points_cells(
            "inte{}.vtk".format(INTE),
            xyz,
            cells,
            # Optionally provide extra data on points, cells, etc.
            point_data={'vx':vx,'vy':vy,'psi':psip, 'omega': omegap}
            # cell_data=cell_data,
            # field_data=field_data
            )
    fim = time.time()
    if INTE%10==0:
        print ('Interação:',INTE)
        print('Tempo da interação:',(fim - inicio))
        print(omega.max(),omega.min())
