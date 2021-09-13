import numpy as np



import multiprocessing as mp
from Ferramentas import *

from scipy.sparse.linalg import cgs
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import spsolve
import meshio


import time

inicio = time.time()
# definicao das constantes do problema
nu = 1.0


velini = 1.0 #velocidade inicial
dt = 0.01
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

opl=mp.cpu_count()
K =  lil_matrix( (npoints,npoints), dtype='float')
M =  lil_matrix( (npoints,npoints), dtype='float')
Gx =  lil_matrix( (npoints,npoints), dtype='float')
Gy =  lil_matrix( (npoints,npoints), dtype='float')
Kest =  lil_matrix( (npoints,npoints), dtype='float')


# cc is the boundary condition indices
cc = msh.cells_dict["line"]
cc.reshape((cc.size)) # reshape for line vector
cc.sort()
cc = np.unique(cc) # removing duplicates
cc2=[]
for i in cc:
    if X[i]**2 + Y[i]**2 < 0.501**2:
        cc2.append(i)
    else:
        pass
listOfNeighbors = [0]*npoints
for i in range(0,len(listOfNeighbors)):
    listOfNeighbors[i]=[]

for e in range(0,ne):
    v = IEN[e]
    for i in range(0,len(v)):
        b=np.delete(v,i)
        for j in v:
            listOfNeighbors[v[i]].append(j)
            listOfNeighbors[v[i]]=np.array(listOfNeighbors[v[i]])
            listOfNeighbors[v[i]]=np.unique(listOfNeighbors[v[i]])
            listOfNeighbors[v[i]]=listOfNeighbors[v[i]].tolist()
#print (msh.cells_dict)

# inicializar os campos vx e vy
vx = np.zeros( (npoints,1),dtype='float')
vy = np.zeros( (npoints,1),dtype='float')
mt = np.ones( (npoints,1),dtype='float')

# loop dos elementos da malha



eps = 1e-6

for i in cc:
    if X[i] - eps < X.min():
        vx[i] = velini
        vy[i] = 0.0
    elif Y[i] + eps > Y.min():
        vx[i] = velini
        vy[i] = 0.0
    elif Y[i] + eps > Y.max():
        vx[i] = velini
        vy[i] = 0.0
    elif X[i] + eps > X.max():
        vx[i] = velini
        vy[i] = 0.0
    else:
        pass
for i in cc2:
    vx[i] = 0.0
    vy[i] = 0.0



fim = time.time()


print('Tempo de montagem',(fim - inicio))
y_coordinate = 0
velocidade = 0

#############################################################################
for INTE in range(0,10000):
    inicio = time.time()
    K, M, Gx, Gy, Kest, Kx, Ky, Kxy = matrix_cretation(npoints,ne,X,Y,IEN,vx,vy,dt,nu)


    ips=1e-6
    b = Gx.dot(vy) - Gy.dot(vx)
    omega, error = cgs(M,b,tol=ips)
    omega=(mt.transpose()*omega).transpose()
   # calculo do omega para inclusao no contorno
    omegacc = omega.copy()
# # zerando os valores de omegacc no interior da malha
    for i in cc:
         omegacc[i] = 0.0
         # v \dot \nabla \omega

    vgo = Gx.multiply(vx) + Gy.multiply(vy)

    A = (1.0/dt) * M + vgo + K + Kest

    b_1 = ((1.0/dt)*M.dot(omega))# - (vgo + K + Kest).dot(omega)

    A_2 = K.tocsr()

    psicc =  np.zeros( (npoints,1), dtype='float')
    A = A.tocsr()
    for i in cc:
        csr_zero_rows(A,i)#zerando
        csr_zero_rows(A_2,i)#zerando
    A = lil_matrix(A)
    A_2 = lil_matrix(A_2)
    for i in cc:
        A[i,i] = 1.0 # impondo 1 na diagonal
        A_2[i,i] = 1.0 # impondo 1 na diagonal psi
        b_1[i] = omegacc[i]

    A_2 = A_2.tocsc()
    A = A.tocsc()
    M_x = lambda x: spsolve(A,b_1)
    pre = LinearOperator((npoints, npoints), M_x)
# resolve eq. transporte solve
    omega, error = cgs(A,b_1,tol=ips,M=pre)
    omegap = (mt.transpose()*omega).transpose()

    b_2 = M.dot(omega)
    for i in cc: #condições de contorno psi
        if X[i] - eps < X.min():
            psicc[i] = Y[i]*velini
        elif Y[i] - eps < Y.min():
            psicc[i] = Y[i]*velini
        elif Y[i] + eps > Y.max():
            psicc[i] = Y[i]*velini
        elif X[i] + eps > X.max():
            psicc[i] = Y[i]*velini
        else:
            psicc[i] = Y[i]*velini
    for i in cc2:
        psicc[i] = y_coordinate*velini
    for i in cc:
        b_2[i] = psicc[i] #psi



    M_x1 = lambda x: spsolve(A_2,b_2)
    pre1 = LinearOperator((npoints, npoints), M_x1)
    psi, error = cgs(A_2,b_2,tol=ips,M=pre1)
    psip = (mt.transpose()*psi).transpose()


    Kpressure = K*1
    Kpressure = Kpressure.tocsr()
    bpressure = 2*((Kx.dot(psi))*(Ky.dot(psi))-(Kxy.dot(psi)*Kxy.dot(psi)))

    for i in cc:
        if X[i] + eps > X.max():
            csr_zero_rows(Kpressure,i)
            bpressure[i] = 0
        else:
            bpressure[i] = 0
    Kpressure = Kpressure.tolil()
    for i in cc:
        if X[i] + eps > X.max():
            Kpressure[i,i] = 1
    Kpressure = Kpressure.tocsr()


    M_x10 = lambda x: spsolve(Kpressure,bpressure)
    pre10 = LinearOperator((npoints, npoints), M_x10)
    p, error = cgs(Kpressure,bpressure,tol=ips,M=pre10)
    pp = (mt.transpose()*p).transpose()

# encontrar velocidades M vx = Gy \psi, M vy = -Gx \psi
    M=M.tocsc()

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
    eps = 1e-6
    for i in cc:
        if X[i] - eps < X.min():
            vx[i] = velini
            vy[i] = 0.0
        elif Y[i] + eps > Y.min():
            vx[i] = velini
            vy[i] = 0.0
        elif Y[i] + eps > Y.max():
            vx[i] = velini
            vy[i] = 0.0
        elif X[i] + eps > X.max():
            vx[i] = velini
            vy[i] = 0.0
        else:
            pass
    for i in cc2:
        vx[i] = 0.0
        vy[i] = 0.0
    gradu = Gx.dot(vx)
    gradv = Gy.dot(vy)
    gradtotal = gradu + gradv
    if INTE%1==0:
        meshio.write_points_cells(
            "inte{}.vtk".format(INTE),
            xyz,
            cells,
            # Optionally provide extra data on points, cells, etc.
            point_data={'vx':vx,'vy':vy,'psi':psip, 'omega': omegap, 'p':pp, 'arrasto':gradtotal}
            # cell_data=cell_data,
            # field_data=field_data
            )
    forceTotalX = 0
    forceTotalY = 0

    for i in range(0,len(cc2)):
        if i+1 == len(cc2):
            modulo_vetor = (((X[cc2[i]]+X[cc2[0]])/2)**2 + ((Y[cc2[i]]+Y[cc2[0]])/2)**2)**1/2
            direcao_do_vetor_unitario_X = -((X[cc2[i]]+X[cc2[0]])/2)/modulo_vetor
            direcao_do_vetor_unitario_Y = -((Y[cc2[i]]+Y[cc2[0]])/2)/modulo_vetor
            forceTotalX += (p[cc2[i]]+p[cc2[0]])*direcao_do_vetor_unitario_X / 2
            forceTotalY += (p[cc2[i]]+p[cc2[0]])*direcao_do_vetor_unitario_Y / 2
        else:
            modulo_vetor = (((X[cc2[i]]+X[cc2[i+1]])/2)**2 + ((Y[cc2[i]]+Y[cc2[i]+1])/2)**2)**1/2
            direcao_do_vetor_unitario_X = -((X[cc2[i]+1]+X[cc2[i+1]])/2)/modulo_vetor
            direcao_do_vetor_unitario_Y = -((Y[cc2[i]]+Y[cc2[i+1]])/2)/modulo_vetor
            forceTotalX += (p[cc2[i]]+p[cc2[i+1]])*direcao_do_vetor_unitario_X / 2
            forceTotalY += (p[cc2[i]]+p[cc2[i+1]])*direcao_do_vetor_unitario_Y / 2


    #print ("Cd:",forceTotalX*2/velini**2)
    #print("Cl:",forceTotalY*2/velini**2)
    k_mola = 100000.0
    massa_cilindro = 100.0
    forca_mola = y_coordinate*k_mola
    aceleracao = (forceTotalY-forca_mola)/massa_cilindro
    velocidade += aceleracao*dt
    y_coordinate += velocidade*dt
    print("velocidade:",velocidade)
    #print("forca da mola:",forca_mola)
    #print("forca de lift:",forceTotalY)

    for i in range(0,npoints):
        if (i in cc) == False and np.absolute(X[i]) <= 5.0:
            a1 = ( 1.02 / ( y_coordinate - 14.5 ) ) * np.absolute(Y[i])
            b1 = ( 15 * 1.02 / (14.5 - y_coordinate ) )
            a2 = ( 1.02 / (y_coordinate - velocidade*dt - 14.5) ) * np.absolute(Y[i])
            b2 = ( 15 * 1.02 / (14.5 - (y_coordinate - velocidade*dt) ) )
            zxcv = y_coordinate*(a1+b1)*((4.5-np.absolute(X[i]))/5.0)
            zxcv2 = (y_coordinate - velocidade*dt)*(a2+b2)*((4.5-np.absolute(X[i]))/5.0)
            vy[i] = vy[i] - (zxcv-zxcv2)/dt
            Y[i] = Y[i] + zxcv-zxcv2


    for _ in range(0,3):
        for i in range(0,npoints):
            medX=0
            medY=0
            for j in listOfNeighbors[i]:
                medX += X[j] / len(listOfNeighbors[i])
                medY += Y[j] / len(listOfNeighbors[i])
            vx[i] = vx[i] - (X[i]-medX)/dt
            vy[i] = vy[i] - (Y[i]-medY)/dt
            if i in cc and (i in cc2) == False:
                X[i] = Xstatic[i]
                Y[i] = Ystatic[i]
                vx[i] = velini
                vy[i] = 0.0
            elif i in cc2:
                X[i] = Xstatic[i]
                Y[i] += velocidade*dt
                vx[i] = 0.0
                vy[i] = 0.0
            else:
                X[i] = medX
                Y[i] = medY
    fim = time.time()
    if INTE%1==0:
        print ('Interação:',INTE)
        print('Tempo da interação:',(fim - inicio))
        print(omega.max(),omega.min())
