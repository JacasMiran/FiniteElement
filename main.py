import numpy as np
import multiprocessing as mp
from scipy.sparse.linalg import cgs
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix
import meshio
import time

from module_for_matrix_creation import Matrix_Creation


nu = 1.0
dt = 0.001

#definition of problem constants


msh = meshio.read("MeshAtualizada.msh")

IEN = msh.cells_dict["triangle"]
#print(dir(msh))
cells = [("triangle", IEN)]

X = msh.points[:,0] #O indice do array é o numero do elemento
Y = msh.points[:,1] #O indice do array é o numero do elemento
xyz = msh.points

npoints = len(X) #number of points
ne = len(IEN)
vx = np.zeros( (npoints,1),dtype='float')
vy = np.zeros( (npoints,1),dtype='float')

cc = msh.cells_dict["line"] #retorna os elementos linha 
cc_data = msh.cell_data_dict["gmsh:physical"]["line"] #retorna o label orientado a lista cc

#for i in range(0,len(cc)):
#    print(X[cc[i][0]],Y[cc[i][1]],cc_data[i])

test = Matrix_Creation(npoints, ne, X, Y, IEN, vx, vy, dt, nu)

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

def retorna_algo():
    with mp.Pool(8) as p:
        a = p.starmap(test.matrix_mouting, list_cores)
    return a

a = retorna_algo()

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

end_par = time.time() # Fim do tempo da  computação paralela

print ("Tempo de computação paralela",end_par - start_par)

start_normal = time.time()

K, M, Gx, Gy, Kest, Kx, Ky, Kxy = test.matrix_mouting(0, ne)

end_normal = time.time()

print("Tempo de computação sincrona", end_normal - start_normal)

print ("Numero de Elementos: ", ne)