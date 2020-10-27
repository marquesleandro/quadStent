# =======================
# Importing the libraries
# =======================

import os
initial_path = os.getcwd()

import sys
folderClass = './libClass'
sys.path.insert(0, folderClass)

from tqdm import tqdm
from time import time

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg

import searchMSH
import importMSH
import assembly
import benchmarkProblems
import importVTK
import ALE	
import semiLagrangian
import exportVTK
import relatory



print '''
               COPYRIGHT                    
 ======================================
 Simulator: %s
 created by Leandro Marques at 02/2019
 e-mail: marquesleandro67@gmail.com
 Gesar Search Group
 State University of the Rio de Janeiro
 ======================================
\n''' %sys.argv[0]



print ' ------'
print ' INPUT:'
print ' ------'

print ""


print ' ----------------------------------------------------------------------------'
print ' (0) - Import VTK OFF'
print ' (1) - Import VTK ON'
import_option = int(raw_input("\n enter option above: "))
if import_option == 1:
 folderName = raw_input("\n enter simulation folder name VTK import: ")
 numberStep = int(raw_input("\n enter number step VTK import: "))
print' ----------------------------------------------------------------------------\n'


print ' ----------------------------------------------------------------------------'
print ' (0) - Debug'
print ' (1) - Simulation'
simulation_option = int(raw_input("\n enter simulation option above: "))
print' ----------------------------------------------------------------------------\n'





print ' ----------------------------------------------------------------------------'
print ' (1) - Taylor Galerkin Scheme'
print ' (2) - Semi Lagrangian Scheme'
scheme_option = int(raw_input("\n Enter simulation scheme option above: "))
if scheme_option == 1:
 scheme_name = 'Taylor Galerkin'
elif scheme_option == 2:
 scheme_name = 'Semi Lagrangian'
print' ----------------------------------------------------------------------------\n'



print ' ----------------------------------------------------------------------------'
print ' (0) - Analytic Linear Element'
print ' (1) - Linear Element'
print ' (2) - Mini Element'
print ' (3) - Quadratic Element'
print ' (4) - Cubic Element'
polynomial_option = int(raw_input("\n Enter polynomial degree option above: "))
print' ----------------------------------------------------------------------------\n'


if simulation_option == 1:
 if polynomial_option == 0:
  gausspoints = 3

 else:
  print ' ----------------------------------------------------------------------------'
  print ' 3 Gauss Points'
  print ' 4 Gauss Points'
  print ' 6 Gauss Points'
  print ' 12 Gauss Points'
  gausspoints = int(raw_input("\n Enter Gauss Points Number option above: "))
  print' ----------------------------------------------------------------------------\n'


 
 print ' ----------------------------------------------------------------------------'
 nt = int(raw_input(" Enter number of time interations (nt): "))
 print' ----------------------------------------------------------------------------\n'
 
 
 print ' ----------------------------------------------------------------------------'
 folderResults = raw_input(" Enter folder name to save simulations: ")
 print' ----------------------------------------------------------------------------\n'

 print ' ----------------------------------------------------------------------------'
 observation = raw_input(" Digit observation: ")
 print' ----------------------------------------------------------------------------\n'


elif simulation_option == 0:
 gausspoints = 3
 nt = 3
 folderResults  = 'deletar'
 observation = 'debug'



print '\n ------------'
print ' IMPORT MESH:'
print ' ------------'

start_time = time()

# Linear and Mini Elements
if polynomial_option == 0 or polynomial_option == 1 or polynomial_option == 2:
 #mshFileName = 'linearHalfPoiseuille.msh'
 #mshFileName = 'linearStraightGeo.msh'
 mshFileName = 'linearCurvedGeoStrut1.msh'

 pathMSHFile = searchMSH.Find(mshFileName)
 if pathMSHFile == 'File not found':
  sys.exit()

 if polynomial_option == 0 or polynomial_option == 1:
  mesh = importMSH.Linear2D(pathMSHFile, mshFileName)

  numNodes               = mesh.numNodes
  numElements            = mesh.numElements
  x                      = mesh.x
  y                      = mesh.y
  IEN                    = mesh.IEN
  boundaryEdges          = mesh.boundaryEdges
  boundaryNodes          = mesh.boundaryNodes
  neighborsNodes         = mesh.neighborsNodes
  neighborsNodesALE      = mesh.neighborsNodesALE
  neighborsElements      = mesh.neighborsElements
  minLengthMesh          = mesh.minLengthMesh
  FreedomDegree          = mesh.FreedomDegree
  numPhysical            = mesh.numPhysical 

  Re = 54.5
  Sc = 1.0
  CFL = 0.5
  dt = float(CFL*minLengthMesh)
  #dt = 0.1   #SL 

 elif polynomial_option == 2:
  mesh = importMSH.Mini2D(pathMSHFile, mshFileName)

  numNodes               = mesh.numNodes
  numElements            = mesh.numElements
  x                      = mesh.x
  y                      = mesh.y
  IEN                    = mesh.IEN
  boundaryEdges          = mesh.boundaryEdges
  boundaryNodes          = mesh.boundaryNodes
  neighborsNodes         = mesh.neighborsNodes
  neighborsNodesALE      = mesh.neighborsNodesALE
  neighborsElements      = mesh.neighborsElements
  minLengthMesh          = mesh.minLengthMesh
  FreedomDegree          = mesh.FreedomDegree
  numPhysical            = mesh.numPhysical 
  Re = 54.5
  Sc = 1.0
  CFL = 0.5
  dt = float(CFL*minLengthMesh)
  #dt = 0.1   #linear result ok 




# Quad Element
elif polynomial_option == 3:
 #mshFileName = 'quadHalfPoiseuille.msh'
 #mshFileName = 'quadStraightGeo.msh'
 #mshFileName = 'quadCurvedGeoStrut.msh'
 mshFileName = 'quadRealGeoStrut.msh'
 #mshFileName = 'quadCurvedGeo.msh'


 
 pathMSHFile = searchMSH.Find(mshFileName)
 if pathMSHFile == 'File not found':
  sys.exit()

 mesh = importMSH.Quad2D(pathMSHFile, mshFileName)

 numNodes               = mesh.numNodes
 numElements            = mesh.numElements
 x                      = mesh.x
 y                      = mesh.y
 IEN                    = mesh.IEN
 boundaryEdges          = mesh.boundaryEdges
 boundaryNodes          = mesh.boundaryNodes
 neighborsNodes         = mesh.neighborsNodes
 neighborsNodesALE      = mesh.neighborsNodesALE
 neighborsElements      = mesh.neighborsElements
 minLengthMesh          = mesh.minLengthMesh
 FreedomDegree          = mesh.FreedomDegree
 numPhysical            = mesh.numPhysical 

 Re = 54.5
 Sc = 1.0
 CFL = 0.5
 #dt = float(CFL*minLengthMesh)
 dt = 0.0005  




# Cubic Element
elif polynomial_option == 4:
 mshFileName = 'cubicStent_cubic.msh'

 pathMSHFile = searchMSH.Find(mshFileName)
 if pathMSHFile == 'File not found':
  sys.exit()

 mesh = importMSH.Cubic2D(pathMSHFile, mshFileName, equation_number)
 mesh.coord()
 mesh.ien()



end_time = time()
import_mesh_time = end_time - start_time
print ' time duration: %.1f seconds \n' %import_mesh_time




# -------------------------- Import VTK File ------------------------------------
if import_option == 0:
 import_option = 'OFF'
 
 print ' ---------'
 print ' ASSEMBLY:'
 print ' ---------'
 
 start_time = time()
 Kxx, Kxy, Kyx, Kyy, K, M, MLump, Gx, Gy, polynomial_order = assembly.Element2D(simulation_option, polynomial_option, FreedomDegree, numNodes, numElements, IEN, x, y, gausspoints)
 
 
 end_time = time()
 assembly_time = end_time - start_time
 print ' time duration: %.1f seconds \n' %assembly_time
 
 
 
 
 print ' --------------------------------'
 print ' INITIAL AND BOUNDARY CONDITIONS:'
 print ' --------------------------------'
 
 start_time = time()
 
 
 # ------------------------ Boundaries Conditions ----------------------------------
 
 # Linear and Mini Elements
 if polynomial_option == 0 or polynomial_option == 1 or polynomial_option == 2:
 
  # Applying vx condition
  xVelocityLHS0 = sps.lil_matrix.copy(M)
  xVelocityBC = benchmarkProblems.linearStent(numPhysical,numNodes,x,y)
  xVelocityBC.xVelocityCondition(boundaryEdges,xVelocityLHS0,neighborsNodes)
  benchmark_problem = xVelocityBC.benchmark_problem
 
  # Applying vr condition
  yVelocityLHS0 = sps.lil_matrix.copy(M)
  yVelocityBC = benchmarkProblems.linearStent(numPhysical,numNodes,x,y)
  yVelocityBC.yVelocityCondition(boundaryEdges,yVelocityLHS0,neighborsNodes)
  
  # Applying psi condition
  streamFunctionLHS0 = sps.lil_matrix.copy(Kxx) + sps.lil_matrix.copy(Kyy)
  streamFunctionBC = benchmarkProblems.linearStent(numPhysical,numNodes,x,y)
  streamFunctionBC.streamFunctionCondition(boundaryEdges,streamFunctionLHS0,neighborsNodes)
 
  # Applying vorticity condition
  vorticityDirichletNodes = boundaryNodes
 
  # Applying concentration condition
  concentrationLHS0 = (sps.lil_matrix.copy(M)/dt) + (1.0/(Re*Sc))*sps.lil_matrix.copy(Kxx) + (1.0/(Re*Sc))*sps.lil_matrix.copy(Kyy)
  concentrationBC = benchmarkProblems.linearStent(numPhysical,numNodes,x,y)
  concentrationBC.concentrationCondition(boundaryEdges,concentrationLHS0,neighborsNodes)
 
 
 # Quad Element
 elif polynomial_option == 3:
 
  # Applying vx condition
  xVelocityLHS0 = sps.lil_matrix.copy(M)
  xVelocityBC = benchmarkProblems.quadStent(numPhysical,numNodes,x,y)
  xVelocityBC.xVelocityCondition(boundaryEdges,xVelocityLHS0,neighborsNodes)
  benchmark_problem = xVelocityBC.benchmark_problem
 
  # Applying vr condition
  yVelocityLHS0 = sps.lil_matrix.copy(M)
  yVelocityBC = benchmarkProblems.quadStent(numPhysical,numNodes,x,y)
  yVelocityBC.yVelocityCondition(boundaryEdges,yVelocityLHS0,neighborsNodes)
  
  # Applying psi condition
  streamFunctionLHS0 = sps.lil_matrix.copy(Kxx) + sps.lil_matrix.copy(Kyy)
  streamFunctionBC = benchmarkProblems.quadStent(numPhysical,numNodes,x,y)
  streamFunctionBC.streamFunctionCondition(boundaryEdges,streamFunctionLHS0,neighborsNodes)
 
  # Applying vorticity condition
  vorticityDirichletNodes = boundaryNodes
 
  # Applying concentration condition
  concentrationLHS0 = (sps.lil_matrix.copy(M)/dt) + (1.0/(Re*Sc))*sps.lil_matrix.copy(Kxx) + (1.0/(Re*Sc))*sps.lil_matrix.copy(Kyy)
  concentrationBC = benchmarkProblems.quadStent(numPhysical,numNodes,x,y)
  concentrationBC.concentrationCondition(boundaryEdges,concentrationLHS0,neighborsNodes)
 # ---------------------------------------------------------------------------------
 
 
 
 # -------------------------- Initial condition ------------------------------------
 vx = np.copy(xVelocityBC.aux1BC)
 vy = np.copy(yVelocityBC.aux1BC)
 psi = np.copy(streamFunctionBC.aux1BC)
 w = np.zeros([numNodes,1], dtype = float)
 c = np.copy(concentrationBC.aux1BC)
 # ---------------------------------------------------------------------------------
 
 
 
 
 #---------- Step 1 - Compute the vorticity and stream field --------------------
 # -----Vorticity initial-----
 vorticityRHS = sps.lil_matrix.dot(Gx,vy) - sps.lil_matrix.dot(Gy,vx)
 vorticityLHS = sps.lil_matrix.copy(M)
 w = scipy.sparse.linalg.cg(vorticityLHS,vorticityRHS,w, maxiter=1.0e+05, tol=1.0e-05)
 w = w[0].reshape((len(w[0]),1))
 
 
 # -----Streamline initial-----
 streamFunctionRHS = sps.lil_matrix.dot(M,w)
 streamFunctionRHS = np.multiply(streamFunctionRHS,streamFunctionBC.aux2BC)
 streamFunctionRHS = streamFunctionRHS + streamFunctionBC.dirichletVector
 psi = scipy.sparse.linalg.cg(streamFunctionBC.LHS,streamFunctionRHS,psi, maxiter=1.0e+05, tol=1.0e-05)
 psi = psi[0].reshape((len(psi[0]),1))
 #----------------------------------------------------------------------------------


 end_time = time()
 bc_apply_time = end_time - start_time
 print ' time duration: %.1f seconds \n' %bc_apply_time


 

elif import_option == 1:
 import_option = 'ON'
 print "Import option ON"
 

 start_time = time()
 #numNodes, numElements, IEN, x, y, vx, vy, w, psi, c, polynomial_order, benchmark_problem = importVTK.vtkFile("/home/marquesleandro/quadStent/results/vorticityNull1/vorticityNull1311.vtk", polynomial_option)
 numNodes, numElements, IEN, x, y, vx, vy, w, psi, c, polynomial_order, benchmark_problem = importVTK.vtkFile("/home/marquesleandro/quadStent/results/" + folderName + "/" + folderName + str(numberStep) + ".vtk", polynomial_option)
 end_time = time()
 assembly_time = end_time - start_time 
 bc_apply_time = end_time - start_time 
 print ' time duration: %.1f seconds \n' %bc_apply_time
#----------------------------------------------------------------------------------







print ' -----------------------------'
print ' PARAMETERS OF THE SIMULATION:'
print ' -----------------------------'

print ' Benchmark Problem: %s' %benchmark_problem
print ' Scheme: %s' %str(scheme_name)
print ' Element Type: %s' %str(polynomial_order)
print ' Gaussian Quadrature (Gauss Points): %s' %str(gausspoints)
print ' Mesh: %s' %mshFileName
print ' Number of nodes: %s' %numNodes
print ' Number of elements: %s' %numElements
print ' Smallest edge length: %f' %minLengthMesh
print ' Time step: %s' %dt
print ' Import VTK: %s' %import_option
print ' Number of time iteration: %s' %nt
print ' Reynolds number: %s' %Re
print ' Schmidt number: %s' %Sc
print ""


print ' ----------------------------'
print ' SOLVE THE LINEARS EQUATIONS:'
print ' ---------------------------- \n'

print ' Saving simulation in %s \n' %folderResults



solution_start_time = time()
os.chdir(initial_path)



# ------------------------ Export VTK File ---------------------------------------
# Linear and Mini Elements
if polynomial_option == 0 or polynomial_option == 1 or polynomial_option == 2:   
 save = exportVTK.Linear2D(x,y,IEN,numNodes,numElements,w,psi,c,vx,vy)
 save.create_dir(folderResults)
 save.saveVTK(folderResults + str(0))

# Quad Element
elif polynomial_option == 3:   
 save = exportVTK.Quad2D(x,y,IEN,numNodes,numElements,w,psi,c,vx,vy)
 save.create_dir(folderResults)
 save.saveVTK(folderResults + str(0))
# ---------------------------------------------------------------------------------



vorticityAux1BC = np.zeros([numNodes,1], dtype = float) 
x_old = np.zeros([numNodes,1], dtype = float)
y_old = np.zeros([numNodes,1], dtype = float)
vx_old = np.zeros([numNodes,1], dtype = float)
vy_old = np.zeros([numNodes,1], dtype = float)
end_type = 0
for t in tqdm(range(1, nt)):
 numIteration = t

 try:
  print ""
  print '''
                 COPYRIGHT                    
   ======================================
   Simulator: %s
   created by Leandro Marques at 04/2019
   e-mail: marquesleandro67@gmail.com
   Gesar Search Group
   State University of the Rio de Janeiro
   ======================================
  ''' %sys.argv[0]
 
 
 
  print ' -----------------------------'
  print ' PARAMETERS OF THE SIMULATION:'
  print ' -----------------------------'
 
  print ' Benchmark Problem: %s' %benchmark_problem
  print ' Scheme: %s' %str(scheme_name)
  print ' Element Type: %s' %str(polynomial_order)
  print ' Gaussian Quadrature (Gauss Points): %s' %str(gausspoints)
  print ' Mesh: %s' %mshFileName
  print ' Number of nodes: %s' %numNodes
  print ' Number of elements: %s' %numElements
  print ' Smallest edge length: %f' %minLengthMesh
  print ' Time step: %s' %dt
  print ' Import VTK: %s' %import_option
  print ' Number of time iteration: %s' %numIteration
  print ' Reynolds number: %s' %Re
  print ' Schmidt number: %s' %Sc
 
 
 
  # ------------------------- ALE Scheme --------------------------------------------
  xmeshALE_dif = np.linalg.norm(x-x_old)
  ymeshALE_dif = np.linalg.norm(y-y_old)
  if not xmeshALE_dif < 5e-3 and not ymeshALE_dif < 5e-3:
   x_old = np.copy(x)
   y_old = np.copy(y)
  
   print ""
   print ' ------------'
   print ' MESH UPDATE:'
   print ' ------------'
  
  
   start_time = time()
  
  
   kLagrangian = 0.0
   kLaplace = 0.0
   kVelocity = 0.0
   
   vxLaplacianSmooth, vyLaplacianSmooth = ALE.Laplacian_smoothing(neighborsNodesALE, numNodes, x, y, dt)
   #vxLaplacianSmooth, vyLaplacianSmooth = ALE.Laplacian_smoothing_avg(neighborsNodesALE, numNodes, x, y, dt)
   vxVelocitySmooth,  vyVelocitySmooth  = ALE.Velocity_smoothing(neighborsNodesALE, numNodes, vx, vy)
 
   vxALE = kLagrangian*vx + kLaplace*vxLaplacianSmooth + kVelocity*vxVelocitySmooth
   vyALE = kLagrangian*vy + kLaplace*vyLaplacianSmooth + kVelocity*vyVelocitySmooth
  
  
   for i in boundaryNodes:
    node = i-1 
    vxALE[node] = 0.0
    vyALE[node] = 0.0
  
   x = x + vxALE*dt
   y = y + vyALE*dt
  
   vxSL = vx - vxALE
   vySL = vy - vyALE
  
   end_time = time()
   ALE_time_solver = end_time - start_time
   print ' time duration: %.1f seconds' %ALE_time_solver
   # ---------------------------------------------------------------------------------
  
 
 
 
   # ------------------------- Assembly --------------------------------------------
   print ""
   print ' ---------'
   print ' ASSEMBLY:'
   print ' ---------'
 
   Kxx, Kxy, Kyx, Kyy, K, M, MLump, Gx, Gy, polynomial_order = assembly.Element2D(simulation_option, polynomial_option, FreedomDegree, numNodes, numElements, IEN, x, y, gausspoints)
   # --------------------------------------------------------------------------------
 
 
 
 
   # ------------------------ Boundaries Conditions ----------------------------------
   print ""
   print ' --------------------------------'
   print ' INITIAL AND BOUNDARY CONDITIONS:'
   print ' --------------------------------'

   
   # Linear and Mini Elements
   if polynomial_option == 0 or polynomial_option == 1 or polynomial_option == 2:
 
    # Applying vx condition
    start_xVelocityBC_time = time()
    xVelocityLHS0 = sps.lil_matrix.copy(M)
    xVelocityBC = benchmarkProblems.linearStent(numPhysical,numNodes,x,y)
    xVelocityBC.xVelocityCondition(boundaryEdges,xVelocityLHS0,neighborsNodes)
    benchmark_problem = xVelocityBC.benchmark_problem
    end_xVelocityBC_time = time()
    xVelocityBC_time = end_xVelocityBC_time - start_xVelocityBC_time
    print ' xVelocity BC: %.1f seconds' %xVelocityBC_time

   
    # Applying vy condition
    start_yVelocityBC_time = time()
    yVelocityLHS0 = sps.lil_matrix.copy(M)
    yVelocityBC = benchmarkProblems.linearStent(numPhysical,numNodes,x,y)
    yVelocityBC.yVelocityCondition(boundaryEdges,yVelocityLHS0,neighborsNodes)
    end_yVelocityBC_time = time()
    yVelocityBC_time = end_yVelocityBC_time - start_yVelocityBC_time
    print ' yVelocity BC: %.1f seconds' %yVelocityBC_time



    
    # Applying psi condition
    start_streamfunctionBC_time = time()
    streamFunctionLHS0 = sps.lil_matrix.copy(Kxx) + sps.lil_matrix.copy(Kyy)
    streamFunctionBC = benchmarkProblems.linearStent(numPhysical,numNodes,x,y)
    streamFunctionBC.streamFunctionCondition(boundaryEdges,streamFunctionLHS0,neighborsNodes)
    end_streamfunctionBC_time = time()
    streamfunctionBC_time = end_streamfunctionBC_time - start_streamfunctionBC_time
    print ' streamfunction BC: %.1f seconds' %streamfunctionBC_time



   
    # Applying vorticity condition
    vorticityDirichletNodes = boundaryNodes


   
    # Applying concentration condition
    start_concentrationBC_time = time()
    concentrationLHS0 = (sps.lil_matrix.copy(M)/dt) + (1.0/(Re*Sc))*sps.lil_matrix.copy(Kxx) + (1.0/(Re*Sc))*sps.lil_matrix.copy(Kyy)
    concentrationBC = benchmarkProblems.linearStent(numPhysical,numNodes,x,y)
    concentrationBC.concentrationCondition(boundaryEdges,concentrationLHS0,neighborsNodes)
    end_concentrationBC_time = time()
    concentrationBC_time = end_concentrationBC_time - start_concentrationBC_time
    print ' concentration BC: %.1f seconds' %concentrationBC_time



   
  
   # Quad Element
   elif polynomial_option == 3:
 
    # Applying vx condition
    start_xVelocityBC_time = time()
    xVelocityLHS0 = sps.lil_matrix.copy(M)
    xVelocityBC = benchmarkProblems.quadStent(numPhysical,numNodes,x,y)
    xVelocityBC.xVelocityCondition(boundaryEdges,xVelocityLHS0,neighborsNodes)
    benchmark_problem = xVelocityBC.benchmark_problem
    end_xVelocityBC_time = time()
    xVelocityBC_time = end_xVelocityBC_time - start_xVelocityBC_time
    print ' xVelocity BC: %.1f seconds' %xVelocityBC_time

   


    # Applying vy condition
    start_yVelocityBC_time = time()
    yVelocityLHS0 = sps.lil_matrix.copy(M)
    yVelocityBC = benchmarkProblems.quadStent(numPhysical,numNodes,x,y)
    yVelocityBC.yVelocityCondition(boundaryEdges,yVelocityLHS0,neighborsNodes)
    end_yVelocityBC_time = time()
    yVelocityBC_time = end_yVelocityBC_time - start_yVelocityBC_time
    print ' yVelocity BC: %.1f seconds' %yVelocityBC_time



    
    # Applying psi condition
    start_streamfunctionBC_time = time()
    streamFunctionLHS0 = sps.lil_matrix.copy(Kxx) + sps.lil_matrix.copy(Kyy)
    streamFunctionBC = benchmarkProblems.quadStent(numPhysical,numNodes,x,y)
    streamFunctionBC.streamFunctionCondition(boundaryEdges,streamFunctionLHS0,neighborsNodes)
    end_streamfunctionBC_time = time()
    streamfunctionBC_time = end_streamfunctionBC_time - start_streamfunctionBC_time
    print ' streamfunction BC: %.1f seconds' %streamfunctionBC_time

   
    # Applying vorticity condition
    vorticityDirichletNodes = boundaryNodes


    # Applying concentration condition
    start_concentrationBC_time = time()
    concentrationLHS0 = (sps.lil_matrix.copy(M)/dt) + (1.0/(Re*Sc))*sps.lil_matrix.copy(Kxx) + (1.0/(Re*Sc))*sps.lil_matrix.copy(Kyy)
    concentrationBC = benchmarkProblems.quadStent(numPhysical,numNodes,x,y)
    concentrationBC.concentrationCondition(boundaryEdges,concentrationLHS0,neighborsNodes)
    end_concentrationBC_time = time()
    concentrationBC_time = end_concentrationBC_time - start_concentrationBC_time
    print ' concentration BC: %.1f seconds' %concentrationBC_time
   # ---------------------------------------------------------------------------------
   
   
 



  # ------------------------ semi-Lagrangian Method --------------------------------
  if scheme_option == 2:
   print ""
   print ' -----------------------'
   print ' SEMI-LAGRANGIAN METHOD:'
   print ' -----------------------'
   start_SL_time = time()

   # Linear Element   
   if polynomial_option == 0 or polynomial_option == 1:
    w_d, c_d = semiLagrangian.Linear2D(numNodes, neighborsElements, IEN, x, y, vxSL, vySL, dt, w, c)

   # Mini Element   
   elif polynomial_option == 2:
    w_d = semiLagrangian.Mini2D(numNodes, neighborsElements, IEN, z, r, vz, vr, dt, w)
 
   # Quad Element   
   elif polynomial_option == 3:
    w_d, c_d = semiLagrangian.Quad2D(numNodes, neighborsElements, IEN, x, y, vxSL, vySL, dt, w, c)
 
   end_SL_time = time()
   SL_time = end_SL_time - start_SL_time
   print ' time duration: %.1f seconds' %SL_time
  #----------------------------------------------------------------------------------






  # ------------------------ SOLVE LINEAR EQUATIONS ----------------------------------
  print ""
  print ' ----------------------------'
  print ' SOLVE THE LINEARS EQUATIONS:'
  print ' ----------------------------'

 

 
  #---------- Step 2 - Compute the boundary conditions for vorticity --------------
  start_vorticityBC_time = time()
  vorticityRHS = sps.lil_matrix.dot(Gx,vy) - sps.lil_matrix.dot(Gy,vx)
  vorticityLHS = sps.lil_matrix.copy(M)
  vorticityAux1BC = scipy.sparse.linalg.cg(vorticityLHS,vorticityRHS,vorticityAux1BC, maxiter=1.0e+05, tol=1.0e-05)
  vorticityAux1BC = vorticityAux1BC[0].reshape((len(vorticityAux1BC[0]),1))


  if polynomial_option == 0 or polynomial_option == 1: #Linear
   for i in range(0,len(boundaryEdges)):
    line = boundaryEdges[i][0]
    v1 = boundaryEdges[i][1] - 1
    v2 = boundaryEdges[i][2] - 1

    if line == 4: #vorticity null forced
     vorticityAux1BC[v1] = 0.0
     vorticityAux1BC[v2] = 0.0
 

  elif polynomial_option == 3: #Quad
   for i in range(0,len(boundaryEdges)):
    line = boundaryEdges[i][0]
    v1 = boundaryEdges[i][1] - 1
    v2 = boundaryEdges[i][2] - 1
    v3 = boundaryEdges[i][3] - 1

    if line == 4: #vorticity null forced
     vorticityAux1BC[v1] = 0.0
     vorticityAux1BC[v2] = 0.0
     vorticityAux1BC[v3] = 0.0
 
 
  # Gaussian elimination
  vorticityDirichletVector = np.zeros([numNodes,1], dtype = float)
  vorticityNeumannVector = np.zeros([numNodes,1], dtype = float)
  vorticityAux2BC = np.ones([numNodes,1], dtype = float)
 
  vorticityLHS = (np.copy(M)/dt) + (1.0/Re)*np.copy(Kxx) + (1.0/Re)*np.copy(Kyy)
  for mm in vorticityDirichletNodes:
   for nn in neighborsNodes[mm]:
    vorticityDirichletVector[nn] -= float(vorticityLHS[nn,mm]*vorticityAux1BC[mm])
    vorticityLHS[nn,mm] = 0.0
    vorticityLHS[mm,nn] = 0.0
    
   vorticityLHS[mm,mm] = 1.0
   vorticityDirichletVector[mm] = vorticityAux1BC[mm]
   vorticityAux2BC[mm] = 0.0

  end_vorticityBC_time = time()
  vorticityBC_time = end_vorticityBC_time - start_vorticityBC_time
  print ' Vorticity BC: %.1f seconds' %vorticityBC_time
  #----------------------------------------------------------------------------------
 
 
 
  #---------- Step 3 - Solve the vorticity transport equation ----------------------
  start_vorticitysolver_time = time()
  # Taylor Galerkin Scheme
  if scheme_option == 1:
   A = np.copy(M)/dt 
   vorticityRHS = sps.lil_matrix.dot(A,w) - np.multiply(vx,sps.lil_matrix.dot(Gx,w))\
         - np.multiply(vy,sps.lil_matrix.dot(Gy,w))\
         - (dt/2.0)*np.multiply(vx,(np.multiply(vx,sps.lil_matrix.dot(Kxx,w)) + np.multiply(vy,sps.lil_matrix.dot(Kyx,w))))\
         - (dt/2.0)*np.multiply(vy,(np.multiply(vx,sps.lil_matrix.dot(Kxy,w)) + np.multiply(vy,sps.lil_matrix.dot(Kyy,w))))
   vorticityRHS = np.multiply(vorticityRHS,vorticityAux2BC)
   vorticityRHS = vorticityRHS + vorticityDirichletVector
   w = scipy.sparse.linalg.cg(vorticityLHS,vorticityRHS,w, maxiter=1.0e+05, tol=1.0e-05)
   w = w[0].reshape((len(w[0]),1))
 

  # semi-Lagrangian Scheme
  elif scheme_option == 2:
   A = np.copy(M)/dt
   vorticityRHS = sps.lil_matrix.dot(A,w_d)
 
   vorticityRHS = vorticityRHS + (1.0/Re)*vorticityNeumannVector
   vorticityRHS = np.multiply(vorticityRHS,vorticityAux2BC)
   vorticityRHS = vorticityRHS + vorticityDirichletVector
 
   w = scipy.sparse.linalg.cg(vorticityLHS,vorticityRHS,w, maxiter=1.0e+05, tol=1.0e-05)
   w = w[0].reshape((len(w[0]),1))
 
  end_vorticitysolver_time = time()
  vorticitysolver_time = end_vorticitysolver_time - start_vorticitysolver_time
  print ' Vorticity Solver: %.1f seconds' %vorticitysolver_time
  #----------------------------------------------------------------------------------
 
 
 
  #---------- Step 4 - Solve the streamline equation --------------------------------
  # Solve Streamline
  # psi condition
  start_streamfunctionsolver_time = time()

  streamFunctionRHS = sps.lil_matrix.dot(M,w)
  streamFunctionRHS = np.multiply(streamFunctionRHS,streamFunctionBC.aux2BC)
  streamFunctionRHS = streamFunctionRHS + streamFunctionBC.dirichletVector
  psi = scipy.sparse.linalg.cg(streamFunctionBC.LHS,streamFunctionRHS,psi, maxiter=1.0e+05, tol=1.0e-05)
  psi = psi[0].reshape((len(psi[0]),1))

  end_streamfunctionsolver_time = time()
  streamfunctionsolver_time = end_streamfunctionsolver_time - start_streamfunctionsolver_time
  print ' Streamfunction Solver: %.1f seconds' %streamfunctionsolver_time
  #----------------------------------------------------------------------------------
 
 
 
  #---------- Step 5 - Compute the velocity field -----------------------------------
  start_velocitysolver_time = time()

  # Velocity vx
  vx_old = np.copy(vx)
  xVelocityRHS = sps.lil_matrix.dot(Gy,psi)
  xVelocityRHS = np.multiply(xVelocityRHS,xVelocityBC.aux2BC)
  xVelocityRHS = xVelocityRHS + xVelocityBC.dirichletVector
  vx = scipy.sparse.linalg.cg(xVelocityBC.LHS,xVelocityRHS,vx, maxiter=1.0e+05, tol=1.0e-05)
  vx = vx[0].reshape((len(vx[0]),1))
  
  # Velocity vy
  vy_old = np.copy(vy)
  yVelocityRHS = -sps.lil_matrix.dot(Gx,psi)
  yVelocityRHS = np.multiply(yVelocityRHS,yVelocityBC.aux2BC)
  yVelocityRHS = yVelocityRHS + yVelocityBC.dirichletVector
  vy = scipy.sparse.linalg.cg(yVelocityBC.LHS,yVelocityRHS,vy, maxiter=1.0e+05, tol=1.0e-05)
  vy = vy[0].reshape((len(vy[0]),1))

  end_velocitysolver_time = time()
  velocitysolver_time = end_velocitysolver_time - start_velocitysolver_time
  print ' Velocity Solver: %.1f seconds' %velocitysolver_time
  #----------------------------------------------------------------------------------
 


  

  #---------- Step 7 - Solve the specie transport equation ----------------------
  start_concentrationsolver_time = time()

  c_old = np.copy(c)
  # Taylor Galerkin Scheme
  if scheme_option == 1:
   A = np.copy(M)/dt 
   concentrationRHS = sps.lil_matrix.dot(A,c) - np.multiply(vx,sps.lil_matrix.dot(Gx,c))\
         - np.multiply(vy,sps.lil_matrix.dot(Gy,c))\
         - (dt/2.0)*np.multiply(vx,(np.multiply(vx,sps.lil_matrix.dot(Kxx,c)) + np.multiply(vy,sps.lil_matrix.dot(Kyx,c))))\
         - (dt/2.0)*np.multiply(vy,(np.multiply(vx,sps.lil_matrix.dot(Kxy,c)) + np.multiply(vy,sps.lil_matrix.dot(Kyy,c))))
   concentrationRHS = np.multiply(concentrationRHS,concentrationBC.aux2BC)
   concentrationRHS = concentrationRHS + concentrationBC.dirichletVector
   c = scipy.sparse.linalg.cg(concentrationBC.LHS,concentrationRHS,c, maxiter=1.0e+05, tol=1.0e-05)
   c = c[0].reshape((len(c[0]),1))
 
 
 
  # Semi-Lagrangian Scheme
  elif scheme_option == 2:
   A = np.copy(M)/dt
   concentrationRHS = sps.lil_matrix.dot(A,c_d)
 
   concentrationRHS = np.multiply(concentrationRHS,concentrationBC.aux2BC)
   concentrationRHS = concentrationRHS + concentrationBC.dirichletVector
 
   c = scipy.sparse.linalg.cg(concentrationBC.LHS,concentrationRHS, c, maxiter=1.0e+05, tol=1.0e-05)
   c = c[0].reshape((len(c[0]),1))

  end_concentrationsolver_time = time()
  concentrationsolver_time = end_concentrationsolver_time - start_concentrationsolver_time
  print ' Concentration Solver: %.1f seconds' %concentrationsolver_time
  #----------------------------------------------------------------------------------
 

 
 
 
 

  # ------------------------ Export VTK File ---------------------------------------
  print ""
  print ' ----------------'
  print ' EXPORT VTK FILE:'
  print ' ----------------'
  print ' Saving simulation in %s' %folderResults
  start_exportVTK_time = time()

  # Linear and Mini Elements
  if polynomial_option == 0 or polynomial_option == 1 or polynomial_option == 2:   
   save = exportVTK.Linear2D(x,y,IEN,numNodes,numElements,w,psi,c,vx,vy)
   save.create_dir(folderResults)
   save.saveVTK(folderResults + str(t))
 
  # Quad Element
  elif polynomial_option == 3:   
   save = exportVTK.Quad2D(x,y,IEN,numNodes,numElements,w,psi,c,vx,vy)
   save.create_dir(folderResults)
   save.saveVTK(folderResults + str(t))

  end_exportVTK_time = time()
  exportVTK_time = end_exportVTK_time - start_exportVTK_time
  print ' time duration: %.1f seconds' %exportVTK_time
  #----------------------------------------------------------------------------------
 




 
 
  # ---------------------------------------------------------------------------------
  print ""
  print ' -------'
  print ' CHECKS:'
  print ' -------'
  start_checks_time = time()
 
  # CHECK STEADY STATE
  #if np.all(vx == vx_old) and np.all(vy == vy_old):
  # end_type = 1
  # break
 
  # CHECK CONVERGENCE OF THE SOLUTION
  if np.linalg.norm(vx) > 10e2 or np.linalg.norm(vy) > 10e2:
   end_type = 2
   break

  end_checks_time = time()
  checks_time = end_checks_time - start_checks_time
  print ' time duration: %.1f seconds' %checks_time
  # ---------------------------------------------------------------------------------
 
  print "" 
  print "" 
  print " ---------------------------------------------------------------------------------"
  


 except KeyboardInterrupt:
  end_type = 3
  break 




end_time = time()
solution_time = end_time - solution_start_time


print ""
print ' ----------------'
print ' SAVING RELATORY:'
print ' ----------------'

if end_type == 0:
 print ' END SIMULATION. NOT STEADY STATE'
 print ' Relatory saved in %s' %folderResults
 print ""

elif end_type == 1:
 print ' END SIMULATION. STEADY STATE'
 print ' Relatory saved in %s' %folderResults
 print ""

elif end_type == 2:
 print ' END SIMULATION. ERROR CONVERGENCE RESULT'
 print ' Relatory saved in %s' %folderResults
 print ""

elif end_type == 3:
 print ' END SIMULATION. FORCED INTERRUPTION'
 print ' Relatory saved in %s' %folderResults
 print ""




# -------------------------------- Export Relatory ---------------------------------------
relatory.export(save.path, folderResults, sys.argv[0], benchmark_problem, scheme_name, mshFileName, numNodes, numElements, minLengthMesh, dt, numIteration, Re, Sc, import_mesh_time, assembly_time, bc_apply_time, solution_time, polynomial_order, gausspoints, observation)
# ----------------------------------------------------------------------------------------



