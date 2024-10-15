# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 20:35:56 2024

@author: marah
"""

import numpy as np
import random
def function(m,V,np,nn,J):
 f=0
    
 for i in range(len(m)):
     state=0
     if m[i][0]==np and m[i][1]==nn and m[i][2]==J  :
      for j in range(10):
          index=5
          state+=m[i][index]*V[j]# this will multiply the b values with our V values 
          index+=index
      f+=(m[i][3]-state)**2    #the state will contain the sum of the bxr  and this m[i][0] will be the value of Eexp
 return f
def diagonalize_matrix(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    
    # Print results
    print("Eigenvalues:")
    print(eigenvalues)
    print("\nEigenvectors:")
    print(eigenvectors)
    
    return eigenvalues, eigenvectors

def simulated_annealing(energy_function,V, initial_temp, cooling_rate, min_temp, max_iterations,m,np,nn,J):
    temp = initial_temp
    V=V
    j=0
    f= function(m,V,np,nn,J)
    

    for k in range(max_iterations):
        if temp < min_temp:
            break
        new_V=[] # new list containing that will contain the new values of V
        # Generate a new state by a small random change to the current state
        for j in range(len(V)):
         new_V.append(V[j]+np.random.uniform(-1, 1))
        new_f=function(m,new_V,np,nn,J)
        delta_f = new_f - f

        # Accept new state if it improves energy or by probability
        if delta_f < 0 or np.random.rand() < np.exp(-delta_f / temp):
            f= new_f
            for i in range(len(V)):
                V[i]=new_V[i]

        # Decrease temperature
        temp *= cooling_rate

    return V
def b_values_calculator(eigenvectors,matrix):
    transpose_matrix=np.transpose(matrix)
    

m=[]# this will be a list with a list inside of it looking like : np,nn,J,Eexp,Ecal,b0,b1.........,b9

initial_state = []     # this will be the initial guess of the parameters 
for i in range(10):
    initial_state.append(random.uniform(0,5000))

