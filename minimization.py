import numpy as np
import random
from extraction_of_hamiltonien_matrix_elements import calculate_loss_function
from extraction_of_experimental_energies import experimental_values
from extraction_of_hamiltonien_matrix_elements import root_mean_square_loss_function
def function(m,np,nn,J,eigenvalues):
 f=0   
 for i in range(len(eigenvalues)):
     if m[i][0]==np and m[i][1]==nn and m[i][2]==J :
      f+=(m[i][3]-eigenvalues)**2    #the state will contain the sum of the bxr  and this m[i][0] will be the value of Eexp
 return f

# the monte carlo method to compute the minimum value of the 
def simulated_annealing(initial_temp, cooling_rate, min_temp, max_iterations,p,n,filename):
    temp = initial_temp
    filename=['mat1000.txt','mat1001.txt','mat1002.txt','mat1003.txt','mat1004.txt','mat1004.txt','mat1003.txt','mat1002.txt']   
    
    v=[]
    v.append(0) # energies will be relative to this energy 
    for i in range(9):
      v.append(random.uniform(0,2000))
    f= root_mean_square_loss_function(filename,p,n,v)  
    print(f)
    for k in range(max_iterations):
        if temp < min_temp:
            break
        new_v=[]
        new_v.append(0) # energies will be relative to this energy 
        for i in range(9):
          new_v.append(random.uniform(0,2000))
        new_f= root_mean_square_loss_function(filename,p,n,new_v) 
        delta_f = new_f - f

        # Accept new state if it improves energy or by probability
        if delta_f < 0 or np.random.rand() < np.exp(-delta_f / temp):
            f= new_f
            v=new_v

        # Decrease temperature
        temp *= cooling_rate

    return v,f

# Suggested parameters
initial_temp = 500       # Lower initial temperature for more focused search
cooling_rate = 0.9      # Faster cooling rate
min_temp = 1e-4          # Minimum temperature to stop the loop earlier if convergence is found
max_iterations = 5000000
p=['0','1','2','3','4','6','7','8']
n='10'
filename=['mat1000.txt','mat1001.txt','mat1002.txt','mat1003.txt','mat1004.txt','mat1004.txt','mat1003.txt','mat1002.txt']   
# Running the simulated annealing algorithm
result_v, result_f = simulated_annealing(initial_temp, cooling_rate, min_temp, max_iterations,p,n,filename)

# Printing the results
print("Final variable values:", result_v)
print("Final loss function value:", result_f)
