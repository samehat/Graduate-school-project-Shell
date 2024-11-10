import numpy as np
import random
import time 
from extraction_of_hamiltonien_matrix_elements import calculate_loss_function
from extraction_of_experimental_energies import experimental_values
from extraction_of_hamiltonien_matrix_elements import root_mean_square_loss_function

start_time = time.time()

# the monte carlo method to compute the minimum value of the 
def simulated_annealing(initial_temp, cooling_rate, min_temp, max_iterations,p,n,filename):
    temp = initial_temp  
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
"""initial_temp = 500       # Lower initial temperature for more focused search
cooling_rate = 0.9      # Faster cooling rate
min_temp = 1e-4          # Minimum temperature to stop the loop earlier if convergence is found
max_iterations = 5000000"""
initial_temp = 800       # Higher initial temperature for broader exploration
cooling_rate = 0.95      # Slower cooling rate for more thorough exploration
min_temp = 1e-6          # Lower minimum temperature for finer convergence
max_iterations = 10000000  #
p=['0','1','2','3','4','6','7','8','9','10']
n=['8','9','10']
filename=['mat0800.txt','mat0801.txt','mat0802.txt','mat0803.txt','mat0804.txt','mat0804.txt','mat0803.txt','mat0802.txt','mat0801.txt','mat0800.txt','mat0900.txt','mat0901.txt','mat0902.txt','mat0903.txt','mat0904.txt','mat0904.txt','mat0903.txt','mat0902.txt','mat0901.txt','mat0900.txt','mat1000.txt','mat1001.txt','mat1002.txt','mat1003.txt','mat1004.txt','mat1004.txt','mat1003.txt','mat1002.txt','mat1001.txt','mat1000.txt']   
file_dict = {}
for fname in filename:
    # Extract the two digits after 'mat' and remove any leading zero
    key = str(int(fname[3:5]))
    
    # Append the filename to the list for this key
    if key in file_dict:
        file_dict[key].append(fname)
    else:
        file_dict[key] = [fname]
result_v, result_f = simulated_annealing(initial_temp, cooling_rate, min_temp, max_iterations,p,n,file_dict)


print("Final variable values:", result_v)
print("Final loss function value:", result_f)
end_time = time.time()
# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

