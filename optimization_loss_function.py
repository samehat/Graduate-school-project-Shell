
from scipy.optimize import minimize
import numpy as np
import random
import time 
from extraction_of_hamiltonien_matrix_elements import calculate_loss_function

# Start the timer
start_time = time.time()

# Initialize a global variable to store the last Ecal_used
last_Ecal_used = None
last_Ecal_used_not_excited=None

def root_mean_square_loss_function(filename, p, n, v):
    global last_Ecal_used
    global last_Ecal_used_not_excited
    s, N, Ecal_used,Ecal_used_not_excited = calculate_loss_function(filename, p, n, v)
    
    # Store the last Ecal_used globally for retrieval after optimization
    last_Ecal_used = Ecal_used
    last_Ecal_used_not_excited= Ecal_used_not_excited
    return np.sqrt((1 / N) * s)

def optimize_using_scipy(filename, p, n, v_initial):
    # Fix the first parameter to 0
    fixed_v = np.zeros_like(v_initial)
    
    def objective_function(v):
        # Combine fixed first parameter with the variable parameters
        combined_v = np.concatenate(([0], v))
        return root_mean_square_loss_function(filename, p, n, combined_v)

    # Initial guess for the rest of the parameters (excluding the fixed first one)
    initial_guess = v_initial[1:]  # Exclude the first parameter
    print("initial parameter  values: ",initial_guess)

    # Define bounds for the parameters (0 to 2000 for each parameter, except the fixed first one)
    bounds = [(0, 10000)] * (len(v_initial) - 1)  # Exclude the fixed first parameter

    # Run the optimizer with bounds
    result = minimize(objective_function, initial_guess, method='L-BFGS-B', bounds=bounds)

    # Combine the optimized parameters with the fixed first parameter
    optimized_v = np.concatenate(([0], result.x))  # Include the fixed first parameter
    print(f"Optimized Parameters: {optimized_v}")
    print(f"Minimum Loss Reached: {result.fun}")
    print(f"Last Ecal Used After Minimization: {last_Ecal_used}")
    print(f"Last Ecal not excited Used After Minimization: {last_Ecal_used_not_excited} ")
    
    return optimized_v, result.fun, last_Ecal_used , last_Ecal_used_not_excited

# Example usage (assuming root_mean_square_loss_function is defined and v_initial is provided)
p = ['0', '1', '2', '3', '4','5','6','7','8','9','10']
n = ['9']
            
file_dict = {}
filename = ['mat1000.txt', 'mat1001.txt', 'mat1002.txt', 'mat1003.txt', 'mat1004.txt','mat1005.txt', 'mat1006.txt', 'mat1007.txt', 'mat1008.txt', 'mat1009.txt','mat1010.txt',
            'mat0900.txt', 'mat0901.txt', 'mat0902.txt', 'mat0903.txt', 'mat0904.txt','mat0905.txt', 'mat0906.txt', 'mat0907.txt', 'mat0908.txt', 'mat0909.txt','mat0910.txt',
            'mat0800.txt', 'mat0801.txt', 'mat0802.txt', 'mat0803.txt', 'mat0804.txt','mat0805.txt', 'mat0806.txt', 'mat0807.txt', 'mat0808.txt', 'mat0809.txt','mat0810.txt']
for fname in filename:
    key = str(int(fname[3:5]))  # Extract two digits after 'mat' and remove any leading zero
    file_dict.setdefault(key, []).append(fname)

v_initial = [0] + [random.uniform(0, 10000) for _ in range(9)]  # Set first element to zero
print("initial parameters : ",v_initial)
optimized_params, min_loss, Ecal,Ecal_not_excited = optimize_using_scipy(file_dict, p, n, v_initial)

# End the timer and print the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
