
from scipy.optimize import minimize
import numpy as np
import random
import time 
from extraction_v2 import diagonalisation,data_exp_form,mask_state,initalization


filepath = "data\data2.csv"

# Initialize a global variable to store the last Ecal_used
last_Ecal_used = None
last_Ecal_used_not_excited=None

data = data_exp_form(filepath)
data = mask_state(data)

dict_th = initalization(data)

# Start the timer
start_time = time.time()



def optimize_using_scipy(n_p_list, v_initial):
    # Fix the first parameter to 0
    fixed_v = np.zeros_like(v_initial)
    
    def objective_function(v):
        # Combine fixed first parameter with the variable parameters
        combined_v = np.concatenate(([0], v))
        return diagonalisation(n_p_list, combined_v)

    # Initial guess for the rest of the parameters (excluding the fixed first one)
    initial_guess = [1714,1487,1925,2352,2053,2528,2247,2656,820]    #v_initial[1:]  # Exclude the first parameter
    print("initial parameter  values: ",initial_guess)

    # Define bounds for the parameters (0 to 2000 for each parameter, except the fixed first one)
    #bounds = [(0, 10000)] * (len(v_initial) - 1)  # Exclude the fixed first parameter

    # Run the optimizer with bounds
    result = minimize(objective_function, initial_guess, method='L-BFGS-B')#bounds=bounds

    # Combine the optimized parameters with the fixed first parameter
    optimized_v = np.concatenate(([0], result.x))  # Include the fixed first parameter
    print(f"Optimized Parameters: {result.x}")
    print(f"Minimum Loss Reached: {result.fun}")
    print(f"Last Ecal Used After Minimization: {last_Ecal_used}")
    print(f"Last Ecal not excited Used After Minimization: {last_Ecal_used_not_excited} ")
    
    return optimized_v, result.fun, last_Ecal_used , last_Ecal_used_not_excited

# Example usage (assuming root_mean_square_loss_function is defined and v_initial is provided)
fit_list=[[10,8],[10,7],[10,6],[10,5],[10,4],[10,3],[10,2],[10,1],[9,9],[9,7],[9,6],[9,5],[9,4],[9,3],[9,2],[9,1]]
            


v_initial = [0] + [random.uniform(0, 3500) for _ in range(9)]  # Set first element to zero
print("initial parameters : ",v_initial)
optimized_params, min_loss, Ecal,Ecal_not_excited = optimize_using_scipy(fit_list, v_initial)

# End the timer and print the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
v_fin=[0,1714.21,1486.04,1926.65,2355.08,2052.56,2524.65,2249.85,2657.33,820]
# 9,10 : [   0. 2137.30166028 1459.64035162 2545.49266986 2223.59395654 2902.55528279 2428.40201565 2940.18077546 2563.63134406 1495.19825606]
# 10 : [   0. 1506.38979531 1446.93184139 2021.95694999 2197.61882959 337.58333812 2409.98423534 3118.39960499 2525.38827848 1751.16590996]
# 9,10 : [   0. 955.56118732 1478.44817646 1460.0976269  2234.29215947 1767.65553973 2430.25964712 1695.19572299 2547.22572698  757.26704366]
print((1019915/220))
list1 = np.array([1,2,3,4,5])
list = np.array([2,4,4,7,6])
print((list1-list)**2)