
from scipy.optimize import minimize
import numpy as np
import random
import time 
print("a")
from extraction_v2 import diagonalisation , data_exp_form , mask_state , initalization


filepath = "data\data3.csv"
print("begining")
# Initialize a global variable to store the last Ecal_used
last_Ecal_used = None
last_Ecal_used_not_excited=None

data = data_exp_form(filepath)
data = mask_state(data)

test_list=[[10,9],[10,8],[10,7],[10,6],[10,5],[10,3],[10,4],[10,2],
[9,9],[9,8],[9,7],[9,6],[9,5],[9,3],[9,4],[9,2],[9,1],
[8,7],[8,6],[8,5],[8,4],[8,3],[8,2],[8,1],[8,0]]

dict_th = initalization(data)
print("step1")
# Start the timer
start_time = time.time()



def optimize_using_scipy(n_p_list,v_init):
    # Fix the first parameter to 0
    
    list = n_p_list
    print(list)
    
    def objective_function(v):
        # Combine fixed first parameter with the variable parameters
        combined_v = np.concatenate(([0], v))
        return diagonalisation(list, combined_v)[0]


    # Run the optimizer with bounds
    result = minimize(objective_function, v_init, method='L-BFGS-B')#bounds=bounds

    # Combine the optimized parameters with the fixed first parameter
    optimized_v = np.concatenate(([0], result.x))  # Include the fixed first parameter
    print(f"Optimized Parameters: {result.x}")
    print(f"Minimum Loss Reached: {result.fun}")

        # Calculate uncertainties and correlations
    if result.hess_inv is not None:
        covariance_matrix = result.hess_inv.todense()  # Convert to dense matrix if sparse
        uncertainties = np.sqrt(np.diag(covariance_matrix))  # Standard deviations (uncertainties)
        correlations = covariance_matrix / np.outer(uncertainties, uncertainties)  # Correlation matrix

        print(f"Uncertainties: {uncertainties}")
        print(f"Correlation Matrix:\n{correlations}")
    else:
        print("Hessian not available. Cannot calculate uncertainties and correlations.")
    
    return optimized_v, result.fun, last_Ecal_used , last_Ecal_used_not_excited

# Example usage (assuming root_mean_square_loss_function is defined and v_initial is provided)
fit_list2=[[10,8],[10,7],[10,6],[10,5],[10,4],[10,3],[10,2],[10,1],
[9,9],[9,7],[9,6],[9,5],[9,4],[9,3],[9,2],[9,1],
[8,7],[8,6],[8,5],[8,4],[8,3],[8,2],[8,1],[8,0]]
            
#v_9_10 = np.array([1200,1400,1600,2200,2000,2600,2000,2600,800])

v_8 = [1713.8091925,  1484.48172167, 1929.37515306, 2359.68310125, 2039.11355149,
2528.85057569, 2260.29128492, 2653.27379747,  816.88134078]

v_initial = np.array([random.uniform(-500, 500) for _ in range(9)])  # Set first element to zero
print("initial parameters : ",v_initial)
v_test = np.add(v_8, v_initial)
#v_test = [1175.64646937, 1478.22919049, 1650.00380408, 2242.33845357, 1967.66590463,
#2451.55156364, 1950.48169652, 2550.60485793,  828.50555058]
print(v_test)

optimized_params, min_loss, Ecal,Ecal_not_excited = optimize_using_scipy(fit_list2,v_test)

# End the timer and print the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
v_fin=[0,1714.21,1486.04,1926.65,2355.08,2052.56,2524.65,2249.85,2657.33,820]

