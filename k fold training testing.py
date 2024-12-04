# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 20:39:08 2024

@author: marah
"""

import numpy as np
import random
from scipy.optimize import minimize
from sklearn.model_selection import KFold
import time
from extraction_of_hamiltonien_matrix_element_with_cross_validation import calculate_loss_function

# Initialize global variables
last_Ecal_used = None
last_Ecal_used_not_excited = None

def root_mean_square_loss_function(files, p, n, v):
    global last_Ecal_used
    global last_Ecal_used_not_excited
    
    s, N, Ecal_used, Ecal_used_not_excited = calculate_loss_function(files, p, n, v)
    
    last_Ecal_used = Ecal_used
    last_Ecal_used_not_excited = Ecal_used_not_excited
    return np.sqrt((1 / N) * s) if N != 0 else np.inf  # Handle division by zero

def optimize_using_scipy(files, p, n, v_initial):
    def objective_function(v):
        combined_v = np.concatenate(([0], v))  # Fix first parameter to 0
        return root_mean_square_loss_function(files, p, n, combined_v)

    initial_guess = v_initial[1:]
    bounds = [(0, 10000)] * (len(v_initial) - 1)

    result = minimize(objective_function, initial_guess, method='L-BFGS-B', bounds=bounds,
                      options={'ftol': 1e-6, 'maxiter': 1000})

    optimized_v = np.concatenate(([0], result.x))
    return optimized_v, result.fun

# K-fold cross-validation function
def cross_validate(files, p, n, k_folds=5):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    initial_v = [0] + np.random.uniform(0, 10000, 9).tolist()
    losses = []

    for train_index, test_index in kf.split(files):
        train_files = [files[i] for i in train_index]
        test_files = [files[i] for i in test_index]

        print(f"Training on: {train_files}")
        print(f"Testing on: {test_files}")
       
        test_file_dict={}
        for fname in test_files:
                key = str(int(fname[3:5]))  # Extract two digits after 'mat' and remove any leading zero
                key1 = str(int(fname[5:7]))
                test_file_dict.setdefault((key, key1), []).append(fname)
        print("test file dictionary: ", test_file_dict)
        # Extract p and n from filename
        train_file_dict={}
        for fname in train_files:
                 key = str(int(fname[3:5]))  # Extract two digits after 'mat' and remove any leading zero
                 key1 = str(int(fname[5:7]))
                 train_file_dict.setdefault((key, key1), []).append(fname)
        print("file dictionary: ", train_file_dict)
        optimized_v, loss = optimize_using_scipy(train_file_dict, p, n, initial_v)
        print("training loss : ", loss)
        test_loss = root_mean_square_loss_function(test_file_dict, p, n, optimized_v)
        losses.append(test_loss)
        print("the predicted Energies: ",last_Ecal_used)
        print("the optimized parameters: ", optimized_v)
        print(f"Validation Loss: {test_loss}")

    avg_loss = np.mean(losses)
    print(f"Average Cross-Validation Loss: {avg_loss}")
    return avg_loss

# Example usage
selected_files = [
    'mat1001.txt', 'mat1002.txt', 'mat1003.txt', 'mat1004.txt',
    'mat1005.txt', 'mat1006.txt', 'mat1007.txt', 'mat1008.txt',
    'mat0901.txt', 'mat0902.txt', 'mat0903.txt', 'mat0904.txt',
    'mat0905.txt', 'mat0906.txt', 'mat0908.txt', 'mat0909.txt'
]
p = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
n = ['9','10']
initial_v = [0] + np.random.uniform(0, 10000, 9).tolist()
# Start timer
cross_validate(selected_files, p, n, k_folds=5)
start_time = time.time()

end_time = time.time()

print(f"Total elapsed time: {end_time - start_time:.2f} seconds")
