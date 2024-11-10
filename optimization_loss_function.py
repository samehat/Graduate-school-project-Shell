
import numpy as np
import random
import time 
from extraction_of_hamiltonien_matrix_elements import root_mean_square_loss_function
start_time = time.time()
class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(grads)
            self.v = np.zeros_like(grads)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        params -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return params

def optimize_loss_function(filename, p, n, v_initial, num_iterations=1000):
    optimizer = AdamOptimizer(learning_rate=0.01)
    v = np.array(v_initial, dtype=float)  # Convert to a NumPy array if not already

    for iteration in range(num_iterations):
        loss = root_mean_square_loss_function(filename, p, n, v)
        
        # Calculate gradients (this part may need custom implementation)
        # Placeholder: You'll need to compute the gradient of your loss function.
        # For example, using finite differences:
        grads = np.zeros_like(v)
        for i in range(len(v)):
            v_eps = np.copy(v)
            v_eps[i] += 1e-5
            loss_plus = root_mean_square_loss_function(filename, p, n, v_eps)
            grads[i] = (loss_plus - loss) / 1e-5

        # Update parameters using Adam
        v = optimizer.update(v, grads)

        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Loss: {loss}")

    return v
from scipy.optimize import minimize

from scipy.optimize import minimize
import numpy as np

def optimize_using_scipy(filename, p, n, v_initial):
    # Fix the first parameter to 0
    fixed_v = np.zeros_like(v_initial)
    
    def objective_function(v):
        # Combine fixed first parameter with the variable parameters
        combined_v = np.concatenate(([0], v))
        return root_mean_square_loss_function(filename, p, n, combined_v)

    # Initial guess for the rest of the parameters (excluding the fixed first one)
    initial_guess = v_initial[1:]  # Exclude the first parameter

    # Define bounds for the parameters (0 to 2000 for each parameter, except the fixed first one)
    bounds = [(0, 2000)] * (len(v_initial) - 1)  # Exclude the fixed first parameter

    # Run the optimizer with bounds
    result = minimize(objective_function, initial_guess, method='L-BFGS-B', bounds=bounds)

    # Combine the optimized parameters with the fixed first parameter
    optimized_v = np.concatenate(([0], result.x))  # Include the fixed first parameter
    print(f"Optimized Parameters: {optimized_v}")
    print(f"Minimum Loss Reached: {result.fun}")
    
    return optimized_v, result.fun

# Example usage (assuming root_mean_square_loss_function is defined and v_initial is provided)
# optimized_params, min_loss = optimize_using_scipy("your_file.txt", p, n, v_initial)
# Example usage
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
v_initial = [0] + [random.uniform(0, 2000) for _ in range(9)]  # Set first element to zero
optimize_using_scipy(file_dict, p, n, v_initial)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
