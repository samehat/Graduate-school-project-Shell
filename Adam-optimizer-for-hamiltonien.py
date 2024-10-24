# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 17:43:16 2024

@author: marah
"""

import numpy as np
import torch

# Step 1: Initialize random parameters V[0] to V[9]
lower_bound = -10  # Example lower bound
upper_bound = 10   # Example upper bound
V = np.random.uniform(lower_bound, upper_bound, 10)

# Step 2: Convert to torch tensor
V_tensor = torch.tensor(V, requires_grad=True)

# Step 3: Define Hamiltonian matrix function
def hamiltonian_matrix(V):
    H = torch.stack([
        torch.stack([V[0] + V[1], V[2], V[3], V[4]]),
        torch.stack([V[2], V[5] + V[6], V[7], V[8]]),
        torch.stack([V[3], V[7], V[4] + V[9], V[1]]),
        torch.stack([V[4], V[8], V[1], V[2] + V[0]])
    ])
    return H

# Step 4: Set up the optimizer with a lower learning rate
optimizer = torch.optim.Adam([V_tensor], lr=0.001)

# Step 5: Define experimental energies
E_exp = torch.tensor([1.0, 2.5, 3.0], dtype=torch.float32)

# Step 6: Optimization loop
for iteration in range(5000):  # Increase number of iterations
    optimizer.zero_grad()  # Clear gradients from previous step
    
    # Recalculate the Hamiltonian, Eigenvalues, and Loss Function
    H = hamiltonian_matrix(V_tensor)
    eigenvalues, _ = torch.linalg.eigh(H)  # Diagonalize H to get eigenvalues
    min_eigenvalue = torch.min(eigenvalues)  # Get the minimum (ground state) eigenvalue
    excitation_energies = eigenvalues - min_eigenvalue  # Calculate excitation energies

    # Calculate the loss (sum of squared differences between calculated and experimental energies)
    loss = torch.sum((excitation_energies[:len(E_exp)] - E_exp)**2)

    # Optionally, add a regularization term to avoid large values for V[0] to V[9]
    regularization_term = 0.01 * torch.sum(V_tensor**2)  # L2 regularization
    loss += regularization_term

    # Backpropagation and optimization step
    loss.backward()  # Compute gradients
    optimizer.step()  # Update V_tensor parameters
    
    # Print loss every 100 iterations
    if iteration % 100 == 0:
        print(f"Iteration {iteration}, Loss: {loss.item()}")

# Step 7: Print optimized parameters
print(f"Optimized V values: {V_tensor.detach().numpy()}")
