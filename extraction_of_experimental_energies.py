# -*- coding: utf-8 -*-
def extract_J_values(file_path, target_nn, target_np):
    matching_J_values = []
    
    with open(file_path, 'r') as file:
        # Skip the header line
        next(file)
        
        for line in file:
            # Split the line into columns based on whitespace
            columns = line.strip().split()
            
            # Check if there are exactly three columns
            if len(columns) == 3:
                nn, np, J = columns
                
                # Check if nn and np match the target values
                if nn == target_nn and np == target_np:
                    matching_J_values.append(J)
    
    return matching_J_values

def extract_energies(file_path, target_nn, target_np, target_j):
    energy_dict = {}

    with open(file_path, 'r') as file:
        for line in file:
            columns = line.split()
            if len(columns) >= 4:
                nn, np, j, energy = columns[:4]
                extra_columns = columns[4:]
                
                # Check if nn, np, and j match the target values and there are no '1' in extra columns
                if int(nn) == target_nn and int(np) == target_np and j == target_j and '1' not in extra_columns:
                    label = (f"{nn}",f"{np}",f"{j}")
                    
                    # Add energy to the list for this label
                    if label not in energy_dict:
                        energy_dict[label] = []
                    energy_dict[label].append(energy)
                    
    return energy_dict


def experimental_values(file_name,n,p):
 J=extract_J_values('J values.txt',str(n),str(p))
 energies = extract_energies(file_name,n,p,J[0])
 for i in range(len(J)-1):
   
   # Initialize an empty list for this key if it does not exist
    energies.update(extract_energies(file_name,n,p,J[i+1]))
 return energies 
