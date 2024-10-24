# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:30:44 2024

@author: marah
"""
# this function will return the experimental values of the specific nn np and  J that you choose 
def find_all_values_in_column(file_name, col1_val, col2_val, col3_val):
    matching_values = []  # To store all matching values from column 4 as floats
    
    with open(file_name, 'r') as file:
        for line in file:
            # Assuming the file is space or tab-separated
            columns = line.split()  # Adjust split depending on your delimiter, e.g., .split(',')
            
            # Make sure the line has at least 4 columns to avoid index errors
            if len(columns) >= 4:
                # Compare values in columns 1, 2, and 3
                if columns[0] == str(col1_val) and columns[1] == str(col2_val) and columns[2] == str(col3_val):
                    try:
                        # Convert value from column 4 to float and append it to the list
                        matching_values.append(float(columns[3]))
                    except ValueError:
                        print(f"Could not convert {columns[3]} to float")
    
    return matching_values

# Example usage
file_name = 'Data.txt'
col1_val = '10'
col2_val = '4'
col3_val = '0'

values = find_all_values_in_column(file_name, col1_val, col2_val, col3_val)

if values:
    print(f"Values in column 4: {values}")
else:
    print("No matches found")
