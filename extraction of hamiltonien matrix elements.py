
import re
import numpy as np
def extract_matrices_from_file(file_path):
    matrices = []

    with open(file_path, 'r') as file:
        content = file.read()

        # Use regex to find all occurrences of text within {}
        matches = re.findall(r'\{(.*?)\}', content)

        for match in matches:
            # Check if match is empty (i.e., `{}`)
            if not match.strip():
                matrices.append(['-1+0'])
            else:
                # Split the string by commas and strip whitespace, storing as list of strings
                matrix_elements = [element.strip() for element in match.split(',')]
                matrices.append(matrix_elements)

    return matrices


def remove_curly_braces(lst):
    # Loop through each string in the list and remove '{' and '}' characters
    return [s.replace('{', '').replace('}', '') for s in lst]
def replace_V_with_numbers(m,v):
    l=['v[0]','v[1]','v[2]','v[3]','v[4]','v[5]','v[6]','v[7]','v[8]','v[9]']
    for i in range(len(v)):
      m= [s.replace(l[i],v[i]) for s in m] 
    return m
def change_Sqrt(m): 
    return [s.replace('Sqrt','np.sqrt').replace('[','(').replace(']',')') for s in m]

def hamiltonian(v,file_path,nn,p):   
 v=[str(f) for f in v]

# Extract matrices from the file and print them
 extracted_matrices = extract_matrices_from_file(file_path)
 stored_matrices=[]
 for i in range(len(extracted_matrices)):
  m=extracted_matrices[i]
  m=remove_curly_braces(m)
  m=replace_V_with_numbers(m,v) 
  m= change_Sqrt(m)
  m= [eval(expr) for expr in m]
  if len(m)!=1:
    
   matrix = [[0 for _ in range(int(np.sqrt(len(m))))] for _ in range(int(np.sqrt(len(m))))]
   index = 0
   dimension= int(np.sqrt(len(m)))
   for k in range(dimension):
    for j in range(dimension):
      matrix[k][j] = m[index]
      index += 1
  else:
     matrix= m[0] 
  stored_matrices.append(matrix)
  if (nn+p)%2==0:
   labeled_dict = {f'J[{i}]': value for i, value in enumerate(stored_matrices)}
  else:
   labeled_dict = {f'J[{i + 1/2}]': value for i, value in enumerate(stored_matrices)} 
  
 return labeled_dict

# Example usage:
print(hamiltonian([2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'mat1004.txt', 10, 4))
