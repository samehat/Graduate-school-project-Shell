
import re
import numpy as np
import random
from extraction_of_experimental_energies import experimental_values
import torch
def extract_matrices_from_file(file_path):
    matrices = []

    with open(file_path, 'r') as file:
        content = file.read()

        # Use regex to find all occurrences of text within {{...}}
        matches = re.findall(r'\{\{(.*?)\}\}', content)

        for match in matches:
            # Split the string by commas and strip whitespace, storing as list of strings
            matrix_elements = [element.strip() for element in match.split(',')]
            matrices.append(matrix_elements)

    return matrices
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
def hamiltonian(v,n,p,filename):   #n and p should be strings but v not , the function convert it to string 
 v=[str(f) for f in v]
# Specify the path to your input file
 file_path = filename  # Replace with your actual file path

# Extract matrices from the file and print them
 extracted_matrices = extract_matrices_from_file(file_path)
 stored_matrices=[]
 for i in range(len(extracted_matrices)):
  m=extracted_matrices[i]
  m=remove_curly_braces(m)
  m=replace_V_with_numbers(m,v) 
  m= change_Sqrt(m)
  m= [eval(expr) for expr in m]
  dimension= int(np.sqrt(len(m)))
  if len(m)!=1:
    
   matrix = [[0 for _ in range(int(np.sqrt(len(m))))] for _ in range(int(np.sqrt(len(m))))]
   index = 0
 
   for k in range(dimension):
    for j in range(dimension):
      matrix[k][j] = m[index]
      index += 1
  else:
     matrix= m[0] 

  stored_matrices.append(matrix)
 J=extract_J_values('J values.txt',n,p)
 labeled_dict = {(n,p,J[i]): stored_matrices[i] for i in range(len(J))}


 return labeled_dict, J
def diagonalize_matrix(filename,p,n,v):# will be a list with the file names as elements 
  #  v=[]
   # v.append(0) # energies will be relative to this energy 
    #for i in range(9):
     #v.append(random.uniform(0,2000))
    Calculated_energies={}# a dictionary with the key (n,p,j)
    for z in range(len(n)):
     for j in range(len(p)):
            Hamiltonian, J=hamiltonian(v,n[z],p[j],filename[n[z]][j])
            # it will give me a dictionary labeled with (n,p,J)
            for k in range(len(J)):
             matrix=Hamiltonian[(n[z],p[j],J[k])]
             if isinstance(matrix, list) and all(isinstance(row, list) for row in matrix):
              eigenvalues, eigenvectors = np.linalg.eig(matrix)
              Calculated_energies[(n[z],p[j],J[k])]=eigenvalues
             else:
              Calculated_energies[(n[z],p[j],J[k])]=matrix 
         
    return Calculated_energies          
#filename=['mat1000.txt','mat1001.txt','mat1002.txt','mat1003.txt','mat1004.txt']   
def calculated_excitation_energies(filename,p,n,v):# n and p are lists 
 m=diagonalize_matrix(filename,p,n,v)
 for z in range(len(n)):   
  for l in range(len(p)):
   J=J=extract_J_values('J values.txt',n[z],p[l])  
   if isinstance(m[(n[z],p[l],J[0])],np.ndarray):
    minimum_energy= np.min(m[(n[z],p[l],J[0])])
   else:
      minimum_energy= m[(n[z],p[l],J[0])]
   for i in range(len(J)-1):
      if isinstance(m[(n[z],p[l],J[i+1])],np.ndarray):
       energy= np.min(m[(n[z],p[l],J[i+1])])
      else:
       energy= m[(n[z],p[l],J[i+1])]
      minimum_energy=min([energy,minimum_energy])
   for i in range(len(J)):    
    if isinstance(m[(n[z],p[l],J[i])],np.ndarray):
        for j in range(m[(n[z],p[l],J[i])].size):
            m[(n[z],p[l],J[i])][j]=m[(n[z],p[l],J[i])][j]-minimum_energy
    else:
      m[(n[z],p[l],J[i])]=m[(n[z],p[l],J[i])]-minimum_energy
   for i in range(len(J)):
       if isinstance(m[(n[z],p[l],J[i])],np.ndarray): 
         m[(n[z],p[l],J[i])].sort()
 return m  
def calculate_loss_function(filename,p,n,v):#n,p as lists 
 N=0
 s=0   
 Ecal= calculated_excitation_energies(filename,p,n,v)
 for j in range(len(n)):
   Eexp=[]  
   Eexp=experimental_values('Data.txt',int(n[j]),0)
   for i in range(9):
        Eexp.update(experimental_values('Data.txt',int(n[j]),i+1))
     
   for l in range(len(p)):
    J=extract_J_values('J values.txt',n[j],p[l])   
    for k in range(len(J)):
      if (n[j],p[l],J[k]) in Eexp:   
        for i in range(len(Eexp[(n[j],p[l],J[k])])):
          N=N+1 
          if isinstance(Ecal[(n[j],p[l],J[k])], float) or isinstance(Ecal[(n[j],p[l],J[k])], int) :  
           s+=(float(Eexp[(n[j],p[l],J[k])][i])-Ecal[(n[j],p[l],J[k])])**2
          else:
           s+=(float(Eexp[(n[j],p[l],J[k])][i])-Ecal[(n[j],p[l],J[k])][i])**2
 return s,N
def root_mean_square_loss_function(filename,p,n,v):
    s,N= calculate_loss_function(filename,p,n,v)
    return np.sqrt((1/N)*(s))
      

