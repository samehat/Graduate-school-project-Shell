import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

filepath = "data\data1.csv"

file_path_data = "data\p5\mat1005.txt"

def data_exp_form(file_path_2):


    file = pd.read_csv(file_path_2,sep=',') # read

    file['E'] = file['E'].str.replace(',', '.').astype(float) # change type
    file['J'] = file['J'].apply(lambda x: pd.eval(x) if '/' in x else float(x))

    mask = (file['State'] !=1) & (file['Parity'] != 1) & (file['Delta_E'] !=1) # ignore the uncertainties for the moment
    filtered_file = file[mask]

    data1 = filtered_file.drop(columns=['State','Parity','Delta_E'])
    return data1

data = data_exp_form(filepath)
#print(data.head())

def extract_matrices_from_file(file_path):
    #matrices = []

    with open(file_path, 'r') as file:
        content = file.read()
    
        modified_text = re.sub(r'\{',r'[', content)
        modified_text = re.sub(r'\}',r']', modified_text)
        modified_text = re.sub(r'\bSqrt\[(.*?)\]',r'np.sqrt(\1)', modified_text)

    modified_text = re.sub(r'\n',r',', modified_text)
    return [modified_text]


def extract_E_values(n,p,J):
    mask = (data['n'] == n) & (data['p'] == p) & (data['J'] == J)
    ndata = data[mask]

    return ndata['E'].values[:]

mat1005 = extract_matrices_from_file(file_path_data)



v = [1,0,0,0,0,0,0,0,0,0]
expression_list = eval(mat1005[0])
print(expression_list)

J = extract_E_values(4,0,2)

#print(J)


'''
Manière de compter les états par ligne : 

Commencer par 0 ou 1 en fonction de la parité de n+p qui se trouve dans le nom du fichier 
    -----> donc il faut écrire le nom du fichier en terme de variable n et p : (n,p)
Puis augmenter l'itérateur de 1 à chaque ligne donc maintenant chaque sous liste 

Renvoyer à chaque liste d'énergie exp la matrice associée pour le bon isotope donc listé par (n,p,J)
    -----> 1) renvoyer un dictionnaire comme super structure 
    -----> 2) ??? se baser uniquement sur les itérateurs et les différentes listes et matrices non homogènes // pas de super structure

'''


