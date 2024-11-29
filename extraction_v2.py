import numpy as np
import pandas as pd
import re
import ast
from scipy.optimize import curve_fit
from fractions import Fraction


filepath = "data\data2.csv"



def data_exp_form(file_path_2):


    file = pd.read_csv(file_path_2,sep=',') # read

    file['E'] = file['E'].str.replace(',', '.').astype(float) # change type
    #file['J'] = file['J'].apply(lambda x: pd.eval(x) if '/' in x else float(x))

    mask = (file['State'] !=1) & (file['Parity'] != 1) & (file['Delta_E'] !=1) # ignore the uncertainties for the moment
    filtered_file = file[mask]

    data1 = filtered_file.drop(columns=['State','Parity','Delta_E'])
    return data1

def mask_state(data): # add to thins function the filter with not good states are filtered with v as parameter should be put inside intialization
    # Group by 'n' and 'p'
    grouped = data.groupby(['n', 'p'])

    # Identify groups with at least one row where E = 0
    valid_groups = grouped.filter(lambda group: (group['E'] == 0).any())

    #print("Filtered DataFrame:")
    #print(valid_groups[6:10])
    return valid_groups  

data = data_exp_form(filepath)
data = mask_state(data)
#print(data.head())

def extract_n_p(data):
    data2 = data[["n","p"]].drop_duplicates()
    return data2.to_numpy()

n_p_mat=extract_n_p(data)

def extract_J_GS(n,p):
    value = data.loc[(data['E'] == 0) & (data['n'] == n) & (data['p'] == p),'J'].iloc[0]
    return Fraction(value)

def extract_matrices_from_file(file_path):

    with open(file_path, 'r') as file:
        content = file.read()
    
        modified_text = re.sub(r'\{',r'[', content)
        modified_text = re.sub(r'\}',r']', modified_text)
        modified_text = re.sub(r'\bSqrt\[(.*?)\]',r'np.sqrt(\1)', modified_text)

    modified_text = re.sub(r'\n',r',', modified_text)
    return [modified_text]

def modify_string(string):
    string = re.sub(r',',r'","',string)
    string = re.sub(r'{{',r'[["',string)
    string = re.sub(r'}}',r'"]]',string)
    string = re.sub(r'}"," {',r'"],["',string)
    string = re.sub(r'\bSqrt\[(.*?)\]',r'np.sqrt(\1)', string)
    string = re.sub(r'\n','',string)
    return string

def extract_E_values(n,p,J):
    mask = (data['n'] == n) & (data['p'] == p) & (data['J'] == J)
    ndata = data[mask]

    return ndata['E'].values[:]



def extract_J_values(n,p):
    data1 = data
    mask = (data1['n'] == n) & (data1['p'] == p) 
    ndata = data1[mask]
    listj=ndata['J'].apply(lambda x: pd.eval(x) if '/' in x else float(x))

    return (pd.unique(listj))

def extract_E_J_states(n,p):
    mask = (data['n'] == n) & (data['p'] == p) 
    ndata = data[mask]

    return np.array(ndata['J'].values[:]),np.array(ndata['E'].values[:])

def evaluate_expression(expr, *v):
    # Replace variables in the expression
    for i, val in enumerate(v):
        expr = expr.replace(f"v[{i}]", str(val))
    # Evaluate the resulting expression
    return eval(expr)

def v_evaluate(mat,*v):
    return np.array([[evaluate_expression(expr, *v) for expr in sublist] for sublist in mat], dtype=float)
    


test_list=[[10,9],[10,8],[10,7],[10,6],[10,5],[10,3],[10,4],[10,2],[10,1],[9,10],[9,9],[9,8],[9,7],[9,6],[9,3],[9,4],[9,2],[9,1]]

def initalization(data):

    dict_mat={}

    for np in test_list:  # to replace with :  np in n_p_mat

        n=np[0]
        p=np[1]
        i = (np[0]+np[1]) % 2 
        J_list = extract_J_values(n,p)

        path = "data\p"+str(p)+"\mat"+f"{n:02}"+f"{p:02}"+".txt"
        with open(path ,"r") as f:
            line = f.read()
            line_list = re.split(r"\n", line)

            for j in J_list:
                t=j-i/2
                val = line_list[int(t)]
                val = modify_string(val)
                val = ast.literal_eval(val)
                dict_mat[(str(n),str(p),str(j))] = val

    
    return dict_mat
    
dict_th = initalization(data)
def diagonalisation2(np_list,v):  # plutot une liste de p et de n

    loss_list = np.array([])
    loss_list2 = np.array([])
    E_loss_list = np.array([])

    for n_p in np_list:

        n,p=n_p[0],n_p[1]
        J_list = extract_J_values(n,p)
        J1_list,E_list = extract_E_J_states(n,p)
        dico_j = {}

        sub = 1000000000000
        print(sub)

        for j in J_list: # one iteration of each j in this list
            mat = v_evaluate(dict_th[(str(n),str(p),str(j))],*v)
            eigen_mat = np.linalg.eig(mat)[0]
            print(np.min(eigen_mat))
            print(sub)
            if (np.min(eigen_mat) < sub): # if Ground State which should be in first position butif not the case mmmhh
                sub = np.min(eigen_mat)
                print(sub)
            dico_j[j] = np.sort(eigen_mat)
 
        ordered_list = np.zeros(0)
        print("sub = ",sub)
        for k in J1_list:
            ordered_list = np.append(ordered_list,dico_j[Fraction(k)][0])
            dico_j[Fraction(k)] = np.delete(dico_j[Fraction(k)],0)

        print("sub = ",sub)
        loss_list2 = np.append(loss_list2,ordered_list)
        loss_list = np.append(loss_list,ordered_list-sub)
        E_loss_list = np.append(E_loss_list,E_list)
        
    print(loss_list2),print(loss_list)
    return np.sqrt(np.sum((loss_list-E_loss_list)**2)/loss_list.size)

def diagonalisation(np_list,v):  # plutot une liste de p et de n

    loss_list = np.array([])
    loss_list2 = np.array([])
    E_loss_list = np.array([])

    for n_p in np_list:

        n,p=n_p[0],n_p[1]
        J_list = extract_J_values(n,p)
        J1_list,E_list = extract_E_J_states(n,p)
        dico_j = {}

        sub = 0

        for j in J_list: # one iteration of each j in this list
            mat = v_evaluate(dict_th[(str(n),str(p),str(j))],*v)
            eigen_mat = np.linalg.eig(mat)[0]
            if j == extract_J_GS(n,p): # if Ground State which should be in first position butif not the case mmmhh
                sub = np.min(eigen_mat)
            dico_j[j] = np.sort(eigen_mat)

        ordered_list = np.zeros(0)

        for k in J1_list:
            ordered_list = np.append(ordered_list,dico_j[Fraction(k)][0])
            dico_j[Fraction(k)] = np.delete(dico_j[Fraction(k)],0)

        loss_list2 = np.append(loss_list2,ordered_list)
        loss_list = np.append(loss_list,ordered_list-sub)
        E_loss_list = np.append(E_loss_list,E_list)
        
    print(loss_list2),print(loss_list)
    return np.sqrt(np.sum((loss_list-E_loss_list)**2)/loss_list.size)

def func_diago(np_list,*v):  # plutot une liste de p et de n

    loss_list = np.array([])

    for n_p in np_list:

        n,p=n_p[0],n_p[1]
        J_list = extract_J_values(n,p)
        J1_list = extract_E_J_states(n,p)[0]
        dico_j = {}

        sub = 0

        for j in J_list: # one iteration of each j in this list
            mat = v_evaluate(dict_th[(str(n),str(p),str(j))],*v)
            eigen_mat = np.linalg.eig(mat)[0]
            if j == extract_J_GS(n,p): # if Ground State which should be in first position butif not the case mmmhh
                sub = np.min(eigen_mat)
            dico_j[j] = np.sort(eigen_mat)

        ordered_list = np.zeros(0)

        for k in J1_list:
            ordered_list = np.append(ordered_list,dico_j[Fraction(k)][0])
            dico_j[Fraction(k)] = np.delete(dico_j[Fraction(k)],0)

        loss_list = np.append(loss_list,ordered_list-sub)
        
    return loss_list


#v = np.array([0,1000,2000,1500,3000,2600,2700,990,1600,2000])
#a = diagonalisation(test_list,v)
#print(a)

  
# x and y have to be the same size and x must be flatten out in the function donc suivre un certain ordre en connaissant leur taille pour les déflatten out
# input de la fonction curve_fit : liste de (n,p) à fit
# créer une liste d'output exp 
# pour le premier (n1,p1) : faire une liste des J,E pour (n1,p1) dans l'ordre expérimental, la liste des E va se concaténer à une liste
# E_tot qui prend tout, on garde la taille de cette liste pour flatten out
# on enchaine pour tous les (nx,px) dans la liste initiale
# on se retrouve avec des y = E_tot dans l'ordre et une liste de taille des (nx,px)
# en input juste créer une liste tot de (n_i,p_i) dans le même ordre et les redécouper par taille 
# la fonction découpe l'input par changement de (ni,pi), crée une liste de (n,p) unique et appelle func_diago qui renvoie 
# une liste de E_théorique dans le même sens que les E_exp 




fit_list=[[9,6],[9,4],[9,2],[9,3],[9,1]]

def func_inter(input1_list,*v):
    result = []
    #print(input1_list)
    for num in input1_list:
        # Extract the first two digits and the last digit
        split = [int(str(int(num))[:-2]), int(str(int(num))[-2:])]
        # Add only if not already in the result
        if split not in result:
            result.append(split)

    return func_diago(result,*v)

def func_inter_0(input1_list,*v):
    v1 = np.concatenate(([0],v[1:10]),axis=None)
    return func_inter(input1_list,*v1)

def curv_fit_function(curv_list):

    input_list = np.array([],dtype=int)
    output_list = np.array([])

    for n_p_ in curv_list:
        e = extract_E_J_states(n_p_[0],n_p_[1])[1]
        output_list = np.append(output_list,e)

        partial_list = np.full(shape=e.size,fill_value=int(str(n_p_[0])+f"{n_p_[1]:02}"),dtype=int)
        input_list = np.append(input_list,partial_list)
        
    #print(input_list),#print(output_list)
    v_th,cov_v_th=curve_fit(func_inter_0,input_list,output_list,p0=v_test)
    v_th[0]=v_th[0] - v_test [0]
    loss = diagonalisation(fit_list,v_th)
    for i,vi in enumerate(v_th):
        print("v["+str(i)+"] = ",vi)

    print("loss : ",loss)



v_test= [-2321,-1524,-937,-700,-160,-447,140,-640,241,-1752]
a=[[9,6]]
curv_fit_function(fit_list)