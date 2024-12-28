import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import ast
from scipy.optimize import curve_fit
from fractions import Fraction
import random



filepath = "data\data4.csv"



def data_exp_form(file_path_2):
    '''
    Filter the Experimental data file by cutting off the states with uncertainties noted by 1

    Parameters :
    ------------
    `filepath` : the filepath where the data is saved

    Returns :
    ---------
    `data frame` : a panda data frame with (n,p,J,E) as columns
    '''


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

def extract_J_GS(n,p):
    """
    Return :
    -----------
    `Float` : value of the spin of the ground state
    
    """
    value = data.loc[(data['E'] == 0) & (data['n'] == n) & (data['p'] == p),'J'].iloc[0]
    return Fraction(value)

def modify_string(string):
    '''
    Convert the string file as proper matrix of string
    Ex : {{ -> [[

    Parameters :
    ------------
    `string` : string , here an isotope file read as a string

    Returns :
    ---------
    `string` : string, a string of matrix of string
    '''
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
    """
    Same as for E_J but return a list with unique J
    """
    data1 = data
    mask = (data1['n'] == n) & (data1['p'] == p) 
    ndata = data1[mask]
    listj=ndata['J'].apply(lambda x: pd.eval(x) if '/' in x else float(x))

    return (pd.unique(listj))

def extract_E_J_states(n,p):
    '''
    Extract the energies and spin states from an isotope

    Parameters :
    ------------
    `float` : n and p, the number of proton and neutron

    Returns :
    ---------
    `list` : E and j lists in the same order (it matters here)
    '''
    mask = (data['n'] == n) & (data['p'] == p) 
    ndata = data[mask]

    return np.array(ndata['J'].values[:]),np.array(ndata['E'].values[:])

def evaluate_expression_v2(expr, *v):
    # Replace variables in the expression
    for i, val in enumerate(v[0:5]):                 # v = vpp,vnn,vpn
        expr = expr.replace(f"vpp[{2*i}]", str(val))
    for i, val in enumerate(v[5:10]):                 # v = vpp,vnn,vpn
        expr = expr.replace(f"vnn[{2*i}]", str(val))
    for i, val in enumerate(v[10:]):                 # v = vpp,vnn,vpn
        expr = expr.replace(f"vnp[{i}]", str(val))
    # Evaluate the resulting expression
    return eval(expr)

def v_evaluate(mat,*v):
    return np.array([[evaluate_expression_v2(expr, *v) for expr in sublist] for sublist in mat], dtype=float)
    
# list of the isotopes with data ( to increase if needed or more ) 

test_list=[[10,9],[10,8],[10,7],[10,6],[10,5],[10,3],[10,4],[10,2],
[9,9],[9,8],[9,7],[9,6],[9,5],[9,3],[9,4],[9,2],[9,1],
[8,7],[8,6],[8,5],[8,4],[8,3],[8,2],[8,1],[8,0]]

def initalization(data):
    '''
    Extract the relevant matrix from the Theoritcal file from Mathematica. 

    Shape them in the good format for python

    Stock them into a dictionnary

    Parameter :
    ------------
    `data` : Filtered panda frame

    Return :
    ---------
    `Dictionnary` : a key for each spin state in the following format, (n,p,J)
    '''

    dict_mat={}

    for np in test_list:  # to replace with :  np in n_p_mat

        n=np[0]
        p=np[1]
        i = (np[0]+np[1]) % 2 
        J_list = extract_J_values(n,p)

        path = "data\piso"+str(p)+"\mat"+f"{n:02}"+f"{p:02}"+".txt"
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
    
def initalization_ph(data):
    '''
    Extract the relevant matrix from the Theoritcal file from Mathematica. 

    Shape them in the good format for python

    Stock them into a dictionnary

    Parameter :
    ------------
    `data` : Filtered panda frame

    Return :
    ---------
    `Dictionnary` : a key for each spin state in the following format, (n,p,J)
    '''

    dict_mat={}

    for np in test_list:  # to replace with :  np in n_p_mat

        n=np[0]
        p=np[1]
        i = (np[0]+np[1]) % 2 
        J_list = extract_J_values(n,p)

        path = "data\piso"+str(p)+"\mat"+f"{10-n:02}"+f"{p:02}"+".txt"
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

#dict_th = initalization(data) # dictionnary of the matrix
dict_th_ph = initalization_ph(data)

def diagonalisation(np_list,v):  # plutot une liste de p et de n
    '''
    For every isotopes, evaluate all the experimentaly observed spin matrix with the coefficient v

    Diagonalise the matrix and substract to all, for an isotope, the minimum value of the Ground State matrix

    Return the loss function, the experimental and calculated energies in the same order 
    


    Parameters :
    ------------
    `list` : np_list, list of isotopes
    `list` : v, list of coefficient


    Return :
    ---------
    `float` : the mean square root of the loss function
    '''
    loss_list = np.array([])
    loss_list2 = np.array([])
    E_loss_list = np.array([])
    J0_list = np.array([])

    for n_p in np_list:

        n,p=n_p[0],n_p[1]
        #print(n,p)
        J_list = extract_J_values(n,p)
        J1_list,E_list = extract_E_J_states(n,p)
        J0_list = np.append(J1_list,J0_list)
        dico_j = {}

        sub = 0

        for j in J_list: # one iteration of each j in this list
            mat = v_evaluate(dict_th_ph[(str(n),str(p),str(j))],*v)
            eigen_mat = np.linalg.eigh(mat)[0]
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
        
    #print("not sub list :",loss_list2),print("sub list :",loss_list),print("diff",E_loss_list-loss_list)

    return np.sqrt(np.sum((loss_list[loss_list !=0]-E_loss_list[E_loss_list !=0])**2)/loss_list[loss_list !=0].size),loss_list,E_loss_list,J0_list

def func_diago(np_list,*v):  

    loss_list = np.array([])
    print(v)

    for n_p in np_list:

        n,p=n_p[0],n_p[1]
        #print(n),print(p)
        J_list = extract_J_values(n,p)
        J1_list = extract_E_J_states(n,p)[0]
        dico_j = {}

        sub = 10e100

        for j in J_list: # one iteration of each j in this list
            mat = v_evaluate(dict_th_ph[(str(n),str(p),str(j))],*v) 
            eigen_mat = np.linalg.eigh(mat)[0]
            #print(eigen_mat)
            if j == extract_J_GS(n,p): # if Ground State which should be in first position butif not the case 
                sub = np.min(eigen_mat)
            #if (np.min(eigen_mat) < sub): # if Ground State which should be in first position butif not the case mmmhh
                #sub = np.min(eigen_mat)
                #print(sub)
            dico_j[j] = np.sort(eigen_mat)
        ordered_list = np.zeros(0)

        for k in J1_list:
            ordered_list = np.append(ordered_list,dico_j[Fraction(k)][0])
            dico_j[Fraction(k)] = np.delete(dico_j[Fraction(k)],0)
        loss_list = np.append(loss_list,ordered_list-sub)

    #print(loss_list)
    return loss_list

# list of isotopes to fit
fit_list=[[10,8],[10,7],[10,6],[10,5],[10,4],[10,3],[10,2],
[9,9],[9,6],[9,5],[9,4],[9,3],[9,2],[9,1],
[8,7],[8,6],[8,5],[8,4],[8,3],[8,2],[8,1],[8,0]]

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
    v1=np.concatenate(([0], v[:4]))
    v2=np.concatenate(([0], v[4:8]))
    v3=np.concatenate(([0], v[8:]))
    v=np.concatenate((v1,v2,v3))
    return func_inter(input1_list,*v)

def curv_fit_function(curv_list):
    '''
    Use fitting method : scipy curve_fit which requires that x and y must have the same length

    Create as many np list there is as there are experimental energies

    Print energies and coefficient
    


    Parameters :
    ------------
    `list` : curv_list, list of isotopes

    Return :
    ---------
    `plot` : Plot of the energies

    `list` : Coefficient values, v_fin

    `array` : Covariant matrix, cov_v_th



    '''

    input_list = np.array([],dtype=int)
    output_list = np.array([])
    v0 = [random.uniform(0, 3500) for _ in range(8)] + [random.uniform(-3500, 0) for _ in range(9)] 
    print(" intitial parameters : ",v0)

    for n_p_ in curv_list:
        e = extract_E_J_states(n_p_[0],n_p_[1])[1]
        output_list = np.append(output_list,e)

        partial_list = np.full(shape=e.size,fill_value=int(str(n_p_[0])+f"{n_p_[1]:02}"),dtype=int)
        input_list = np.append(input_list,partial_list)
        
    #print(input_list),#print(output_list)
    v_th,cov_v_th=curve_fit(func_inter_0,input_list,output_list,p0=v0) #p0 =v_test
    std_devs = np.sqrt(np.diagonal(cov_v_th))

    # Créer une matrice de corrélation
    cor_matrix = cov_v_th / np.outer(std_devs, std_devs)

    print("Matrice de corrélation :\n", cor_matrix)
    print("Uncertainties :\n",std_devs)
    v_1=np.concatenate(([0], v_th[:4]))
    v_2=np.concatenate(([0], v_th[4:8]))
    v_3=np.concatenate(([0], v_th[8:]))
    v_fin=np.concatenate((v_1,v_2,v_3))
    loss,E_calc,E_exp,list_J = diagonalisation(fit_list,v_fin)

    for i,vi in enumerate(v_fin):
        print("v["+str(i)+"] = ",vi)
    x = np.arange(len(E_calc))
    comp_list = np.vstack((list_J,(E_calc-E_exp)))
    print(comp_list)
    print("loss : ",loss)
    plt.plot(x, E_exp, marker='^', mfc='r', mec='r', ms=6, ls='--', c='r', lw=2)
    plt.plot(x, E_calc, marker='+', mfc='k', mec='k', ms=6, ls='--', c='b', lw=2)
    plt.show()
    return loss,E_calc,E_exp 

a,b,c=curv_fit_function(fit_list)

#v=[0.0, 1481.7253626150946, 2302.1162652646462, 2469.8448335009834, 2624.0736069385393, 0.0, 1437.4638489830231, 2260.0796044311396, 2793.905999859872, 2741.630536194231, 0.0, 2644.9797287207307, 1294.1784477120023, 3140.2632639601443, 2053.040849030701, 2826.8972038158167, 3098.887133491251, 2245.6031970398094, 3483.070225454131, 822.5005559807602]
#a,b,c,d = diagonalisation(test_list,v)
#print(a)