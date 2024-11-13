import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

filepath = "data\data1.csv"

def data_exp_form(file_path_2):


    file = pd.read_csv(file_path_2,sep=',') # read

    file['E'] = file['E'].str.replace(',', '.').astype(float) # change type
    file['J'] = file['J'].apply(lambda x: pd.eval(x) if '/' in x else float(x))

    mask = (file['State'] !=1) & (file['Parity'] != 1) & (file['Delta_E'] !=1) # ignore the uncertainties for the moment
    filtered_file = file[mask]

    data1 = filtered_file.drop(columns=['State','Parity','Delta_E'])
    return data1

data = data_exp_form(filepath)

def extract_E_J_states(n,p):
    mask = (data['n'] == n) & (data['p'] == p) 
    ndata = data[mask]

    return np.array(ndata['J'].values[:]),np.array(ndata['E'].values[:])


def plot_energy_level(n,p,dic):  # input : (n,p) and the dictionnary with the key n,p,j and with output the list of energies

    J_list,E_list=extract_E_J_states(n,p)
    J_list=np.array(J_list)
    j_list=[]
    E_calc=[]

    for j in J_list:
        i=j_list.count(j)
        E=dic.get((str(n),str(p),j))
        E_calc.append(E[i])
        j_list.append(j)

    energy_levels = {
        '2B-S': E_calc,
        'Exp': E_list
    }
    # Define positions for each model on the x-axis
    model_positions = list(energy_levels.keys())

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot energy levels for each state across different models
    for i, state in enumerate(J_list):
        energies = [energy_levels[model][i] for model in model_positions]
        ax.plot(model_positions, energies, marker=0,markersize=40 ,linestyle='--', label=state)

    # Add annotations for each state level next to the points
    for i, state in enumerate(J_list):
        for j, model in enumerate(model_positions):
            energy = energy_levels[model][i]
            ax.text(j, energy, state, ha='right', va='bottom', fontsize=5)

    # Set axis labels and title
    ax.set_xlabel("Models")
    ax.set_ylabel("Energy (keV)")
    ax.set_title("Energy levels : ("+str(n)+","+str(p)+")")

    # Display the legend
    ax.legend(title="States")

    # Show plot
    plt.show()
    
 
dico_test={
        ('10','4',0) : [0,2900],
        ('10','4',2) : [1500],
        ('10','4',4) : [3000],
        ('10','4',6) : [2600],
        ('10','4',8) : [2500,4000],
        ('10','4',10) : [4000],
        ('10','4',12) : [4500]

           }
#J_list,E_list=extract_E_J_states(10,4)
#print(J_list[0])
plot_energy_level(10,4,dico_test)

n,p=10,4
print(dico_test[(str(n),str(p),2.0)])

"""


# Define energy levels (in keV) for different models and states
# Replace with actual data for each model and state if available
energy_levels = {
    '2B-SM': [0, 1000, 2000, 3000],
    'Exp': [0, 1000, 2000, 3100]
}

# Corresponding state labels for each energy level in the right order
state_labels = ['0⁺', '2⁺', '4⁺', '6⁺']




# Define positions for each model on the x-axis
model_positions = list(energy_levels.keys())

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot energy levels for each state across different models
for i, state in enumerate(state_labels):
    energies = [energy_levels[model][i] for model in model_positions]
    ax.plot(model_positions, energies, marker='+', linestyle='--', label=state)

# Add annotations for each state level next to the points
for i, state in enumerate(state_labels):
    for j, model in enumerate(model_positions):
        energy = energy_levels[model][i]
        ax.text(j, energy, state, ha='right', va='bottom', fontsize=10)

# Set axis labels and title
ax.set_xlabel("Models")
ax.set_ylabel("Energy (keV)")
ax.set_title("Energy levels :","n and","p")

# Display the legend
ax.legend(title="States")

# Show plot
plt.show()

"""
"""
Get the J_E for (n,p) which are the input of the function

List_E , List_J

for j in the list take the values (n,p,j) in the dictionnary in the irgth order



"""