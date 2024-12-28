from sklearn.model_selection import KFold
import numpy as np
from extraction_v2 import curv_fit_fold,diagonalisation


def k_fold(X):
    # Define the K-Fold cross-validator
    kf = KFold(n_splits=5, shuffle=True, random_state=45)

    # Placeholder to store results
    fold_accuracies = []

    # Iterate through each split
    for train_index, test_index in kf.split(X):
        # Split the data
        X_train = [X[i] for i in train_index]
        X_test = [X[i] for i in test_index]

        print(X_train),print(X_test)
        # Train the model (Logistic Regression in this example)
    
        train_loss,v,E_exp,E_th = curv_fit_fold(X_train)
        print(v)
        # Test the model
        test_loss,b,c,d = diagonalisation(X_test,v)
    
        # Evaluate accuracy
        fold_accuracies.append([train_loss,test_loss])
        print(f"Fold accuracy train set: ",train_loss),print(f"Fold accuracy test set: ",test_loss)

    # Overall results
    mean_train = np.array(fold_accuracies)
    mean_test = np.array(fold_accuracies)

    print(f"Mean accuracy train folds: ", np.mean(mean_train[:,0]))
    print(f"Mean accuracy test folds: ", np.mean(mean_test[:,1]))


# Example data
data = [[10,8],[10,7],[10,6],[10,5],[10,4],[10,3],[10,2],
[9,9],[9,6],[9,5],[9,4],[9,3],[9,2],[9,1],
[8,7],[8,6],[8,5],[8,4],[8,3],[8,2],[8,1],[8,0]] 

k_fold(data)


