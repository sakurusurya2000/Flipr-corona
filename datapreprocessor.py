#%%
import pandas as pd
import numpy as np
from sklearn import preprocessing

#%%
variable_path = "./Variable_Description.xlsx"
train_path = "./Train_dataset.xlsx"
test_path = "./Test_dataset.xlsx"
modified_test_path = "./modified_test_dataset.xlsx"
#%%
nonMLVariables = np.array(['people_ID', 'Region', 'Designation', 'Name'])
maybeVariables = np.array(['Insurance','salary'])

#%%
def get_data(type):
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    train_dataset = pd.read_excel(train_path)
    if(type == "test"):
        test_dataset = pd.read_excel(test_path)
    elif(type == "modified_test"):
        test_dataset = pd.read_excel(modified_test_path)
        
    variables = train_dataset.columns.values
    variables = np.setdiff1d(variables, nonMLVariables)
    variables = np.setdiff1d(variables, maybeVariables)
    test_variables = np.setdiff1d(variables, np.array(["Infect_Prob"]))
    for i in test_variables:
        print(i)
        train_arr = train_dataset[i]
        test_arr = test_dataset[i]
        if(i in ["Gender","Married","Occupation","Mode_transport","comorbidity","Pulmonary score","cardiological pressure"]):
            
            arr = train_arr
            codes, uniq = arr.factorize(sort=True)
            uniq = np.insert(uniq.values, 0, "0None", axis=0)
            arr = arr.fillna("0None")
            cat = pd.Categorical(arr, categories=uniq, ordered=False)
            arr = cat.codes
            train_data[i] = arr.astype("float32")
            
            arr = test_arr
            codes, uniq = arr.factorize(sort=True)
            uniq = np.insert(uniq.values, 0, "0None", axis=0)
            arr = arr.fillna("0None")
            cat = pd.Categorical(arr, categories=uniq, ordered=False)
            arr = cat.codes
            test_data[i] = arr.astype("float32")
        else:
            train_data[i] = train_dataset[i]
            test_data[i] = test_dataset[i]      
        min_max_scaler = preprocessing.MinMaxScaler()    
        temp = min_max_scaler.fit_transform(train_data[i].values.reshape(-1,1)).flatten()
        train_data[i] = pd.Series(temp, name=i)    
        temp = min_max_scaler.transform(test_data[i].values.reshape(-1,1)).flatten()
        test_data[i] = pd.Series(temp, name=i)    
    train_data = train_data.fillna(0.0)
    test_data = test_data.fillna(0.0)
    Y_train = train_dataset["Infect_Prob"].values/100
    X_train = train_data.values
    X_test = test_data.values
    return X_train, Y_train, X_test
# %%
X_train, Y_train, X_test = get_data("test")

#%%
X_train, Y_train, X_modified_test = get_data("modified_test")

#%%
np.savez("data.npz", X_train=X_train, Y_train=Y_train, X_test=X_test, X_modified_test=X_modified_test)

# %%
data = np.load("data.npz")
