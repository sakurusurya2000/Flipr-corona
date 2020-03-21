#%%
import pandas as pd
import numpy as np
from sklearn import preprocessing

#%%
variable_path = "./Variable_Description.xlsx"
train_path = "./Train_dataset.xlsx"
test_path = "./Test_dataset.xlsx"

#%%
nonMLVariables = np.array(['people_ID', 'Region', 'Designation', 'Name'])
maybeVariables = np.array(['Insurance','salary'])

#%%
def get_data(path,type,min_max_scaler=None):
    data = pd.DataFrame()
    excel_data = pd.read_excel(path)
    variables = excel_data.columns.values
    variables = np.setdiff1d(variables, nonMLVariables)
    variables = np.setdiff1d(variables, maybeVariables)
    print(variables)
    for i in variables:
        print(i)
        arr = excel_data[i]
        if(i in ["Gender","Married","Occupation","Mode_transport","comorbidity","Pulmonary score","cardiological pressure"]):
            codes, uniq = arr.factorize(sort=True)
            uniq = np.insert(uniq.values, 0, "0None", axis=0)
            arr = arr.fillna("0None")
            cat = pd.Categorical(arr, categories=uniq, ordered=False)
            arr = cat.codes
        data[i] = arr.astype("float32")
        if (i in ["Insurance","salary"]):
            data[i] = data[i]/100000
        if(type == "train"):
            min_max_scaler = preprocessing.MinMaxScaler()    
            if (i == "Infect_Prob"):
                data[i] = data[i]/100    
            else:
                temp = min_max_scaler.fit_transform(data[i].values.reshape(-1,1)).flatten()
                data[i] = pd.Series(temp, name=i)
        elif(type == "test"):
            temp = min_max_scaler.transform(data[i].values.reshape(-1,1)).flatten()
            data[i] = pd.Series(temp, name=i)
    data = data.fillna(0.0)
    del excel_data       
    if(type == "train"):  
        Y = data["Infect_Prob"].values
        data = data.drop("Infect_Prob", axis=1)
        X = data.values
        return X,Y,min_max_scaler
    elif(type == "test"):
        X = data.values
        return X
    else:
        raise ValueError()    
# %%
X_train, Y_train, min_max_scaler = get_data(train_path, "train")

# %%
X_test = get_data(test_path, "test", min_max_scaler)

#%%
np.savez("data.npz", X_train=X_train, Y_train=Y_train, X_test=X_test)

# %%
data = np.load("data.npz")