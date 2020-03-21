#!/usr/bin/env python
# coding: utf-8

# DATA PREPROCESSING

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


def split_dataset(sequence, n_steps):
    X = list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix == len(sequence):
            break
        seq_x = sequence[i:end_ix+1]
        X.append(seq_x)
    return pd.DataFrame(X)


# In[3]:


def get_data(path,n_steps):
    data = pd.read_excel(path, sheet_name="Diuresis_TS", dtype="int32")
    people_ID = data["people_ID"].values
    data = data.drop("people_ID", axis = 1)
    database = pd.DataFrame()
    for i in range(len(data)):
        arr = split_dataset(data.values[i],n_steps)
        database = database.append(arr, ignore_index=True)
    print("data:", data.shape)
    print("database:", database.shape)
    return (database, people_ID)


# In[4]:


n_steps = 1
train_database, train_people_ID = get_data("./Train_dataset.xlsx",n_steps)
Y_train = train_database[n_steps].values
train_database = train_database.drop(n_steps, axis=1)
X_train = train_database.values
print(X_train.shape)
print(Y_train.shape)


# In[5]:


np.savez("time_data.npz", X_train=X_train, Y_train=Y_train, train_people_ID= train_people_ID, n_steps=np.array([n_steps]))


# GET DATA

# In[6]:


import numpy as np
data = np.load("time_data.npz")
X_train = data["X_train"]
Y_train = data["Y_train"]
train_people_ID = data["train_people_ID"]
n_steps = data["n_steps"][0]


# In[7]:


print(n_steps)


# MODEL COMPILE

# In[8]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input


# In[9]:


model = Sequential()
model.add(Input((n_steps,1)))
model.add(LSTM(units=50, activation="relu"))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")


# In[10]:


X = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))


# In[20]:


Y = Y_train.reshape(-1,1)


# In[12]:


print("X:", X.shape)
print("Y:", Y.shape)


# In[13]:


model.fit(X,Y,epochs=7, validation_split=0.2)


# In[14]:


model.summary()


# In[15]:


model.save("Time_model.h5")


# LOAD MODELS

# In[1]:


from tensorflow.keras.models import load_model
model = load_model("model.h5")
time_model = load_model("Time_model.h5")


# In[2]:


time_model.summary()


# MODIFYING TEST DATASET FOR 27th march

# In[3]:


import pandas as pd
test_dataset = pd.read_excel("Test_dataset.xlsx")
test_diuresis = test_dataset["Diuresis"]


# In[15]:


input_arr = test_diuresis.values.reshape(-1,1,1)
## n is no of days
n = 7
## 27th march - 20th march = 7 days
print(input_arr.shape)


# In[16]:


for i in range(n):
    input_arr = time_model.predict(input_arr)
    input_arr = input_arr.reshape(-1,1,1)


# In[21]:


input_arr = input_arr.flatten()


# In[28]:


test_dataset["Diuresis"] = input_arr.astype("int32")


# In[29]:


test_dataset.to_excel("modified_test_dataset.xlsx")


# In[ ]:




