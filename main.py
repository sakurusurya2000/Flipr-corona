#!/usr/bin/env python
# coding: utf-8

# GETTING DATA

# In[1]:


import numpy as np


# In[2]:


data = np.load("./data.npz")
X_train = data["X_train"]
Y_train = data["Y_train"]
X_test = data["X_test"]
print("Shapes")
print("X_train:",X_train.shape)
print("Y_train:",Y_train.shape)
print("X_test:",X_test.shape)


# COMPILING KERAS MODEL

# In[8]:


from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential


# In[9]:


model = Sequential()
model.add(Dense(32, input_dim=21, kernel_initializer='glorot_uniform',
                bias_initializer='zeros', activation="relu"))
model.add(Dense(64, kernel_initializer='glorot_uniform',
                bias_initializer='zeros', activation="relu"))
model.add(Dense(32, kernel_initializer='glorot_uniform',
                bias_initializer='zeros', activation="relu"))
model.add(Dense(1, kernel_initializer='glorot_uniform',
                bias_initializer='zeros', activation="sigmoid"))


# In[10]:


model.compile(loss="mean_squared_error", optimizer="adam")


# TRAINING MODEL

# In[11]:


labels = Y_train.reshape(-1,1)
epochs = 10
labels


# In[12]:


model.fit(X_train,Y_train.reshape(-1,1), validation_split=0.1, epochs=epochs)


# In[14]:


y_pred = model.predict(X_train)


# In[16]:


y_pred


# In[17]:


labels


# In[13]:


model.save("model.h5")

