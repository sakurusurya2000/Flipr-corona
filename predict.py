#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.models import load_model
import numpy as np


# In[2]:


model = load_model("model.h5")
data = np.load("data.npz")
X_test = data["X_test"]


# In[3]:


y_pred = model.predict(X_test)


# In[4]:


y_pred


# In[5]:


import pandas as pd


# In[32]:


data = pd.read_excel("./Test_dataset.xlsx")


# In[33]:


data = data["people_ID"]


# In[34]:


y_pred = y_pred.flatten()


# In[35]:


y_pred = pd.Series(y_pred, name="Infect_Prob")


# In[36]:


df = pd.DataFrame()


# In[37]:


df["people_ID"] = data
df["infect_prob"] = y_pred


# In[38]:


df


# In[39]:


df.to_csv('./results.csv', index = False)


# In[40]:


model.summary()


# In[ ]:




