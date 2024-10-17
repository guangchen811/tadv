#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# In[ ]:


main_df=pd.read_csv('/kaggle/input/healthcare-dataset/healthcare_dataset.csv')


# In[ ]:


main_df.head()


# In[ ]:


df=main_df.copy()


# # EDA

# In[ ]:


df.columns


# In[ ]:


df.shape


# In[ ]:


df["Medication"].value_counts()


# In[ ]:


df.drop(columns=["Date of Admission","Room Number","Name","Doctor","Hospital","Discharge Date"],axis=1,inplace=True)


# In[ ]:


df


# In[ ]:


df.info()


# In[ ]:


df["Gender"] = df["Gender"].astype("category")


# In[ ]:





# In[ ]:


lis=["Blood Type","Medical Condition","Insurance Provider","Admission Type","Medication","Test Results"]
for i in lis:
  df[i]=df[i].astype("category")


# In[ ]:


df.info()


# In[ ]:


df.describe()
#


# In[ ]:


df[df["Billing Amount"]<0]


# In[ ]:


df.drop(index=df[df["Billing Amount"]<0].index,inplace=True)


# In[ ]:


df.describe()


# In[ ]:


df[df["Billing Amount"]<15000]


# In[ ]:


df.duplicated().sum()


# In[ ]:


df.drop_duplicates(inplace=True)


# In[ ]:


plt.boxplot(df.Age)


# In[ ]:


plt.boxplot(df["Billing Amount"])


# In[ ]:


import category_encoders as ce

encoder = ce.BinaryEncoder()
df_encoded = encoder.fit_transform(df)
df_encoded


# In[ ]:


df_encoded=df_encoded.drop("Gender_1",axis=1)


# In[ ]:


feature=df_encoded.drop(["Test Results_0","Test Results_1"],axis=1)
y=df_encoded[["Test Results_0","Test Results_1"]].copy()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Normalize the data
x = pd.DataFrame(scaler.fit_transform(feature), columns=feature.columns)

print("Min-Max Normalized Data:")
x


# In[ ]:




