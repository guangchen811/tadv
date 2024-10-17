#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# for visualization 
import matplotlib.pyplot as plt
import numpy as np
# for operation in data 
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
# models 
from sklearn.linear_model import LogisticRegression
# for evaluation 
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
# to split data
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
# for preprocessing 
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, OneHotEncoder,
                                   StandardScaler)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# In[ ]:


df=pd.read_csv('/kaggle/input/healthcare-dataset/healthcare_dataset.csv')
df.head()


# In[ ]:


df.info()


# # check NAN value

# In[ ]:


df.isna().sum()


# # check duplicated rows

# In[ ]:


df.duplicated().sum()


# # Drop duplicates

# In[ ]:


df.drop_duplicates(inplace=True)


# In[ ]:


df.head()


# In[ ]:


names_count=df.Name.value_counts().sort_values(ascending=False).head(10)
names_count


# In[ ]:


plt.bar(names_count.index,names_count.values)
plt.xticks(rotation=90)

plt.show()


# # change type of date columns

# In[ ]:


df['Date of Admission']=pd.to_datetime(df['Date of Admission'])
df['Discharge Date']=pd.to_datetime(df['Discharge Date'])
df['duration']=(df['Discharge Date']-df['Date of Admission']).dt.days


# In[ ]:


df.drop(columns=['Name','Date of Admission','Discharge Date'],inplace=True)


# # how many blood type in each class

# In[ ]:


plt.figure(figsize=(10,8))
sns.countplot(data=df,x='Blood Type',hue='Test Results')

plt.show()


# # relationship between medical condition and age

# In[ ]:


plt.figure(figsize=(15,10))
for index,i in enumerate(df['Medical Condition'].unique()):
    plt.subplot(2,3,index+1)
    sns.distplot(df[df['Medical Condition']==i]['Age'])
    plt.title(i)
    
plt.show()


# # which hospital receives more patients

# In[ ]:


ax=df['Hospital'].value_counts(ascending=False).head(15).plot(kind='bar')
ax.bar_label(ax.containers[0],fontsize=13)
plt.show()


# In[ ]:


df.head()


# # which Insurance Providers have more patients

# In[ ]:


ax=df['Insurance Provider'].value_counts(ascending=False).plot(kind='bar')
ax.bar_label(ax.containers[0],fontsize=13)
plt.show()


# # how many males and females in each Medical Condition 

# In[ ]:


plt.figure(figsize=(15,10))
for index,i in enumerate(df['Medical Condition'].unique()):
    plt.subplot(2,3,index+1)
    ax=sns.countplot(data=df[df['Medical Condition']==i],x='Gender')
    ax.bar_label(ax.containers[0],fontsize=13)
   
    plt.title(i)
    
plt.show()


# # How much medical condition cost 

# In[ ]:


ax=sns.barplot(x='Medical Condition',y='Billing Amount',data=df)
ax.bar_label( ax.containers[0],fontsize=13)
plt.show()


# # check outlier

# In[ ]:


plt.figure(figsize=(10,9))
sns.boxenplot(data=df['Billing Amount'])
plt.show()


# # encoding data

# In[ ]:


lb_encoder=LabelEncoder()
for col in df.select_dtypes('object'):
    df[col]=lb_encoder.fit_transform(df[[col]])
df.head()


# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='coolwarm')
plt.show()


# In[ ]:


df.drop(columns=['Room Number','Hospital','Doctor'],inplace=True)


# # split data to x,y
# 

# In[ ]:


x=df.drop('Test Results',axis=1)
y=df.loc[:,'Test Results']


# # scaling x 

# In[ ]:


scaler=StandardScaler()
x=pd.DataFrame(scaler.fit_transform(x))


# # split x,y to train and test

# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,
                                              test_size=0.2,shuffle=True
                                              , stratify=y)


# # Decision Tree

# In[ ]:


model_dic={"DT":DecisionTreeClassifier(criterion='entropy'),
          "Logistic_R":LogisticRegression(),
          "NB":GaussianNB(),
          "Random_F":RandomForestClassifier(n_estimators=100)
          }

for key,model in model_dic.items():
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    print(key,accuracy_score(y_test,y_pred))


# # 

# In[ ]:





# In[ ]:




