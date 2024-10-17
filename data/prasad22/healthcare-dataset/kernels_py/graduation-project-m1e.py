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


import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

# In[ ]:


df = pd.read_csv('/kaggle/input/healthcare-dataset/healthcare_dataset.csv')


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.nunique()


# In[ ]:


df.duplicated().sum()


# In[ ]:


df.drop_duplicates(inplace = True)


# In[ ]:


df.isnull().sum()


# In[ ]:


#convert datatype to date
df['Date of Admission']=pd.to_datetime(df['Date of Admission'])
df['Discharge Date']=pd.to_datetime(df['Discharge Date'])
df.dtypes


# In[ ]:


#checking for outliers
plt.figure(figsize=(4,4))
sns.set_style('white')
sns.boxplot(df['Billing Amount'],color='salmon')
plt.title('Billing Amount Distribution')
plt.show()
#now we discover that there is no outliers


# In[ ]:


df['Billing Amount'].describe()


# In[ ]:


df['Gender'].value_counts()


# In[ ]:





# ### This Python code generates a pie chart that visualizes the distribution of genders (Male vs. Female)

# In[ ]:


#we want to discover the percent of males vs females
gender=df['Gender'].value_counts(normalize=True)
plt.figure(dpi=200,figsize=(3,2))
plt.pie(gender,labels=gender.index,colors=['blue','salmon'],
        wedgeprops=dict(width=.3,edgecolor='k'),
        shadow=True,startangle=45,autopct='%.2f%%')
plt.title('Male vs Female',fontsize=12)
plt.tight_layout()
plt.show()


# In[ ]:


df['Blood Type'].value_counts()


# In[ ]:


#now we want to discover the blood type per gender to know which blood types are more common for each gender
plt.figure(figsize=(12,8),dpi=200)
sns.countplot(x=df['Blood Type'],hue=df['Gender'],palette='magma')
plt.legend( bbox_to_anchor=(1.05,1),title='Gender')
plt.title('Blood Type Distribution Per Gender')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# In[ ]:


df['Medical Condition'].value_counts()


# In[ ]:


#here we want to discover what is the medical condition which are more common
#and the results of tests which show us how much dangerous each condition 
sorted_conditoins=df['Medical Condition'].value_counts().sort_values()
plt.figure(figsize=(20,12),dpi=200)
ax=sns.countplot(data=df,x='Medical Condition',palette='magma',edgecolor='k',hue='Test Results')
bars=ax.patches
for i,bar in enumerate(bars):
    height=bar.get_height()
    plt.text(bar.get_x()+bar.get_width()/2,height/1.1,f'{int(height)}',
             va='center',ha='center',color='white',fontweight='bold',fontsize=20,rotation=315)
plt.title('Medical Condion with Test Results Frequency',fontsize=24)
plt.xlabel('Medical Condion',fontsize=16)
plt.ylabel('Frequency',fontsize=16)
plt.legend(bbox_to_anchor=(1,1),title='Test Result',title_fontsize=20,fontsize=16)
plt.tight_layout()
plt.show()


# In[ ]:


df['Medication'].value_counts()


# In[ ]:


df['Doctor'].value_counts()


# In[ ]:


df['Hospital'].value_counts()


# In[ ]:


#here we chech the distribution of age which show us the range of ages for pateints
plt.figure(figsize=(20,14),dpi=200)
sns.set_style('dark')
sns.histplot(df['Age'],color='blue',kde=True)
plt.axvline(np.mean(df['Age']),ls='--',lw=2,label=f"Average Age: {np.mean(df['Age']):.2f}")
plt.title('Age Distribution',fontsize=24)
plt.xlabel('Age',fontsize=20)
plt.ylabel('Frequency',fontsize=20)
plt.legend(fontsize=16)
plt.tight_layout()
plt.show()


# In[ ]:


#let's analyze and visualize the relationship between medical conditions and the medications prescribed for them using a bar chart.
groupby=df.groupby(['Medical Condition','Medication']).size().unstack()
groupby.plot(kind='bar')
plt.xlabel('Medical Condition')
plt.ylabel('Count')
plt.title('Medical Conditions and Medications')
plt.legend(title='Medication')
plt.show()


# In[ ]:


#lets generates a pie chart visualizing the distribution of different "Admission Types" from a dataset.
Admission=df['Admission Type'].value_counts(normalize=True)
plt.figure(dpi=200,figsize=(3,2))
colors=sns.color_palette('magma',len(Admission))
plt.pie(Admission,labels=Admission.index,
        wedgeprops=dict(width=.3,edgecolor='k'),colors=colors,
        shadow=True,startangle=45,autopct='%.2f%%')
plt.title('Admission Types',fontsize=12)
plt.tight_layout()
plt.show()


# In[ ]:


#let's performs aggregation, transformation, and sorting operations on the dataset to analyze the Insurance 
#Provider column and the associated Billing Amount and Patient Count. 
provider=df.groupby(by='Insurance Provider').agg({'Billing Amount': 'sum','Insurance Provider':'size'})
provider.columns=['Billing Amount', 'Patients']
provider.reset_index(inplace=True)
provider['Billing Amount']=np.around(provider['Billing Amount'])
provider=pd.DataFrame(provider.sort_values(by='Patients'))
provider


# In[ ]:


#let's generates a bar plot to visualize the number of patients per insurance provider .
plt.figure(dpi=200,figsize=(8,4))
sns.set_style('white')
ax=sns.barplot(data=provider,x='Insurance Provider',y='Patients',palette='magma')
bars=ax.patches
for i,bar in enumerate(bars):
    height=bar.get_height()
    plt.text(bar.get_x()+bar.get_width()/2,height/1.1,f'{int(height)}',
             va='center',ha='center',color='white',fontweight='bold',fontsize=11)
plt.title('Number Of Patients Per Provider')
plt.tight_layout()
plt.show()


# In[ ]:


# let's creates a bar plot to visualize the top 10 hospitals by frequency ( the number of occurrences of each hospital in the dataset).
Top_10_Hospital=df['Hospital'].value_counts().sort_values(ascending=False).head(10).reset_index(name='Frequency')
plt.figure(figsize=(12,8),dpi=100)
sns.barplot(data=Top_10_Hospital,x='Frequency',y='Hospital',palette='magma')
plt.title('Top 10 Hospital')
plt.tight_layout()
plt.show()


# In[ ]:


#let's creates a comparison between the number of rooms and the number of patients in the dataset
compare={'Number Of Rooms':len(df['Room Number'].unique()),
         'Number Of Patients':len(df['Name'])}
compare_df=pd.DataFrame(compare,index=['NO:'])
print(compare_df)


# In[ ]:


#let's calculates and prints the number of patients per room based on the previously created DataFrame (compare_df), 
#which contains the number of unique rooms and the total number of patients.
Patients_per_Room=np.around(compare_df['Number Of Patients']/compare_df['Number Of Rooms'])
print(f'Patients Per Room: {Patients_per_Room}')


# ### let's calculates the staying period for each patient in the dataset by subtracting the date of admission from the discharge date and then converts the result into the number of days.

# In[ ]:


df['Staying Period']=df['Discharge Date']-df['Date of Admission']
df['Staying Period']=df['Staying Period'].dt.days


# In[ ]:


plt.figure(figsize=(20,14),dpi=200)
sns.set_style('dark')
sns.histplot(df['Staying Period'],color='blue',kde=True)
plt.axvline(np.mean(df['Staying Period']),ls='--',lw=2,label=f"Average Stying Period On Days: {np.mean(df['Staying Period']):.2f}")
plt.title('Staying Period Distribution',fontsize=24)
plt.xlabel('Days',fontsize=20)
plt.ylabel('Frequency',fontsize=20)
plt.legend(fontsize=20)
plt.tight_layout()
plt.show()


# In[ ]:


#Let's  calculates the correlation matrix for selected numerical columns in the DataFrame ( df )
# 'Billing Amount', 'Staying Period', and 'Age'
df_corr=df[['Billing Amount','Staying Period','Age']].corr()
df_corr


# In[ ]:


# let's creates a heatmap to visualize the relationship between 
# medical conditions and medications in the dataset using a cross-tabulation.
data1=pd.crosstab(df['Medical Condition'],df['Medication'])
plt.figure(dpi=200)
sns.heatmap(data1, annot=True, fmt='d', cmap='magma')
plt.show()


# In[ ]:


# let's performs a Chi-squared test of independence to assess whether there is a statistically significant relationship 
# between two categorical variables: Medical Condition and Medication.
alpha=.05
Chi_Stat_1,P_Value_1,DOF_1,Expected_1=stats.chi2_contingency(data1)
print(P_Value_1)
if P_Value_1 < alpha :
    print('We reject the H0')
    print('there is a relationship between Medical Condition and Medication')
else:
    print('We fail to reject the H0')
    print('there is no relationship between Medical Condition and Medication')


# In[ ]:


# let's create a heatmap for the relationship between Blood Type and Medication using the cross-tabulation of these two variables.

# Creating a cross-tabulation for Blood Type and Medication
data2 = pd.crosstab(df['Blood Type'], df['Medication'])
# Setting up the figure for the heatmap
plt.figure(dpi=200)
# Creating the heatmap for the correct cross-tabulation
sns.heatmap(data2, annot=True, fmt='d', cmap='magma')
# Displaying the heatmap
plt.title('Blood Type vs Medication')  # Optional: Add a title for better context
plt.show()


# In[ ]:


# let's performs a Chi-squared test of independence to evaluate whether 
# there is a statistically significant relationship between Blood Type and Medication.

Chi_Stat_2,P_Value_2,DOF_2,Expected_2=stats.chi2_contingency(data2)
print(P_Value_2)
if P_Value_2 < alpha :
    print('We reject the H0')
    print('there is a relationship between Blood Type and Medication')
else:
    print('We fail to reject the H0')
    print('there is no relationship between Blood Type and Medication')

