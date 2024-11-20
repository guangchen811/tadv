#%% md
# # Example Notebook
#%%
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().system('pip install -U tqdm')
import pandas as pd
#%%
# Load the data
df = pd.read_csv('/kaggle/input/example_dataset_1/example_table.csv')

print(df.columns)
#%%
df['FullName'] = df['FirstName'] + ' ' + df['LastName']
#%%
df.to_csv('submission.csv', index=False)