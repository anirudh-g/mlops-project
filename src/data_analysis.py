# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
df = pd.read_csv('/home/ubuntu/mlops-project/data/heart_failure_clinical_records_dataset.csv')
df.head(5)

# %%
df.describe().T

# %%
df['age_bin'] = pd.cut(df['age'], [0, 10, 20, 30, 40, 50, 60, 70, 80, 100], labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '>80'])

# %%
sns.countplot(data=df, x='age_bin', hue='DEATH_EVENT')

# %%
plt.figure(figsize=(10,6))
df_corr = df.corr()
sns.heatmap(df_corr, annot=True)

# %%



