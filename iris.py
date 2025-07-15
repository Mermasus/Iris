import pandas as pds
df = pds.read_csv("Iris.csv")
df.head()
df.shape
df.info()
df.describe()
df.isnull().sum()
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df_unique = df.drop_duplicates(subset ="Species")
df["Species"].value_counts()

sns.scatterplot(x='PetalLengthCm', y='PetalWidthCm', 
	hue='Species', data=df, )
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.show()

df.select_dtypes(include=['number']).corr(method='pearson')

 

# IQR 
Q1 = np.percentile(df['SepalWidthCm'], 25, 
				interpolation = 'linear') 

Q3 = np.percentile(df['SepalWidthCm'], 75, 
				interpolation = 'linear') 
IQR = Q3 - Q1 

print("Old Shape: ", df.shape) 

# Upper bound 
upper = np.where(df['SepalWidthCm'] >= (Q3+1.5*IQR)) 

# Lower bound 
lower = np.where(df['SepalWidthCm'] <= (Q1-1.5*IQR)) 

# Removing the Outliers 
df.drop(upper[0], inplace = True) 
df.drop(lower[0], inplace = True) 

print("New Shape: ", df.shape) 

sns.boxplot(x='SepalWidthCm', data=df)
