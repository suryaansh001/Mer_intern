import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

#read the csv file 
df=pd.read_csv('/home/sury/proj/internship/dataset/e4/MEFAR Dataset Neurophysiological and Biosignal Data/MEFAR_preprocessed/MEFAR_preprocessed/MEFAR_UP.csv')
print(df.columns)
# #drop the columns 
df.drop(['X','Y','Z',   'Delta', 'Theta', 'Alpha1',
       'Alpha2', 'Beta1', 'Beta2', 'Gamma1', 'Gamma2', 'Attention',
      'class'],axis=1,inplace=True)
print(df.columns)

#delete duplicate rows
df.drop_duplicates(inplace=True)
#save the dataframe to a csv file
df.to_csv('/home/sury/proj/internship/dataset/nurse_processed_data.csv', index=False)
#correlation matrix
corr = df.corr()
#plot the correlation matrix
plt.figure(figsize=(10, 8))
plt.matshow(corr, cmap='coolwarm')
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title('Correlation Matrix', pad=20)
plt.tight_layout()
plt.savefig('/home/sury/proj/internship/dataset/correlation_matrix.png')
#show the plot
plt.show()