# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 18:23:26 2018

@author: Shivam Soni
"""

import numpy as np
import pandas as pd
import pandas_profiling
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

df=pd.read_excel("ENB2012_data.xlsx")

#pandas_profiling.ProfileReport(df)

print(df.describe())
print(df.shape)
print(df.dtypes)



    
    
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

for i in df.columns:
    df[i]=df[i].replace(" ",np.NaN)
    
df.isnull().sum()



df.columns = ['relative_compactness', 'surface_area', 'wall_area', 'roof_area', 'overall_height',
                'orientation', 'glazing_area', 'glazing_area_distribution', 'heating_load', 'cooling_load']

X=df.drop(['heating_load', 'cooling_load'],1)
Y=df[['heating_load', 'cooling_load']]
Y1=df[['heating_load']]
Y2=df[['cooling_load']]


for c_n in df.columns:
    #print c_n
   # if X[c_n]=='object' :
    unique_cat=df[c_n].nunique()
    print ("Feature", c_n,"has", unique_cat,"unique categories")
    
to_dummy=['orientation', 'glazing_area_distribution']

X_org=X.copy
for i in to_dummy:
    dummies= pd.get_dummies(X[i],prefix=i)
    #print dummies
    #dummies=dummies.iloc[:,1:]
    X=X.drop(i,1)
    X=pd.concat([dummies,X],axis=1)
    
num_cols=[i for i in df.columns if i not in to_dummy + ['heating_load', 'cooling_load']]
temp=[i for i in X if i not in num_cols]
x_dummies=X[temp]


std = StandardScaler()
x_transformed= std.fit_transform(X[num_cols])
x_transformed_df = pd.DataFrame(x_transformed,columns=num_cols)
 

X=X.drop(num_cols,1)
X=pd.concat([X.reset_index(drop=True),x_transformed_df.reset_index(drop=True) ],axis=1)




pca = PCA(n_components=5)
#X=X.T

fit = pca.fit_transform(X)
pca_data = pd.DataFrame(fit)
# summarize components
print(pca.explained_variance_ratio_)
#print(pca.singular_values_)



# summarize components

tota = 0
for i in range(len(pca.explained_variance_ratio_)):
    tota += pca.explained_variance_ratio_[i]
print('Variance:', tota)
print(pca.singular_values_)


'''x_train, x_test, y_train, y_test = train_test_split(pca_data, Y, test_size=0.20, random_state=0)
x1_train, x1_test, y1_train, y1_test = train_test_split(pca_data, Y1, test_size=0.20, random_state=0)
x2_train, x2_test, y2_train, y2_test = train_test_split(pca_data, Y2, test_size=0.20, random_state=0)'''



    
avg_r2_train=[]
avg_r2_test=[]
avg_meanS_train=[]
avg_meanS_test=[]

avg_meanA_train=[]
avg_meanA_test=[] 

from sklearn.model_selection import KFold
kf = KFold(n_splits=10)
i=1.0
for train_index, test_index in kf.split(pca_data):
    print('Fold',i,'--------------------------------------')
    x_train=pca_data.iloc[train_index]
    #print(x_train.shape)
    y_train=Y.iloc[train_index]
    #print(y_train.shape)
    x_test=pca_data.iloc[test_index]
    y_test=Y.iloc[test_index]
    i=i+1
    
    reg = MLPRegressor(solver='adam', alpha=1e-5,hidden_layer_sizes=(25,50,100), random_state=0)
    m=reg.fit(x_train,y_train)
    pred1=m.predict(x_train)
    pred=m.predict(x_test)
   
    trainr2=r2_score(y_train,pred1)
    testr2=r2_score(y_test,pred)
    print('train_r2=',trainr2)
    print('test_r2=',testr2)
    avg_r2_train.append(trainr2)
    avg_r2_test.append(testr2)
    
    temp_train_MS=mean_squared_error(y_train,pred1)
    temp_test_MS=mean_squared_error(y_test,pred)
    print('TRAIN Mean Squared Error',temp_train_MS)
    print('TEST Mean Squared Error',temp_test_MS)
    avg_meanS_train.append(temp_train_MS)
    avg_meanS_test.append(temp_test_MS)
    print('\n')
    

    temp_train_A=mean_absolute_error(y_train,pred1)
    temp_test_A= mean_absolute_error(y_test,pred)
    print('TRAIN Mean Absolute Error',temp_train_A)
    print('TEST Mean Absolute Error',temp_test_A)
    avg_meanA_train.append(temp_train_A)
    avg_meanA_test.append(temp_test_A)
    print('\n')
    
   
print('Avg r2_train',np.mean(avg_r2_train))
print('Avg r2_test',np.mean(avg_r2_test))

print('Avg TRAIN Mean Squared Error',np.mean(avg_meanS_train))
print('Avg TEST Mean Squared Error',np.mean(avg_meanS_test))


print('Avg MAE',np.mean(avg_meanA_train))
print('Avg MAE',np.mean(avg_meanA_test))














