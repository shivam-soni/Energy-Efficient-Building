# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 13:07:42 2018

@author: Shivam Soni
"""


import numpy as np
import pandas as pd
import pandas_profiling





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
Y1=df[['heating_load']]
Y2=df[['cooling_load']]

to_dummy=['orientation','glazing_area', 'glazing_area_distribution']

X_org=X.copy
for i in to_dummy:
    dummies= pd.get_dummies(X[i],prefix=i)
    #print dummies
    dummies=dummies.iloc[:,1:]
    X=X.drop(i,1)
    X=pd.concat([dummies,X],axis=1)
    
num_cols=[i for i in df.columns if i not in to_dummy + ['heating_load', 'cooling_load']]
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)


from sklearn.preprocessing import StandardScaler
# Fitting parameters for scaling x_train for the numerical columns 
std = StandardScaler().fit(x_train[num_cols])
#Transforming the the original data
x_train_transformed=std.transform(x_train[num_cols])
x_train_transformed_df = pd.DataFrame(x_train_transformed,columns=num_cols)
x_train=x_train.drop(num_cols,1)
x_train=pd.concat([x_train.reset_index(drop=True),x_train_transformed_df.reset_index(drop=True) ],axis=1)

#Transforming the the test data's numerical with x_train scaled parameters
x_test_transformed=std.transform(x_test[num_cols])
x_test_transformed_df = pd.DataFrame(x_test_transformed,columns=num_cols)
x_test=x_test.drop(num_cols,1)
x_test=pd.concat([x_test.reset_index(drop=True),x_test_transformed_df .reset_index(drop=True) ],axis=1)    
    
    