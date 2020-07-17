# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 15:32:47 2018

@author: Shivam Soni
"""

import numpy as np
import pandas as pd
import pandas_profiling

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

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

to_dummy=['orientation', 'glazing_area_distribution']

X_org=X.copy
for i in to_dummy:
    dummies= pd.get_dummies(X[i],prefix=i)
    #print dummies
    #dummies=dummies.iloc[:,1:]
    X=X.drop(i,1)
    X=pd.concat([dummies,X],axis=1)
    
num_cols=[i for i in df.columns if i not in to_dummy + ['heating_load', 'cooling_load']]



from sklearn.model_selection import KFold


kf = KFold(n_splits=10)
print("Using ",kf.get_n_splits(X)," folds")





def Model(regressor,f):
    
    avg_r2_train=[]
    avg_r2_test=[]
    avg_meanS_train=[]
    avg_meanS_test=[]
    
    avg_r2_train_y2=[]
    avg_r2_test_y2=[]
    avg_meanS_train_y2=[]
    avg_meanS_test_y2=[]
    from sklearn.preprocessing import StandardScaler
    i=1.0
    for train_index, test_index in kf.split(X ):
        print("FOLD ",i,"-------------------------------------------------------------------------")
        x_train=X.iloc[train_index]
        y1_train=Y1.iloc[train_index]
        y2_train=Y2.iloc[train_index]
        
        x_test=X.iloc[test_index]
        y1_test=Y1.iloc[test_index]
        y2_test=Y2.loc[test_index]
        
        std = StandardScaler().fit(x_train[num_cols])
        #Transforming the the original data
        x_train_transformed=std.transform(x_train[num_cols])
        x_train_transformed_df = pd.DataFrame(x_train_transformed,columns=num_cols)
        x_train1=x_train.drop(num_cols,1)
        x_train=pd.concat([x_train1.reset_index(drop=True),x_train_transformed_df.reset_index(drop=True) ],axis=1)
        
        #Transforming the the test data's numerical with x_train scaled parameters
        x_test_transformed=std.transform(x_test[num_cols])
        x_test_transformed_df = pd.DataFrame(x_test_transformed,columns=num_cols)
        x_test1=x_test.drop(num_cols,1)
        x_test=pd.concat([x_test1.reset_index(drop=True),x_test_transformed_df .reset_index(drop=True) ],axis=1)  
        
        
        if(f ==0 ):
             
            regressor.fit(x_train,y1_train )
            y1_train_pred= regressor.predict(x_train)
            y1_pred= regressor.predict(x_test)
            print("For Y1:")
            temp_train_r2=r2_score(y1_train,y1_train_pred)
            temp_test_r2=r2_score(y1_test,y1_pred)
            print('r2_train',temp_train_r2)
            print('r2_test',temp_test_r2)
            avg_r2_train.append(temp_train_r2)
            avg_r2_test.append(temp_test_r2)
            
            temp_train_MS=mean_squared_error(y1_train,y1_train_pred)
            temp_test_MS=mean_squared_error(y1_test,y1_pred)
            print('TRAIN Mean Squared Error',temp_train_MS)
            print('TEST Mean Squared Error',temp_test_MS)
            avg_meanS_train.append(temp_train_MS)
            avg_meanS_test.append(temp_test_MS)
            print('\n')
            
            print("For Y2:")
            
            regressor.fit(x_train,y2_train )
            y2_train_pred= regressor.predict(x_train)
            y2_pred= regressor.predict(x_test)
            
            temp_train_r2=r2_score(y2_train,y2_train_pred)
            temp_test_r2=r2_score(y2_test,y2_pred)
            print('r2_train',temp_train_r2)
            print('r2_test',temp_test_r2)
            avg_r2_train_y2.append(temp_train_r2)
            avg_r2_test_y2.append(temp_test_r2)
            
            temp_train_MS=mean_squared_error(y1_train,y1_train_pred)
            temp_test_MS=mean_squared_error(y1_test,y1_pred)
            print('TRAIN Mean Squared Error',temp_train_MS)
            print('TEST Mean Squared Error',temp_test_MS)
            avg_meanS_train_y2.append(temp_train_MS)
            avg_meanS_test_y2.append(temp_test_MS)
            
            print('\n')
            
        else:
            
            x1_poly = regressor.fit_transform(x_train)
            lin_reg2= LinearRegression()
            lin_reg2.fit(x1_poly,y1_train)
            x1_test_poly=regressor.fit_transform(x_test)
            y1_train_pred_po= lin_reg2.predict(x1_poly)
            y1_pred_po= lin_reg2.predict(x1_test_poly)
            
            print("*FOR heating_load(Y1)\n" )
            
            temp_train_r2=r2_score(y1_train,y1_train_pred_po)
            temp_test_r2=r2_score(y1_test,y1_pred_po)
            print("TRAIN r2= ",temp_train_r2)
            print("TEST r2= ", temp_test_r2) 
            avg_r2_train.append(temp_train_r2)
            avg_r2_test.append(temp_test_r2)
            
            temp_train_MS=mean_squared_error(y1_train,y1_train_pred_po)
            temp_test_MS=mean_squared_error(y1_test,y1_pred_po)
            print('TRAIN Mean Squared Error',temp_train_MS)
            print('TEST Mean Squared Error',temp_test_MS)
            avg_meanS_train.append(temp_train_MS)
            avg_meanS_test.append(temp_test_MS)
            print('\n')
            
            #for y2
            x2_poly = regressor.fit_transform(x_train)
            lin_reg2= LinearRegression()
            lin_reg2.fit(x2_poly,y1_train)
            x2_test_poly=regressor.fit_transform(x_test)
            y2_train_pred_po= lin_reg2.predict(x2_poly)
            y2_pred_po= lin_reg2.predict(x2_test_poly)
            
            print("*FOR cooling_load(Y2)\n" )
            
            temp_train_r2=r2_score(y2_train,y2_train_pred_po)
            temp_test_r2=r2_score(y1_test,y2_pred_po)
            print("TRAIN r2= ",temp_train_r2)
            print("TEST r2= ", temp_test_r2) 
            avg_r2_train_y2.append(temp_train_r2)
            avg_r2_test_y2.append(temp_test_r2)
            
            temp_train_MS=mean_squared_error(y1_train,y1_train_pred_po)
            temp_test_MS=mean_squared_error(y1_test,y1_pred_po)
            print('TRAIN Mean Squared Error',temp_train_MS)
            print('TEST Mean Squared Error',temp_test_MS)
            avg_meanS_train_y2.append(temp_train_MS)
            avg_meanS_test_y2.append(temp_test_MS)
            print('\n')
        i=i+1
        
    print("For Y1")
    print('Avg r2_train',np.mean(avg_r2_train))
    print('Avg r2_test',np.mean(avg_r2_test))
        
    print('Avg TRAIN Mean Squared Error',np.mean(avg_meanS_train))
    print('Avg TEST Mean Squared Error',np.mean(avg_meanS_test))
        
    print('\n')
    print("For Y2")
    print('Avg r2_train',np.mean(avg_r2_train_y2))
    print('Avg r2_test',np.mean(avg_r2_test_y2))
        
    print('Avg TRAIN Mean Squared Error',np.mean(avg_meanS_train_y2))
    print('Avg TEST Mean Squared Error',np.mean(avg_meanS_test_y2))

print("\n<------------------------MULTIPLE LINEAR REGRESSION------------------------->\n\n\n")

#Model(LinearRegression(),0)

print("\n<------------------------POLYNOMIAL LINEAR REGRESSION------------------------->\n\n\n")

#Model(PolynomialFeatures(degree=2),1)

print("\n<------------------------LASSO REGRESSION-------------------------->\n\n\n")

#Model(Lasso(alpha=0.0003, normalize=True),0)

print("\n<------------------------RIDGE REGRESSION-------------------------->\n\n\n")

Model(Ridge(alpha=0.0003, normalize=True),0)
