# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 22:34:18 2018

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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

df=pd.read_excel("ENB2012_data.xlsx")

#pandas_profiling.ProfileReport(df)

print(df.describe())
print(df.shape)
print(df.dtypes)



    

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


    
pca = PCA(n_components=3)
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
    
    i=1.0
    for train_index, test_index in kf.split(pca_data ):
        print(i,"-------------------------------------------------------------------------")
        x_train=pca_data.iloc[train_index]
        print("x-train",x_train.shape)
        y1_train=Y1.iloc[train_index]
        print("y1-train",y1_train.shape)
        y2_train=Y2.iloc[train_index]
        print("y2-train",y2_train.shape)
        
        
        x_test=pca_data.iloc[test_index]
        print("x-test",x_test.shape)
        y1_test=Y1.iloc[test_index]
        print("y1-test",y1_test.shape)
        y2_test=Y2.loc[test_index]
        print("y2-test",y2_test.shape)
        
        
        
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

Model(LinearRegression(),0)

print("\n<------------------------POLYNOMIAL LINEAR REGRESSION------------------------->\n\n\n")

Model(PolynomialFeatures(degree=3),1)

print("\n<------------------------LASSO REGRESSION-------------------------->\n\n\n")

Model(Lasso(alpha=0.0003, normalize=True),0)

print("\n<------------------------RIDGE REGRESSION-------------------------->\n\n\n")

Model(Ridge(alpha=0.0003, normalize=True),0)



    
    

    

    
    
    
    
    
    
    
    
    
    
    
    