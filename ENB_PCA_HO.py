# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 15:42:55 2018

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


x_train, x_test, y_train, y_test = train_test_split(pca_data, Y, test_size=0.20, random_state=0)
x1_train, x1_test, y1_train, y1_test = train_test_split(pca_data, Y1, test_size=0.20, random_state=0)
x2_train, x2_test, y2_train, y2_test = train_test_split(pca_data, Y2, test_size=0.20, random_state=0)


print("\n<----------------------------------MLP REGRESSOR------------------------------->\n")

reg=MLPRegressor()

m=reg.fit(x_train,y_train)
pred1=m.predict(x_train)
pred=m.predict(x_test)

trainr2=r2_score(y_train,pred1)
testr2=r2_score(y_test,pred)
print('train_r2=',trainr2)
print('test_r2=',testr2)

from sklearn.metrics import mean_squared_error
print("TRAIN mean_squared_error= ",mean_squared_error(y_train,pred1))
print("TEST mean_squared_error= ",mean_squared_error(y_test,pred))

print("<\n-----------------------------MULTIPLE LINEAR REGRESSION------------------------->\n")


regressor=LinearRegression()
regressor.fit(x1_train,y1_train )

y1_train_pred= regressor.predict(x1_train)
y1_pred= regressor.predict(x1_test)

from sklearn.metrics import r2_score

print("*FOR heating_load(Y1)\n" )
print("TRAIN r2= ",r2_score(y1_train,y1_train_pred))
print("TEST r2= ",r2_score(y1_test,y1_pred))

from sklearn.metrics import mean_squared_error
print("TRAIN mean_squared_error= ",mean_squared_error(y1_train,y1_train_pred))
print("TEST mean_squared_error= ",mean_squared_error(y1_test,y1_pred))


#FOR Y2
regressor.fit(x2_train,y2_train )

y2_train_pred= regressor.predict(x2_train)
y2_pred= regressor.predict(x2_test)

print("*FOR cooling_load(Y2)\n" )
print("TRAIN r2= ",r2_score(y2_train,y2_train_pred))
print("TEST r2= ",r2_score(y2_test,y2_pred))

print("TRAIN mean_squared_error= ",mean_squared_error(y2_train,y2_train_pred))
print("TEST mean_squared_error= ",mean_squared_error(y2_test,y2_pred))


print("\n<------------------------POLYNOMIAL LINEAR REGRESSION-------------------------->\n\n\n")



from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x1_poly = poly_reg.fit_transform(x1_train)
lin_reg2= LinearRegression()
lin_reg2.fit(x1_poly,y1_train)
x1_test_poly=poly_reg.fit_transform(x1_test)

y1_train_pred_po= lin_reg2.predict(x1_poly)
y1_pred_po= lin_reg2.predict(x1_test_poly)


from sklearn.metrics import r2_score

print("*FOR heating_load(Y1)\n" )
print("TRAIN r2= ",r2_score(y1_train,y1_train_pred_po))
print("TEST r2= ",r2_score(y1_test,y1_pred_po))


from sklearn.metrics import mean_squared_error
print("TRAIN mean_squared_error= ",mean_squared_error(y1_train,y1_train_pred_po))
print("TEST mean_squared_error= ",mean_squared_error(y1_test,y1_pred_po))

#FOR Y2

x2_poly = poly_reg.fit_transform(x2_train)
lin_reg2= LinearRegression()
lin_reg2.fit(x2_poly,y2_train)
x2_test_poly=poly_reg.fit_transform(x2_test)

y2_train_pred_po= lin_reg2.predict(x2_poly)
y2_pred_po= lin_reg2.predict(x2_test_poly)


print("*FOR cooling_load(Y2)\n" )

print("TRAIN r2= ",r2_score(y2_train,y2_train_pred_po))
print("TEST r2= ",r2_score(y2_test,y2_pred_po))


print("TRAIN mean_squared_error= ",mean_squared_error(y2_train,y2_train_pred_po))
print("TEST mean_squared_error= ",mean_squared_error(y2_test,y2_pred_po))

print("\n<------------------------LASSO REGRESSION-------------------------->\n\n\n")
predictors = x1_train.columns

from sklearn.linear_model import Lasso

lassoReg = Lasso(alpha=0.0003, normalize=True)
lassoReg.fit(x1_train,y1_train)

y1_lassopred_train=lassoReg.predict(x1_train)
y1_lassopred_test = lassoReg.predict(x1_test)

print("*FOR heating_load(Y1)\n" )
print("TRAIN r2= ",r2_score(y1_train,y1_lassopred_train))
print("TEST r2= ",r2_score(y1_test,y1_lassopred_test ))


from sklearn.metrics import mean_squared_error
print("TRAIN mean_squared_error= ",mean_squared_error(y1_train,y1_lassopred_train))
print("TEST mean_squared_error= ",mean_squared_error(y1_test,y1_lassopred_test ))

#for y2

lassoReg.fit(x2_train,y2_train)

y2_lassopred_train=lassoReg.predict(x2_train)
y2_lassopred_test = lassoReg.predict(x2_test)

print("*FOR cooling_load(Y2)\n" )
print("TRAIN r2= ",r2_score(y2_train,y2_lassopred_train))
print("TEST r2= ",r2_score(y2_test,y2_lassopred_test ))


print("TRAIN mean_squared_error= ",mean_squared_error(y2_train,y2_lassopred_train))
print("TEST mean_squared_error= ",mean_squared_error(y2_test,y2_lassopred_test ))

print("\n<------------------------RIDGE REGRESSION-------------------------->\n\n\n")


from sklearn.linear_model import Ridge

RidgeReg = Ridge(alpha=0.0003, normalize=True)
RidgeReg.fit(x1_train,y1_train)

y1_Ridgepred_train=RidgeReg.predict(x1_train)
y1_Ridgepred_test = RidgeReg.predict(x1_test)

print("*FOR heating_load(Y1)\n" )
print("TRAIN r2= ",r2_score(y1_train,y1_Ridgepred_train))
print("TEST r2= ",r2_score(y1_test,y1_Ridgepred_test ))


from sklearn.metrics import mean_squared_error
print("TRAIN mean_squared_error= ",mean_squared_error(y1_train,y1_Ridgepred_train))
print("TEST mean_squared_error= ",mean_squared_error(y1_test,y1_Ridgepred_test ))   


#for y2

RidgeReg.fit(x2_train,y2_train)

y2_Ridgepred_train=RidgeReg.predict(x2_train)
y2_Ridgepred_test = RidgeReg.predict(x2_test)

print("*FOR cooling_load(Y2)\n" )
print("TRAIN r2= ",r2_score(y2_train,y2_Ridgepred_train))
print("TEST r2= ",r2_score(y2_test,y2_Ridgepred_test ))


from sklearn.metrics import mean_squared_error
print("TRAIN mean_squared_error= ",mean_squared_error(y2_train,y2_Ridgepred_train))
print("TEST mean_squared_error= ",mean_squared_error(y2_test,y2_Ridgepred_test ))











