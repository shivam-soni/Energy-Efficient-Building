# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 15:15:43 2018

@author: Shivam Soni
"""

import numpy as np
import pandas as pd
import pandas_profiling
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

df=pd.read_excel("ENB2012_data.xlsx")

#pandas_profiling.ProfileReport(df)

print(df.describe())
print(df.shape)
print(df.dtypes)



    

for i in df.columns:
    df[i]=df[i].replace(" ",np.NaN)
    
df.isnull().sum()



df.columns = ['relative_compactness', 'surface_area', 'wall_area', 'roof_area', 'overall_height',
                'orientation', 'glazing_area', 'glazing_area_distribution', 'heating_load', 'cooling_load']

X=df.drop(['heating_load', 'cooling_load'],1)
Y=df[['heating_load', 'cooling_load']]

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




from sklearn.neural_network import MLPRegressor


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)


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


std_y = StandardScaler().fit(y_train)
y_train_transformed = std_y.transform(y_train)
y_train_transformed_df = pd.DataFrame(y_train_transformed,columns=['heating_load','cooling_load'])
y_train1=y_train.drop(['heating_load','cooling_load'],1)
y_train=pd.concat([y_train1.reset_index(drop=True),y_train_transformed_df.reset_index(drop= True)],axis=1)

y_test_transformed = std_y.transform(y_test)
y_test_transformed_df = pd.DataFrame(y_test_transformed,columns=['heating_load','cooling_load'])
y_test1=y_test.drop(['heating_load','cooling_load'],1)
y_test=pd.concat([y_test1.reset_index(drop=True),y_test_transformed_df.reset_index(drop= True)],axis=1)



reg = MLPRegressor(activation='logistic', alpha=0.0001, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=100, learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=None, shuffle=True, solver='lbfgs', tol=0.0001,
       validation_fraction=0.1, verbose=False, warm_start=False)

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

print("TRAIN mean_absolute_error= ",mean_absolute_error(y_train,pred1))
print("TEST mean_absolute_error= ",mean_absolute_error(y_test,pred))



import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.ion()
plt.figure(figsize=(5,5))
sns.pairplot(data=df, y_vars=['cooling_load','heating_load'],x_vars=['relative_compactness', 'surface_area', 'wall_area', 'roof_area', 'overall_height','glazing_area'])


'''0, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'adam'}
151) 0.985 (+/-0.003) for {'activation': 'logistic', 'alpha': 1e-05, 'hidden_layer_sizes': 100, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'lbfgs'}
152) 0.893 (+/-0.003) for {'activation': 'logistic', 'alpha': 1e-05, 'hidden_layer_sizes': 100, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'sgd'}
153) 0.822 (+/-0.004) for {'activation': 'logistic', 'alpha': 1e-05, 'hidden_layer_sizes': 100, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'adam'}
154) 0.975 (+/-0.011) for {'activation': 'logistic', 'alpha': 1e-06, 'hidden_layer_sizes': 25, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'lbfgs'}
155) 0.900 (+/-0.002) for {'activation': 'logistic', 'alpha': 1e-06, 'hidden_layer_sizes': 25, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'sgd'}
156) -1.662 (+/-0.554) for {'activation': 'logistic', 'alpha': 1e-06, 'hidden_layer_sizes': 25, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'adam'}
157) 0.981 (+/-0.003) for {'activation': 'logistic', 'alpha': 1e-06, 'hidden_layer_sizes': 50, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'lbfgs'}
158) 0.897 (+/-0.004) for {'activation': 'logistic', 'alpha': 1e-06, 'hidden_layer_sizes': 50, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'sgd'}
159) 0.133 (+/-0.034) for {'activation': 'logistic', 'alpha': 1e-06, 'hidden_layer_sizes': 50, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'adam'}
160) 0.986 (+/-0.005) for {'activation': 'logistic', 'alpha': 1e-06, 'hidden_layer_sizes': 100, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'lbfgs'}
161) 0.894 (+/-0.003) for {'activation': 'logistic', 'alpha': 1e-06, 'hidden_layer_sizes': 100, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'sgd'}
162) 0.818 (+/-0.003) for {'activation': 'logistic', 'alpha': 1e-06, 'hidden_layer_sizes': 100, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'adam'}
163) 0.979 (+/-0.011) for {'activation': 'tanh', 'alpha': 0.1, 'hidden_layer_sizes': 25, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'lbfgs'}
164) 0.927 (+/-0.015) for {'activation': 'tanh', 'alpha': 0.1, 'hidden_layer_sizes': 25, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'sgd'}
165) -1.405 (+/-0.257) for {'activation': 'tanh', 'alpha': 0.1, 'hidden_layer_sizes': 25, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'adam'}
166) 0.987 (+/-0.001) for {'activation': 'tanh', 'alpha': 0.1, 'hidden_layer_sizes': 50, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'lbfgs'}
167) 0.927 (+/-0.019) for {'activation': 'tanh', 'alpha': 0.1, 'hidden_layer_sizes': 50, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'sgd'}
168) 0.384 (+/-0.073) for {'activation': 'tanh', 'alpha': 0.1, 'hidden_layer_sizes': 50, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'adam'}
169) 0.984 (+/-0.004) for {'activation': 'tanh', 'alpha': 0.1, 'hidden_layer_sizes': 100, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'lbfgs'}
170) 0.920 (+/-0.010) for {'activation': 'tanh', 'alpha': 0.1, 'hidden_layer_sizes': 100, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'sgd'}
171) 0.862 (+/-0.004) for {'activation': 'tanh', 'alpha': 0.1, 'hidden_layer_sizes': 100, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'adam'}
172) 0.983 (+/-0.009) for {'activation': 'tanh', 'alpha': 0.01, 'hidden_layer_sizes': 25, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'lbfgs'}
173) 0.929 (+/-0.007) for {'activation': 'tanh', 'alpha': 0.01, 'hidden_layer_sizes': 25, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'sgd'}
174) -1.016 (+/-0.584) for {'activation': 'tanh', 'alpha': 0.01, 'hidden_layer_sizes': 25, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'adam'}
175) 0.984 (+/-0.012) for {'activation': 'tanh', 'alpha': 0.01, 'hidden_layer_sizes': 50, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'lbfgs'}
176) 0.919 (+/-0.012) for {'activation': 'tanh', 'alpha': 0.01, 'hidden_layer_sizes': 50, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'sgd'}
177) 0.405 (+/-0.133) for {'activation': 'tanh', 'alpha': 0.01, 'hidden_layer_sizes': 50, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'adam'}
178) 0.986 (+/-0.003) for {'activation': 'tanh', 'alpha': 0.01, 'hidden_layer_sizes': 100, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'lbfgs'}
179) 0.907 (+/-0.012) for {'activation': 'tanh', 'alpha': 0.01, 'hidden_layer_sizes': 100, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'sgd'}
180) 0.862 (+/-0.019) for {'activation': 'tanh', 'alpha': 0.01, 'hidden_layer_sizes': 100, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'adam'}
181) 0.978 (+/-0.007) for {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': 25, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'lbfgs'}
182) 0.915 (+/-0.033) for {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': 25, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'sgd'}
183) -1.111 (+/-0.365) for {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': 25, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'adam'}
184) 0.984 (+/-0.009) for {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': 50, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'lbfgs'}
185) 0.920 (+/-0.026) for {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': 50, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'sgd'}
186) 0.395 (+/-0.160) for {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': 50, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'adam'}
187) 0.984 (+/-0.004) for {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': 100, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'lbfgs'}
188) 0.911 (+/-0.018) for {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': 100, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'sgd'}
189) 0.862 (+/-0.002) for {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': 100, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'adam'}
190) 0.981 (+/-0.013) for {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': 25, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'lbfgs'}
191) 0.925 (+/-0.016) for {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': 25, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'sgd'}
192) -1.237 (+/-0.749) for {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': 25, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'adam'}
193) 0.982 (+/-0.009) for {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': 50, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'lbfgs'}
194) 0.915 (+/-0.024) for {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': 50, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'sgd'}
195) 0.463 (+/-0.094) for {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': 50, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'adam'}
196) 0.984 (+/-0.002) for {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': 100, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'lbfgs'}
197) 0.919 (+/-0.021) for {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': 100, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'sgd'}
198) 0.860 (+/-0.007) for {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': 100, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'adam'}
199) 0.976 (+/-0.017) for {'activation': 'tanh', 'alpha': 1e-05, 'hidden_layer_sizes': 25, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'lbfgs'}
200) 0.922 (+/-0.037) for {'activation': 'tanh', 'alpha': 1e-05, 'hidden_layer_sizes': 25, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'sgd'}
201) -1.187 (+/-0.433) for {'activation': 'tanh', 'alpha': 1e-05, 'hidden_layer_sizes': 25, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'adam'}
202) 0.982 (+/-0.006) for {'activation': 'tanh', 'alpha': 1e-05, 'hidden_layer_sizes': 50, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'lbfgs'}
203) 0.924 (+/-0.025) for {'activation': 'tanh', 'alpha': 1e-05, 'hidden_layer_sizes': 50, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'sgd'}
204) 0.446 (+/-0.037) for {'activation': 'tanh', 'alpha': 1e-05, 'hidden_layer_sizes': 50, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'adam'}
205) 0.986 (+/-0.007) for {'activation': 'tanh', 'alpha': 1e-05, 'hidden_layer_sizes': 100, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'lbfgs'}
206) 0.909 (+/-0.019) for {'activation': 'tanh', 'alpha': 1e-05, 'hidden_layer_sizes': 100, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'sgd'}
207) 0.858 (+/-0.008) for {'activation': 'tanh', 'alpha': 1e-05, 'hidden_layer_sizes': 100, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'adam'}
208) 0.985 (+/-0.004) for {'activation': 'tanh', 'alpha': 1e-06, 'hidden_layer_sizes': 25, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'lbfgs'}
209) 0.917 (+/-0.012) for {'activation': 'tanh', 'alpha': 1e-06, 'hidden_layer_sizes': 25, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'sgd'}
210) -1.244 (+/-0.534) for {'activation': 'tanh', 'alpha': 1e-06, 'hidden_layer_sizes': 25, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'adam'}
211) 0.987 (+/-0.005) for {'activation': 'tanh', 'alpha': 1e-06, 'hidden_layer_sizes': 50, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'lbfgs'}
212) 0.927 (+/-0.003) for {'activation': 'tanh', 'alpha': 1e-06, 'hidden_layer_sizes': 50, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'sgd'}
213) 0.528 (+/-0.040) for {'activation': 'tanh', 'alpha': 1e-06, 'hidden_layer_sizes': 50, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'adam'}
214) 0.986 (+/-0.006) for {'activation': 'tanh', 'alpha': 1e-06, 'hidden_layer_sizes': 100, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'lbfgs'}
215) 0.913 (+/-0.015) for {'activation': 'tanh', 'alpha': 1e-06, 'hidden_layer_sizes': 100, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'sgd'}
216) 0.863 (+/-0.008) for {'activation': 'tanh', 'alpha': 1e-06, 'hidden_layer_sizes': 100, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'adam'}
Best score: 0.9883
Best parameters:
 {'activation': 'logistic', 'alpha': 0.0001, 'hidden_layer_sizes': 100, 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'lbfgs'}
Best estimator:
 MLPRegressor(activation='logistic', alpha=0.0001, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=100, learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=None, shuffle=True, solver='lbfgs', tol=0.0001,
       validation_fraction=0.1, verbose=False, warm_start=False)
TRAIN r2=  0.9976784718254885
TEST r2=  0.9915235331139262
0.2112939323311926'''









