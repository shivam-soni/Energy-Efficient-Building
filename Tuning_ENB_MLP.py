# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 11:39:14 2018

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
Y=df[['heating_load', 'cooling_load']]

plt.figure(figsize=(12,12))
sns.heatmap(df.corr(),annot=True)

from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

from sklearn.preprocessing import StandardScaler  

scaler = StandardScaler()  
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

'''from sklearn import preprocessing

normalized_X = preprocessing.normalize(X)
normalized_X = pd.DataFrame(normalized_X,columns=['X1','X2','X3','X4','X5','X6','X7','X8'])
normalized_X.isnull().sum()'''


from sklearn.neural_network import MLPRegressor

reg=MLPRegressor()

'''reg = MLPRegressor(solver='adam', alpha=1e-5,hidden_layer_sizes=(25,50,100), random_state=1)
reg.fit(x_train,y_train)

pred=reg.predict(x_test)

from sklearn.metrics import r2_score
print(r2_score(y_test,pred))
#print("Y1: \ntrain r2: {} \ntest r2: {}".format(reg.score(x_train, y_train),reg.score(x_test, y_test)))'''

from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(reg, param_grid={'solver': ['lbfgs','sgd','adam'],'alpha': 10.0 ** -np.arange(1, 7),'learning_rate':['constant'], 'learning_rate_init': [0.001],'hidden_layer_sizes': [25,50,100],'activation': ['identity','relu','logistic', 'tanh']})

gs.fit(x_train,y_train)


means = gs.cv_results_['mean_test_score']
stds = gs.cv_results_['std_test_score']
c=0
for mean, std, params in zip(means, stds, gs.cv_results_['params']):
    c=c+1
    print("%d) %0.3f (+/-%0.03f) for %r"% (c,mean, std * 2, params))

print("Best score: %0.4f" % gs.best_score_)

print("Best parameters:\n",gs.best_params_)


print("Best estimator:\n",gs.best_estimator_)



mlp=gs.best_estimator_

model= mlp.fit(x_train,y_train)

pred1=model.predict(x_train)

pred=model.predict(x_test)

from sklearn.metrics import r2_score
print("TRAIN r2= ",r2_score(y_train,pred1))
print("TEST r2= ",r2_score(y_test,pred))

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.ion()
plt.figure(figsize=(5,5))
sns.pairplot(data=df, y_vars=['cooling_load','heating_load'],
             x_vars=['relative_compactness', 'surface_area', 'wall_area', 'roof_area', 'overall_height',
                     'orientation', 'glazing_area', 'glazing_area_distribution'])

    
    
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_train,pred1))
mean_squared_error(y_test,pred)

'''plt.hist(df.wall_area, color='#86bf91',density=True,rwidth=0.9,stacked=True,bins=15)
plt.xlabel('wall_area')
plt.ylabel('Frequency')
plt.show()


var = df.groupby('relative_compactness').relative_compactness.sum()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('relative_compactness')
ax1.set_ylabel('Frequency')
ax1.set_title("Frequency of relative_compactness")
var.plot(kind='line')'''




