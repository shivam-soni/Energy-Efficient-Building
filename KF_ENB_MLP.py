
import numpy as np
import pandas as pd
import pandas_profiling
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

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

'''plt.figure(figsize=(12,12))
sns.heatmap(df.corr(),annot=True)'''

from sklearn.model_selection import KFold

from sklearn.neural_network import MLPRegressor

reg=MLPRegressor(activation='logistic', alpha=0.0001, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=100, learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=None, shuffle=True, solver='lbfgs', tol=0.0001,
       validation_fraction=0.1, verbose=False, warm_start=False)
kf = KFold(n_splits=10)
print("Using ",kf.get_n_splits(X)," folds")

from sklearn.metrics import r2_score
avg_r2_train=[]
avg_r2_test=[]
avg_meanS_train=[]
avg_meanS_test=[]
avg_meanA_train=[]
avg_meanA_test=[]
i=1.0
for train_index, test_index in kf.split(X ):
    print(i)
    x_train=X.iloc[train_index]
    print(x_train.shape)
    #print("x_train",x_train.shape)
    y_train=Y.iloc[train_index]
    print(y_train.shape)
    #print('y_train',y_train.shape)
    x_test=X.iloc[test_index]
    #print('x_test',x_test.shape)
    y_test=Y.iloc[test_index]
    #print('y_test',y_test.shape)
    i=i+1
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
    print('\n')
    temp_train_MS = mean_squared_error(y_train,pred1)
    temp_test_MS = mean_squared_error(y_test,pred)
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
    
print('Avg MSE',np.mean(avg_meanS_train))
print('Avg MSE',np.mean(avg_meanS_test))
        
print('Avg MAE',np.mean(avg_meanA_train))
print('Avg MAE',np.mean(avg_meanA_test))
        
    
    
    
    
    
    
