
      ##     Water Quality prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv(r'C:\Users\admin\Desktop\python interview questions\projects\water_potability.csv')
print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())

data['ph']=data['ph'].fillna(data['ph'].median())
data['Sulfate']=data['Sulfate'].fillna(data['Sulfate'].median())
data['Trihalomethanes']=data['Trihalomethanes'].fillna(data['Trihalomethanes'].median())
# print(data.isnull().sum())
selected_columns=[]
for column in data.columns:
 if data[column].dtypes == float:
#  if data[column].dtype in [float,int]:
       selected_columns.append(column)
       
print(selected_columns)
for column in selected_columns:   
 plt.figure(figsize=(8,6))
 sns.histplot(data[column],kde=True)
 plt.title(column)
 plt.show()

for column in selected_columns:
    plt.figure(figsize=(8,6))
    plt.boxplot(data[column])
    plt.title(column)
    plt.show()

x=data.drop('Potability',axis=1)
y=data['Potability']


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

from sklearn.linear_model import LogisticRegression,Ridge,Lasso
model=LogisticRegression()
model.fit(x_train,y_train)

y_predict=model.predict(x_test)

print(y_predict)

from sklearn.metrics import mean_absolute_error,r2_score,accuracy_score


print('MAE:',mean_absolute_error(y_test,y_predict))
print('r2_score:',r2_score(y_test,y_predict))

print(accuracy_score(y_test,y_predict))
ridge=Ridge(alpha=10)
lasso=Lasso(alpha=0.1)

ridge.fit(x_train,y_train)
lasso.fit(x_train,y_train)
