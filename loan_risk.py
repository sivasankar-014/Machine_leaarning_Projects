
############  TO PREDICT LOAN APPROVAL #############

## import libraries

import pandas as pd 
import numpy as np

from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


##  Load dataset

data=pd.read_csv(r'C:\Users\admin\Desktop\python interview questions\projects\loan_risk_prediction_dataset.csv') 

# print(data.head())

## 0-not approved, 1-approved
## preprocessing

# print(data.info())
# print(data.describe())

##filling null value

data['Income']=data['Income'].fillna(data['Income'].mean())  
data['CreditScore']=data['CreditScore'].fillna(data['CreditScore'].mean())
data['Education']=data['Education'].fillna(data['Education'].mode()[0])
# print(data.isnull().sum())   

## covert text into numeric
        
le=LabelEncoder()
data['Gender']=le.fit_transform(data['Gender'])
data['Education']=le.fit_transform(data['Education'])
data['City']=le.fit_transform(data['City'])
data['EmploymentType']=le.fit_transform(data['EmploymentType'])

# print(data.head())   

## feature selection

x=data.drop("LoanApproved",axis=1) 
y=data['LoanApproved']


## split data

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

##  standard scaler

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)    
# print(x_train.shape)


## Build model ANN

model=Sequential()

model.add(Dense(32,activation='relu',input_shape=(x_train.shape[1],)))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


## Compile model

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


## Early stopping

EarlyStop=EarlyStopping(monitor='val_loss',
                        patience=5,
                        restore_best_weights=True)

## Train model

model.fit(x_train,y_train,
          epochs=10,
          batch_size=32,
          validation_split=0.2,
          callbacks=[EarlyStop],
          verbose=1)
        
## evaluate model

loss,accuracy=model.evaluate(x_test,y_test,verbose=1)

print('Accuracy: ',accuracy)

## predict new customer

# new_customer=np.array([[29,35000,25000,780,1,1,2,3,1]])

new_customer = pd.DataFrame({
    'Age': [45],
    'Income': [55000],
    'LoanAmount': [25000],
    'CreditScore': [720],
    'YearsExperience': [10],
    'Gender': [1],
    'Education': [1],
    'City': [2],
    'EmploymentType': [1]
})

new_customer=scaler.transform(new_customer)

prediction=model.predict(new_customer)
print(prediction)

if prediction > 0.5 :
    print('loan approved')
else:
    print('not approved')
        