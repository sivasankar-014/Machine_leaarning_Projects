
    # # CANCER PATIENT PREDICTION USING CLASSIFICATION


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('lung cancer survey.csv')
# print(data.head())
# print(data.info())
# print(data.describe())
# print(data.isnull().sum())

# 'below condition used for finding imbalance'
# print(data['LUNG_CANCER'].value_counts())
# sns.countplot(x=data['LUNG_CANCER'])
# plt.show()

# covert text column into number

from sklearn.preprocessing import LabelEncoder,StandardScaler
lung_encoder=LabelEncoder()
gender_encoder=LabelEncoder()
data['LUNG_CANCER']=lung_encoder.fit_transform(data['LUNG_CANCER'])
data['GENDER']=gender_encoder.fit_transform(data['GENDER'])
# print(data["LUNG_CANCER"])

# plt.plot(data['LUNG_CANCER'])
# plt.show()

# feature selection

x=data.drop('LUNG_CANCER',axis=1)
y=data['LUNG_CANCER']

#    find correlation
# plt.figure(figsize=(10,20))
# sns.heatmap(data.corr(),annot=True,cmap='coolwarm')
# plt.show()

#       check how age distributed
# sns.histplot(data['AGE'])
# plt.show()
# #      smoking vs lungcancer
# sns.countplot(x=data['SMOKING'],hue=data['LUNG_CANCER'])
# plt.show()

from sklearn.model_selection import train_test_split


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


#  CHECK Data imbalancing
# print(y_train.value_counts())# imbalancing occur
# from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTE
sm=SMOTE()
x_train_sm,y_train_sm=sm.fit_resample(x_train,y_train)
# print(y_train_sm.value_counts())

# scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_scaler=scaler.fit_transform(x_train_sm)
x_test_scaler=scaler.transform(x_test)

from sklearn.linear_model import LogisticRegression

model=LogisticRegression()
model.fit(x_train_scaler,y_train_sm)

y_predict=model.predict(x_test_scaler)
# print(type(x_test))
print(y_predict)
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,f1_score, classification_report

# print(accuracy_score(y_test,y_predict))
# print('confusion matrix',confusion_matrix(y_test,y_predict))
# print('classification report: ',classification_report(y_test,y_predict))


# print(data.columns)

new_data=pd.DataFrame({'GENDER':['M','F','M','F'],
          'AGE':[19,30,25,60],
          'SMOKING':[1,2,1,2],
          'YELLOW_FINGERS':[1,2,1,2],
          'ANXIETY':[2,1,2,2],
          'PEER_PRESSURE':[1,1,2,1],
          'CHRONIC DISEASE':[1,1,2,2],
          'FATIGUE ':[1,2,1,1],
          'ALLERGY ':[1,1,1,1],
          'WHEEZING':[2,2,2,1],
           'ALCOHOL CONSUMING':[1,1,1,2],
           'COUGHING':[2,1,1,2],
           'SHORTNESS OF BREATH':[1,2,1,2],
       'SWALLOWING DIFFICULTY':[1,1,2,2],
       'CHEST PAIN':[2,1,1,2]
    
})

new_data['GENDER']=gender_encoder.transform(new_data['GENDER'])
new_data_scaler=scaler.transform(new_data)
y_predict1=model.predict(new_data_scaler)
print(y_predict1)

print('classification report: ',classification_report(y_test,y_predict))
