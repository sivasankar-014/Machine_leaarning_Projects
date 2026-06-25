# ## import libraries
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt 
# import seaborn as sns

# ## Load dataset

# data=pd.read_csv(r'C:\Users\admin\Desktop\python interview questions\projects\diabetes_prediction_dataset.csv')

# print(data.head())


# # print(data.info())
# # print(data.describe())

# ## Data preprocessing

# from sklearn.preprocessing import LabelEncoder
# le=LabelEncoder()
# data['gender']=le.fit_transform(data['gender'])
# data['smoking_history']=le.fit_transform(data['smoking_history'])
# for column in data.select_dtypes(include=np.number).columns:
#     sns.boxplot(x=data[column])
#     plt.title(column)
#     plt.show()
# print(data.dtypes)
# sns.heatmap(data.corr(),annot=True)
# plt.show()

# ## Feature selection

# x=data.drop('diabetes',axis=1)
# y=data['diabetes']



# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score,classification_report
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()

# def check_accuracy(x,y):
#       x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
      
#       x_train_scaled=scaler.fit_transform(x_train)
#       x_test_scaled=scaler.transform(x_test)

      
#       algorithms={'Logistic Regression':LogisticRegression(),
#                   'kNeighborsClassifier':KNeighborsClassifier(),
#                   'DecisionTreeClassifier':DecisionTreeClassifier(),
#                   'RandomforestClassifier':RandomForestClassifier(),
#                   'SVC':SVC()}
#       for name,model in algorithms.items():
#             model.fit(x_train_scaled,y_train)
#             predict=model.predict(x_test_scaled)
#             accuracy=accuracy_score(y_test,predict)
#             cls_report=classification_report(y_test,predict)
#             print(f'{name} accuracy:{accuracy}')
#             print(f'{name} classification report: {cls_report}')
# check_accuracy(x,y)
  
  
n=12

if(n>1):
    for i in range(2,n):
        if n % i == 0:
            print(n,' is not a prime number')
            break
    else:
         print(n,'is a prime no')
else:
    print(n,'is not a prime number')
    