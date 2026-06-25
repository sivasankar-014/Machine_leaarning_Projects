
#                 GOLD PRICE PREDICTION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#  load the data
data=pd.read_csv(r'C:\Users\admin\Desktop\python interview questions\projects\Gold (2).csv')
# print(data.head())
# print(data.info())
# print(data.columns)
#         find the missing value
# print(data.isnull().sum())
#         fill missing value
data['Volume']=data['Volume'].fillna(data['Volume'].mean())
# print(data.isnull().sum())
#      convert the date
data['Date']=pd.to_datetime(data['Date'])
# print(data.head())
# print(data.dtypes)

#          feature selection
x=data[['Volume','Open','High','Low']]
y=data['Close/Last']




#     model selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
scaler = StandardScaler()

def check_regression_models(x, y):
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    algorithms = {
        'Linear Regression': LinearRegression(),
        'Decision Tree Regressor': DecisionTreeRegressor(),
        'Random Forest Regressor': RandomForestRegressor()
        
    }

    for name, model in algorithms.items():
        
        model.fit(x_train_scaled, y_train)
        predictions = model.predict(x_test_scaled)

       
        r2 = r2_score(y_test, predictions)

        print(f"\n{name}")
        print("-" * 40)
        print(f"R²   : {r2:.4f}")

# Call function
check_regression_models(x, y)


# #    Visualization
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label='Actual')
plt.plot(predictions, label='Prediction')
plt.legend()
plt.xlabel("Data Points")
plt.ylabel("Gold Price")
plt.title("Actual vs Predicted Gold Prices")
plt.show()























# model=LinearRegression()
# model.fit(x_train,y_train)

# #    prediction or Test model

# y_predict=model.predict(x_test)

# #  model evaluation
# #      MAE
# from sklearn.metrics import mean_absolute_error
# mae=mean_absolute_error(y_test,y_predict)
# # print(mae)

# #     MSE
# from sklearn.metrics import mean_squared_error
# mse=mean_squared_error(y_test,y_predict)
# # print(mse)

# #    r2 
# from sklearn.metrics import r2_score
# r2=r2_score(y_test,y_predict)
# print(r2)

# #    Visualization

# plt.plot(y_test.values,label='Actual')
# plt.plot(y_predict,label='prediction')
# plt.legend()
# plt.show()
