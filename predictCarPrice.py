a = '''import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib.inline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics
from sklearn.model_selection import train_test_split
'''


def predict():
    print("stupiiiiid")

x = '''
cars_data = pd.read_csv('/kaggle/input/vehicle-dataset-from-cardekho/car data.csv')

cars_data.head()
cars_data.shape
cars_data.info()
cars_data.describe()
cars_data.isna().sum()
cars_data.duplicated().sum()
cars_data.columns

#preprocessing
cars_data['Fuel_Type'].value_counts()
cars_data['Seller_Type'].value_counts()
cars_data['Transmission'].value_counts()

from sklearn.preprocessing import LabelEncoder

# Create a sample dataframe with categorical data
Fuel = pd.DataFrame({'Fuel_Type': ['Petrol', 'Diesel', 'CNG']})
Seller = pd.DataFrame({'Seller_Type': ['Dealer', 'Individual']})
Transmiss = pd.DataFrame({'Transmission': ['Manual', 'Automatic']})

print(f"Before Encoding the Data:\n\n{Fuel}\n")
print(f"Before Encoding the Data:\n\n{Seller}\n")
print(f"Before Encoding the Data:\n\n{Transmiss}\n")

# Create a LabelEncoder object
le = LabelEncoder()

# Fit and transform the categorical data
cars_data['Fuel_Type'] = le.fit_transform(cars_data['Fuel_Type'])
cars_data['Seller_Type'] = le.fit_transform(cars_data['Seller_Type'])
cars_data['Transmission'] = le.fit_transform(cars_data['Transmission'])

cars_data.head()

#build model
x=cars_data.drop(['Car_Name','Selling_Price'],axis=1)
y=cars_data['Selling_Price']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=11)

reg=LinearRegression()
reg.fit(x_train,y_train)
training_prediction= reg.predict(x_train)
error_score=metrics.r2_score(y_train,training_prediction)

plt.scatter(y_train, training_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()

test_prediction= reg.predict(x_test)

error_score=metrics.r2_score(y_test,test_prediction)
'''