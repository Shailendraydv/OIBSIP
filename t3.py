# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

def prep(df):
    df.replace('Petrol',1,inplace=True)
    df.replace('Diesel',0,inplace=True)
    df.replace('CNG',2,inplace=True)
    df.replace('Manual',1,inplace=True)
    df.replace('Automatic',0,inplace=True)

path=os.path.dirname(os.path.abspath(__file__))
car_data = pd.read_csv(path+'\car data.csv')
prep(car_data)

# Assume your dataset has relevant columns for features and the target variable
# Adjust these column names based on your actual dataset
X = car_data[['Year', 'Selling_Price', 'Driven_kms','Fuel_Type','Transmission']]
y = car_data['Present_Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Polynomial Regression
poly_degree = 2  # You can adjust this degree as needed
poly_features = PolynomialFeatures(degree=poly_degree)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# Create a linear regression model
model = LinearRegression()

# Train the model on the polynomial features
model.fit(X_train_poly, y_train)

# Make predictions on the testing data
df=pd.DataFrame([[2011,2.85,5200,'Petrol','Manual']],columns=['Year', 'Selling_Price', 'Driven_kms','Fuel_Type','Transmission']) # Give value here
prep(df)
df = poly_features.transform(df)
y_pred = model.predict(df)
print(y_pred)