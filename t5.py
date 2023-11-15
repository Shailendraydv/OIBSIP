import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import os

path=os.path.dirname(os.path.abspath(__file__))

data = pd.read_csv(path+'/Advertising.csv')


X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

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
df=pd.DataFrame([[17.2,45.9,69.3]],columns=['TV', 'Radio', 'Newspaper']) # Give value here
df = poly_features.transform(df)
y_pred = model.predict(df)
print(y_pred)

