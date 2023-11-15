import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os

path=os.path.dirname(os.path.abspath(__file__))
# Load datasets from a CSV file
data = pd.read_csv(path+'/iris.csv')

# Separate features (X) and target (y)
X = data.drop(['Species','Id'], axis=1)  # target_column_name
y = data['Species']
y1=set(y)
d1,d2={},{}
j=0
for i in y1:
    d1[i]=j
    d2[j]=i
    j+=1
for i in range(0,j):
    y.replace(to_replace=d2[i],value=i,inplace=True)

# training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build an ANN model with TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax') 
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Accuracy is: {accuracy:.2f}")
# Replace value to predict here
ans=model.predict([[7,3.2,4.7,1.4]])
print(d2[np.argmax(ans)])