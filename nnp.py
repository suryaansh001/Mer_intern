import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
data = pd.read_csv('/home/sury/proj/internship/dataset/e4/MEFAR Dataset Neurophysiological and Biosignal Data/MEFAR_preprocessed/MEFAR_preprocessed/MEFAR_UP.csv')

# Select features and target
X = data[['BVP', 'TEMP','HR']]
y = data['EDA']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the TensorFlow model
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),

    Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=7, batch_size=32, validation_split=0.1, verbose=1)

# Predict on test set
y_pred = model.predict(X_test_scaled).flatten()  # Flatten for comparison with y_test

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
#print F1 score 


# Predict EDA for new data from ring
# new_data = np.array([[75, 36.5]])
# new_data_scaled = scaler.transform(new_data)
# predicted_eda = model.predict(new_data_scaled)
# print(f'Predicted EDA: {predicted_eda[0][0]}')
