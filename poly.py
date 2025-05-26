# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error

# # Load your dataset (replace 'your_data.csv' with your actual file path)
# data = pd.read_csv('/home/sury/proj/internship/dataset/e4/MEFAR Dataset Neurophysiological and Biosignal Data/MEFAR_preprocessed/MEFAR_preprocessed/MEFAR_UP.csv')

# # Select features (HR and TEMP) and target (EDA)
# X = data[['BVP', 'TEMP']]
# y = data['EDA']

# # Split data into training and testing sets (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create polynomial features (degree 2 for this example)
# poly = PolynomialFeatures(degree=2)
# X_train_poly = poly.fit_transform(X_train)
# X_test_poly = poly.transform(X_test)

# # Fit linear regression on polynomial features
# model = LinearRegression()
# model.fit(X_train_poly, y_train)

# # Predict on test set
# y_pred = model.predict(X_test_poly)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')
# #print accuracy
# print(f'R^2 Score: {model.score(X_test_poly, y_test)}')

# # Example: Predict EDA for new data from your ring
# # new_data = [[75, 36.5]]  # Example HR=75, TEMP=36.5
# # new_data_poly = poly.transform(new_data)
# # predicted_eda = model.predict(new_data_poly)
# # print(f'Predicted EDA: {predicted_eda[0]}')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv('/home/sury/proj/internship/dataset/e4/MEFAR Dataset Neurophysiological and Biosignal Data/MEFAR_preprocessed/MEFAR_preprocessed/MEFAR_UP.csv')

# Select features and target
X = data[['BVP', 'TEMP', 'HR']]
y = data['EDA']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize tracking variables
best_degree = 1
best_score = -float('inf')
best_model = None
best_poly = None
prev_mse = float('inf')

# Loop over polynomial degrees
for degree in range(1,10):  # Try degrees 1 through 10
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Scale
    scaler = StandardScaler()
    X_train_poly_scaled = scaler.fit_transform(X_train_poly)
    X_test_poly_scaled = scaler.transform(X_test_poly)

    # Train model
    model = LinearRegression()
    model.fit(X_train_poly_scaled, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test_poly_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Degree {degree}: MSE = {mse:.4f}, R² = {r2:.4f}")

    # Sweet spot condition: Stop if MSE increases (i.e., model gets worse)
    if mse < prev_mse:
        prev_mse = mse
        best_degree = degree
        best_model = model
        best_poly = poly
        best_scaler = scaler
        best_score = r2
    else:
        print("Performance decreased. Stopping search.")
        break

# Summary
print(f"\nBest degree: {best_degree}")
print(f"Best R² Score: {best_score:.4f}")
print("Polynomial terms used:", best_poly.get_feature_names_out(['BVP', 'TEMP','HR']))
