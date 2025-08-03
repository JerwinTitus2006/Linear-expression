import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('electricity_cost_dataset.csv')

df.rename(columns={
    'site area': 'site_area',
    'structure type': 'structure_type',
    'water consumption': 'water_consumption',
    'recycling rate': 'recycling_rate',
    'utilisation rate': 'utilisation_rate',
    'air qality index': 'air_quality_index',
    'issue reolution time': 'issue_resolution_time',
    'resident count': 'resident_count',
    'electricity cost': 'electricity_cost'
}, inplace=True)

df['structure_type_encoded'] = df['structure_type'].apply(lambda x: 1 if x == 'Mixed-use' else 0)

feature_columns = [
    'site_area',
    'structure_type_encoded',
    'water_consumption',
    'recycling_rate',
    'utilisation_rate',
    'air_quality_index',
    'issue_resolution_time',
    'resident_count'
]

X = df[feature_columns].values
X_with_intercept = np.hstack((np.ones((X.shape[0], 1)), X))
y = df['electricity_cost'].values

# Own Linear Regression (normal equation)
X_T = X_with_intercept.T
weights = np.linalg.inv(X_T @ X_with_intercept) @ X_T @ y
predictions_own = X_with_intercept @ weights

# Sklearn Linear Regression
sk_model = LinearRegression()
sk_model.fit(X, y)
predictions_sklearn = sk_model.predict(X)

# Mean Squared Error
mse_own = np.mean((y - predictions_own)**2)
mse_sklearn = mean_squared_error(y, predictions_sklearn)

print("\nMean Squared Error:")
print(f"Your model: {mse_own:.2f}")
print(f"Sklearn: {mse_sklearn:.2f}")

print("\nYour model weights (first=intercept):")
print(weights)

print("\nSklearn model:")
print(f"Intercept: {sk_model.intercept_}")
print(f"Coefficients: {sk_model.coef_}")

# Plot
plt.figure(figsize=(8,5))
plt.plot(range(len(y)), y, label='Actual', marker='o')
plt.plot(range(len(predictions_own)), predictions_own, label='Your model', marker='x')
plt.plot(range(len(predictions_sklearn)), predictions_sklearn, label='Sklearn', marker='d')
plt.xlabel('Sample index')
plt.ylabel('Electricity Cost')
plt.title('Actual vs Predicted Electricity Cost')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
