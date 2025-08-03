import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
X = np.hstack((np.ones((X.shape[0], 1)), X))
y = df['electricity_cost'].values

X_T = X.T
weights = np.linalg.inv(X_T @ X) @ X_T @ y

predictions = X @ weights

print("\nActual vs Predicted electricity cost:")
for actual, pred in zip(y, predictions):
    print(f"Actual: {actual:.2f}, Predicted: {pred:.2f}")

mse = np.mean((y - predictions)**2)
print(f"\nMean Squared Error: {mse:.2f}")

plt.figure(figsize=(8,5))
plt.plot(range(len(y)), y, label='Actual', marker='o')
plt.plot(range(len(predictions)), predictions, label='Predicted', marker='x')
plt.xlabel('Sample index')
plt.ylabel('Electricity Cost')
plt.title('Actual vs Predicted Electricity Cost')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
