# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# np.random.seed(42)
# size = 150000
# df_dropped_calls = pd.DataFrame({
#     'Technology': np.random.choice(['2G', '3G', '4G', '5G'], size),
#     'Network Congestion (%)': np.random.uniform(0, 100, size),
#     'Signal Interference (dB)': np.random.normal(0, 1, size),
#     'Signal Quality (dBm)': np.random.normal(-85, 10, size),
#     'External Interferences (units)': np.random.normal(0, 0.5, size),
#     'Dropped Calls': np.random.poisson(2, size)
# })
# df_encoded = pd.get_dummies(df_dropped_calls, columns=['Technology'])
# correlation_matrix = df_encoded.corr()
#
# plt.figure(figsize=(12, 10))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Matrix for Dropped Calls Data')
# plt.show()
#
# X = df_encoded.drop('Dropped Calls', axis=1)
# y = df_encoded['Dropped Calls']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)
#
# y_pred = rf_model.predict(X_test)
#
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
#
# print("Mean Squared Error:", mse)
# print("R^2 Score:", r2)

#
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
#
# # Simulate dataset
# np.random.seed(42)
# size = 150000
# df_dropped_calls = pd.DataFrame({
#     'Technology': np.random.choice(['2G', '3G', '4G', '5G'], size),
#     'Network Congestion (%)': np.random.uniform(0, 100, size),
#     'Signal Interference (dB)': np.random.normal(0, 1, size),
#     'Signal Quality (dBm)': np.random.normal(-85, 10, size),
#     'External Interferences (units)': np.random.normal(0, 0.5, size),
#     'Dropped Calls': np.random.poisson(2, size)
# })
#
# # One-hot encode the 'Technology' column
# df_encoded = pd.get_dummies(df_dropped_calls, columns=['Technology'])
#
# # Separate features and target variable
# X = df_encoded.drop('Dropped Calls', axis=1)
# y = df_encoded['Dropped Calls']
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# # Train a Random Forest model
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)
#
# # Predict on the testing set
# y_pred = rf_model.predict(X_test)
#
# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
#
# print("Mean Squared Error:", mse)
# print("R^2 Score:", r2)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Seed for reproducibility
np.random.seed(42)

# Simulating data
size = 150000
df = pd.DataFrame({
    'Network Congestion (%)': np.random.uniform(0, 100, size),
    'Signal Interference (dB)': np.random.normal(0, 1, size),
    'Signal Quality (dBm)': np.random.normal(-85, 10, size),
    'External Interferences (units)': np.random.normal(0, 0.5, size),
    'Dropped Calls': np.random.poisson(2, size)
})

# One-hot encoding if there were categorical variables
# df = pd.get_dummies(df, columns=['CategoricalColumn'])

# Split data into features and target
X = df.drop('Dropped Calls', axis=1)
y = df['Dropped Calls']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Plot the results
plt.figure(figsize=(10, 6))
# Plotting actual values in green
actual_scatter = plt.scatter(X_test['Signal Interference (dB)'], X_test['Signal Quality (dBm)'],
                             c=y_test, cmap='Greens', alpha=0.6, edgecolor='k', label='Actual Dropped Calls')
# Plotting predicted values in red
pred_scatter = plt.scatter(X_test['Signal Interference (dB)'], X_test['Signal Quality (dBm)'],
                           c=y_pred, cmap='Reds', alpha=0.6, edgecolor='k', label='Predicted Dropped Calls')

# Adding a colorbar
plt.colorbar(pred_scatter, label='Number of Dropped Calls')
plt.xlabel('Signal Interference (dB)')
plt.ylabel('Signal Quality (dBm)')
plt.title('Comparison of Actual and Predicted Dropped Calls')
plt.legend()
plt.show()