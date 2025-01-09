import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Load the dataset
# Replace 'usd_inr_rates.csv' with your CSV file
data = pd.read_csv("usd_inr_rate.csv")

# Step 2: Inspect the dataset
print("Dataset Overview:")
print(data.head())
print("\nDataset Info:")
print(data.info())

# Ensure the dataset has 'Year' and 'ExchangeRate' columns
if 'year' not in data.columns or 'rate' not in data.columns:
    raise ValueError("The dataset must contain 'Year' and 'ExchangeRate' columns.")

# Step 3: Prepare the data
X = data[['year']]  # Independent variable
y = data['rate']  # Dependent variable

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

# Step 5: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nModel Evaluation:\nMean Squared Error: {mse:.2f}\nR2 Score: {r2:.2f}")

# Step 7: Predict future values (next 5 years)
current_year = data['year'].max()
future_years = np.array(range(current_year + 1, current_year + 6)).reshape(-1, 1)
future_predictions = model.predict(future_years)

# Step 8: Display predictions
predictions_df = pd.DataFrame({'year': future_years.flatten(), 'PredictedExchangeRate': future_predictions})
print("\nPredicted Exchange Rates for Next 5 Years:")
print(predictions_df)

# Step 9: Plot results
plt.figure(figsize=(10, 6))
plt.scatter(data['year'], data['rate'], color='blue', label='Actual Data')
plt.plot(data['year'], model.predict(data[['year']]), color='green', label='Model Prediction')
plt.scatter(future_years, future_predictions, color='red', label='Future Predictions')
plt.title('USD to INR Exchange Rate Prediction')
plt.xlabel('Year')
plt.ylabel('Rate')
plt.legend()
plt.grid()
plt.show()
