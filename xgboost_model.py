import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import os

# Load preprocessed data
file_path = r"C:\Users\kappala omnath\Desktop\Stock_Prediction_Project\python_file\data\FEATURE_ENGINEERED_DATA.csv"  # Ensure correct path
df = pd.read_csv(file_path)

# Debug: Print column names
print("Column Names in Dataset:", df.columns)

# Ensure column names have no extra spaces
df.columns = df.columns.str.strip()

# Define 'Target' (Predicting Next Day's Close Price)
df["Target"] = df["Close"].shift(-1)
df = df.dropna()  # Remove last row

# Define features and target
X = df.drop(columns=["Target", "Date"])  # Drop 'Date' since it's not useful for training
y = df["Target"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"XGBoost Model Performance:")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")

# Save the trained model
model_path = r"python_file\xgboost_model.pkl"
joblib.dump(xgb_model, model_path)
print(f"✅ Model saved as {model_path}")

# Load the trained XGBoost model
if os.path.exists(model_path):
    xgb_model = joblib.load(model_path)
    print("✅ Model loaded successfully!")
else:
    print("❌ Model file not found! Check the path.")

# Get feature importance
importance = xgb_model.feature_importances_
feature_names = X.columns  # Ensure X is defined

# Plot Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importance, color='skyblue')
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("XGBoost Feature Importance")
plt.show()
plt.savefig("results/xgboost_prediction_plot.png")