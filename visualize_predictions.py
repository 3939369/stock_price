import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf

# Load test data
file_path = "python_file/FEATURE_ENGINEERED_DATA.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()  # Ensure no extra spaces in column names

# Shift Close price to predict next day's price
df["Target"] = df["Close"].shift(-1)
df = df.dropna()

# Separate features and target
X_test = df.drop(columns=["Target", "Date"])
y_test = df["Target"]

# Load trained XGBoost model
xgb_model = joblib.load("python_file/xgboost_model.pkl")
y_pred_xgb = xgb_model.predict(X_test)

# Load scalers
scaler_x = joblib.load("python_file/Scaler_x.pkl")  # Ensure the same scaler used for training
scaler_y = joblib.load("python_file/Scaler_y.pkl")

# Scale test data
X_test_scaled = scaler_x.transform(X_test)

# Debug: Print shape to verify correct dimensions
print(f"X_test_scaled shape: {X_test_scaled.shape}")  # Should be (samples, features)

# Correct reshaping for LSTM input (batch_size, time_steps=1, features)
X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# Load trained LSTM model
lstm_model = tf.keras.models.load_model("python_file/lstm_model.keras")

# Predict using LSTM model
y_pred_lstm_scaled = lstm_model.predict(X_test_lstm).flatten()

# Inverse transform LSTM predictions to get actual prices
y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm_scaled.reshape(-1, 1)).flatten()

# Create a figure and axis
plt.figure(figsize=(12, 6))

# Plot actual prices
plt.plot(y_test.values, label="Actual Prices", color="black", linestyle="dashed", alpha=0.7)

# Plot XGBoost predictions
plt.plot(y_pred_xgb, label="XGBoost Predictions", color="blue", alpha=0.7)

# Plot LSTM predictions
plt.plot(y_pred_lstm, label="LSTM Predictions", color="red", alpha=0.7)

# Labels and title
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.title("Actual vs. Predicted Stock Prices")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()


plt.savefig("python_file/predictions_plot.png")
