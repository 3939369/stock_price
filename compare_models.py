import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load test data
file_path = "python_file/FEATURE_ENGINEERED_DATA.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# Define Target (Predicting Next Day's Close Price)
df["Target"] = df["Close"].shift(-1)
df = df.dropna()

# Select features
features = ["Adj Close", "Close", "High", "Low", "Open", "Volume", 
            "SMA_10", "SMA_50", "EMA_10", "RSI", "EMA_12", "EMA_26", 
            "MACD", "Signal_Line", "BB_Mid", "BB_Upper", "BB_Lower", "OBV"]

X_test = df[features]
y_test = df["Target"].values.reshape(-1, 1)

# Load trained scalers
scaler_X = joblib.load("python_file/scaler_X.pkl")
scaler_y = joblib.load("python_file/scaler_y.pkl")

# Transform test data
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test)

# Load trained XGBoost model
xgb_model = joblib.load("python_file/xgboost_model.pkl")
y_pred_xgb_scaled = xgb_model.predict(X_test_scaled)

# Inverse transform XGBoost predictions
y_pred_xgb = scaler_y.inverse_transform(y_pred_xgb_scaled.reshape(-1, 1)).flatten()

# Load trained LSTM model
lstm_model = tf.keras.models.load_model("python_file/lstm_model.keras")

# Reshape for LSTM model
X_test_lstm = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Predict using LSTM
y_pred_lstm_scaled = lstm_model.predict(X_test_lstm).flatten()

# Inverse transform LSTM predictions
y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm_scaled.reshape(-1, 1)).flatten()

# Evaluate XGBoost
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

# Evaluate LSTM
rmse_lstm = np.sqrt(mean_squared_error(y_test, y_pred_lstm))
r2_lstm = r2_score(y_test, y_pred_lstm)
mae_lstm = mean_absolute_error(y_test, y_pred_lstm)

# Print results
print("XGBoost Model Performance:")
print(f"RMSE: {rmse_xgb:.4f}")
print(f"R² Score: {r2_xgb:.4f}")
print(f"MAE: {mae_xgb:.4f}\n")

print("LSTM Model Performance:")
print(f"RMSE: {rmse_lstm:.4f}")
print(f"R² Score: {r2_lstm:.4f}")
print(f"MAE: {mae_lstm:.4f}")

# Determine the best model
if rmse_xgb < rmse_lstm:
    print("\n XGBoost performs better based on RMSE!")
else:
    print("\n LSTM performs better based on RMSE!")
