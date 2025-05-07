import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load dataset
file_path = "python_file/FEATURE_ENGINEERED_DATA.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# Define Target (Predicting Next Day's Close Price)
df["Target"] = df["Close"].shift(-1)
df = df.dropna()  # Drop last row (since it has no target)

# Select features (excluding Date and Target)
features = ["Adj Close", "Close", "High", "Low", "Open", "Volume", 
            "SMA_10", "SMA_50", "EMA_10", "RSI", "EMA_12", "EMA_26", 
            "MACD", "Signal_Line", "BB_Mid", "BB_Upper", "BB_Lower", "OBV"]

X = df[features]
y = df["Target"].values.reshape(-1, 1)

# Normalize features
scaler_X = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler(feature_range=(0, 1))
y_scaled = scaler_y.fit_transform(y)

# Save scalers
joblib.dump(scaler_X, "python_file/scaler_X.pkl")
joblib.dump(scaler_y, "python_file/scaler_y.pkl")
print(" Scalers saved successfully!")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Reshape input for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Continue with model training...
