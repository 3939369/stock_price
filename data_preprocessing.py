import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load Dataset (Choose CSV or yfinance)
use_csv = True  # Set to False if using yfinance

if use_csv:
    # Load stock data from CSVdf
     df = pd.read_csv(r"C:\Users\kappala omnath\OneDrive\capston\OneDrive\文档\Stock_Prediction_Project\python_file\NVIDIA_STOCK_CLEANED.csv")


else:
    # Fetch stock data using yfinance
    df = yf.download("NVDA", start="2010-01-01", end="2025-01-01")

# Convert Date column to datetime format
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

# Handle missing values
df.fillna(method='ffill', inplace=True)  # Forward fill missing values

# Select features and target variable
features = ["Open", "High", "Low", "Close", "Volume"]
target = "Close"

# Normalize features using MinMaxScaler
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Split dataset into Train & Test (80% Train, 20% Test)
train_size = int(0.8 * len(df))
train, test = df[:train_size], df[train_size:]

X_train, X_test = train[features], test[features]
y_train, y_test = train[target], test[target]

# Save Preprocessed Data
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print(f"Data Preprocessing Complete ")
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
