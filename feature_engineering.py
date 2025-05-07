import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def calculate_technical_indicators(df):
    # Simple Moving Average (SMA)
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Exponential Moving Average (EMA)
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Mid'] + (df['Close'].rolling(window=20).std() * 2)
    df['BB_Lower'] = df['BB_Mid'] - (df['Close'].rolling(window=20).std() * 2)
    
    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    return df

def normalize_features(df):
    features = ['SMA_10', 'SMA_50', 'EMA_10', 'RSI', 'MACD', 'Signal_Line', 'BB_Mid', 'BB_Upper', 'BB_Lower', 'OBV']
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    return df

if __name__ == "__main__":
    # Load the preprocessed stock data
    df = pd.read_csv(r"C:\Users\kappala omnath\OneDrive\capston\OneDrive\文档\Stock_Prediction_Project\python_file\FEATURE_ENGINEERED_DATA.csv")
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Normalize feature values
    df = normalize_features(df)
    
    # Save the feature-engineered dataset
    df.to_csv("FEATURE_ENGINEERED_DATA.csv", index=False)
    print("Feature engineering complete. Saved as FEATURE_ENGINEERED_DATA.csv")
