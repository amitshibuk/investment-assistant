import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import load_model
import os

# --- Model Loading & Prediction Logic ---
MODEL_DIR = "models"
# The Ticker map is now centralized in app.py, so it's removed from here.

# FIX: The function now accepts a ticker symbol directly.
def load_and_predict(ticker, days_ahead=7):
    """
    Loads a pre-trained model for a given ticker and predicts future stock prices.
    """
    if not ticker:
        raise ValueError("A valid ticker symbol must be provided.")

    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found for {ticker}. Please run train_models.py")

    # Load the trained model and its scaler
    model = load_model(model_path)
    scaler = MinMaxScaler(feature_range=(0, 1))

    # --- Fetch latest data for prediction input ---
    today = pd.to_datetime('today').strftime('%Y-%m-%d')
    # Use the provided ticker to download data
    data = yf.download(ticker, start="2018-01-01", end=today)
    if data.empty:
        raise ValueError(f"Could not fetch data for {ticker}")

    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    dataset = data[features]
    
    scaler.fit(dataset)
    scaled_data = scaler.transform(dataset)

    time_step = 60
    if len(scaled_data) < time_step:
        raise ValueError("Not enough historical data to make a prediction.")
        
    last_60_days = scaled_data[-time_step:]

    # --- Prediction Logic ---
    current_seq = last_60_days
    future_predictions_scaled = []
    n_features = len(features)

    for _ in range(days_ahead):
        X_future = current_seq.reshape(1, time_step, n_features)
        next_pred_scaled = model.predict(X_future, verbose=0)[0, 0]
        future_predictions_scaled.append(next_pred_scaled)

        new_row = current_seq[-1].copy()
        new_row[3] = next_pred_scaled # Update 'Close' price
        current_seq = np.vstack((current_seq[1:], new_row))

    # Inverse transform the 'Close' price predictions
    dummy_future = np.zeros((len(future_predictions_scaled), n_features))
    dummy_future[:, 3] = future_predictions_scaled
    future_prices = scaler.inverse_transform(dummy_future)[:, 3]

    return future_prices.tolist()