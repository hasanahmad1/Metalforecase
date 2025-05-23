import numpy as np
from sklearn.ensemble import RandomForestRegressor

def train_and_forecast(prices, horizon):
    """
    Train a Random Forest Regressor using lagged features on price data and forecast future values.

    Parameters:
    prices (array-like): Historical price data (1D array or list)
    horizon (int): Number of future points to forecast

    Returns:
    forecast (list): Forecasted price values
    """

    # Convert to numpy array
    prices = np.array(prices)

    # Remove NaN values to avoid training errors
    prices = prices[~np.isnan(prices)]

    # If after cleaning, not enough data, raise an error
    if len(prices) < 6:
        raise ValueError("Not enough data points after removing NaNs to train the model.")

    # Prepare lagged features: use past 5 days to predict next day
    X, y = [], []
    for i in range(5, len(prices)):
        X.append(prices[i-5:i])
        y.append(prices[i])
    X, y = np.array(X), np.array(y)

    # Initialize and train Random Forest model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Use last 5 days as seed input for forecasting
    last_seq = list(prices[-5:])
    forecast = []

    # Rolling forecast for 'horizon' days
    for _ in range(horizon):
        pred = rf.predict([last_seq])[0]
        forecast.append(pred)
        last_seq.pop(0)
        last_seq.append(pred)

    return forecast
