import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def train_and_forecast(prices, horizon):
    prices = np.array(prices).reshape(-1, 1)

    # Remove NaNs from prices
    prices = prices[~np.isnan(prices).flatten()].reshape(-1,1)

    # Scale prices between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(prices)

    n_input = 10  # timesteps to look back
    n_features = 1

    # Prepare training data
    X, y = [], []
    for i in range(n_input, len(prices_scaled)):
        X.append(prices_scaled[i - n_input:i, 0])
        y.append(prices_scaled[i, 0])
    X, y = np.array(X), np.array(y)

    X = X.reshape((X.shape[0], X.shape[1], n_features))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_input, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X, y, epochs=50, batch_size=16, verbose=0)

    # Forecast future values
    forecast = []
    input_seq = prices_scaled[-n_input:].reshape(1, n_input, n_features)

    for _ in range(horizon):
        pred_scaled = model.predict(input_seq, verbose=0)[0][0]

        # Handle potential NaN from prediction
        if np.isnan(pred_scaled):
            pred_scaled = 0.0

        forecast.append(pred_scaled)

        # Slide input window forward
        input_seq = np.append(input_seq[:, 1:, :], [[[pred_scaled]]], axis=1)

    # Inverse transform predictions back to original scale
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()

    return forecast
