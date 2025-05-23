from statsmodels.tsa.arima.model import ARIMA
import warnings

def train_and_forecast(prices, horizon):
    """
    Train an ARIMA model on the given price series and forecast next 'horizon' points.
    
    Parameters:
    prices (array-like): Historical price data (1D array or list)
    horizon (int): Number of future points to forecast
    
    Returns:
    forecast (numpy.ndarray): Forecasted price values
    """
    warnings.filterwarnings("ignore")  # Suppress convergence warnings
    
    # Fit ARIMA model - order (5,1,0) is a simple choice; can be tuned
    model = ARIMA(prices, order=(5,1,0))
    model_fit = model.fit()
    
    # Forecast next 'horizon' points
    forecast = model_fit.forecast(steps=horizon)
    
    return forecast
