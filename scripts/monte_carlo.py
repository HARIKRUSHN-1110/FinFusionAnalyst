# Monte Carlo simulation logic
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
#from matplotlib.dates import DateFormatter
import plotly.graph_objects as go

# Define paths
PREPROCESSED_DATA_PATH = r'C:\Users\Dell\VENVs\Stock_Advisor\data\processed'
MODEL_SAVE_PATH = r'C:\Users\Dell\VENVs\Stock_Advisor\models\monte_carlo'

# Ensure the model save directory exists
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def calculate_log_returns(prices: np.ndarray):
    """Calculate log returns from price data."""
    return np.log(prices[1:] / prices[:-1])

def monte_carlo_simulation(prices: np.ndarray, num_simulations=1000, num_days=5):
    """Run Monte Carlo simulation using Geometric Brownian Motion (GBM)."""
    log_returns = calculate_log_returns(prices)
    mu = np.mean(log_returns)  # Drift (average log return)
    sigma = np.std(log_returns)  # Volatility (std deviation of log returns)

    last_price = prices[-1]  # Start simulation from the latest price

    simulations = np.zeros((num_simulations, num_days))
    
    for i in range(num_simulations):
        daily_returns = np.random.normal(mu, sigma, num_days)
        price_path = last_price * np.exp(np.cumsum(daily_returns))  # GBM formula
        simulations[i] = price_path

    return simulations

def predict_stock_price_monte_carlo(ticker, num_simulations=1000, num_days=5):
    """Predict future stock prices using Monte Carlo simulation."""
    file_path = os.path.join(PREPROCESSED_DATA_PATH, f"{ticker}.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file for {ticker} not found at {file_path}")

    df = pd.read_csv(file_path, index_col=1)
    df.set_index("date", inplace=True)
    df.index = pd.to_datetime(df.index) # Ensure index is in datetime format
    df['ticker'] = ticker
    
    if "Close" not in df.columns:
        raise KeyError("Missing 'Close' column in dataset.")

    prices = df["Close"].values

    # Run Monte Carlo simulation
    simulations = monte_carlo_simulation(prices, num_simulations, num_days)

    # Calculate mean of the simulated paths as the final predicted prices
    predicted_prices = np.mean(simulations, axis=0)

    # Generate future dates
    last_date = pd.to_datetime(df.index[-1])  # Ensure index is datetime
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_days, freq='B')  # Business days
    future_dates = pd.to_datetime(future_dates)
    
    last_10_days = df.index[-10:]
    last_10_prices = prices[-10:]
    # Plot results
    return plot_monte_carlo(last_10_days, last_10_prices, future_dates, predicted_prices, simulations)

    #return future_dates, predicted_prices

def plot_monte_carlo(last_10_days, last_10_prices, future_dates, predicted_prices, simulations):
    fig = go.Figure()

    # Historical prices (last 10 days)
    fig.add_trace(go.Scatter(
        x=last_10_days,
        y=last_10_prices,
        mode='lines+markers',
        name='Historical Prices',
        line=dict(color='blue'),
        hovertemplate='Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
    ))

    # Monte Carlo simulations (hidden hover labels)
    for sim in simulations:
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=sim,
            mode='lines',
            line=dict(color='gray', width=1),
            opacity=0.1,
            showlegend=False,
            hoverinfo='skip'
        ))

    # Predicted prices (future)
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predicted_prices,
        mode='lines+markers',
        name='Predicted Prices',
        line=dict(color='red'),
        hovertemplate='Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
    ))

    # Continuous line for historical + future prices
    combined_dates = pd.concat([pd.Series(last_10_days), pd.Series(future_dates)])
    combined_prices = pd.concat([pd.Series(last_10_prices), pd.Series(predicted_prices)])

    fig.add_trace(go.Scatter(
        x=combined_dates,
        y=combined_prices,
        mode='lines',
        name='Continuous Prediction',
        line=dict(color='green', dash='dash'),
        hoverinfo='skip'
    ))

    # X-axis with all dates explicitly shown
    fig.update_layout(
        title='Monte Carlo Stock Price Prediction',
        xaxis=dict(
            title='Date (YYYY-MM-DD)',
            type='category',  # Ensure dates show as categories, not continuous values
            tickvals=combined_dates,
            ticktext=[date.strftime('%Y-%m-%d') for date in combined_dates],
            showgrid=True
        ),
        yaxis=dict(
            title='Stock Price'
        ),
        legend=dict(x=0, y=1),
        hovermode='x unified',
        template='plotly_white'
    )

    return fig