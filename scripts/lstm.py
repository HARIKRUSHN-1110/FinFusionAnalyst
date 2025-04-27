import os
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#from tensorflow.keras.models import load_model
from models.lstm_model import build_lstm_model
from scripts.fetch_data import get_nifty_50_data
from scripts.data_preprocessing import add_technical_features
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from datetime import timedelta
import seaborn as sns
import stock_utils.logger
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Paths
PREPROCESSED_DATA_PATH = r'C:\Users\Dell\VENVs\Stock_Advisor\data\processed'
SCALED_DATA_PATH = r'C:\Users\Dell\VENVs\Stock_Advisor\data\scaled_data\lstm'
# Ensure the directory exist
stock_utils.logger.setup_logging_config()
logger = stock_utils.logger.get_logger(__name__)
os.makedirs(SCALED_DATA_PATH, exist_ok=True)

def scale_and_save_data(ticker):
    
    file_path = os.path.join(PREPROCESSED_DATA_PATH, f"{ticker}.csv")

    if not os.path.exists(file_path):
        logger.warning(f"Preprocessed data file for {ticker} not found at {file_path}")
        return
    df = pd.read_csv(file_path)

    df['ticker'] = ticker
    selected_features = ['Low', 'High', 'Daily_Return', 'Price_Diff_1', 'Price_Diff_2', 'EMA_14']
    df = df[selected_features] 

    # Initialize and fit MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    # Save the Scaler (Fixing Your Error)
    scaler_path = os.path.join(SCALED_DATA_PATH, f'{ticker}.pkl')
    joblib.dump(scaler, scaler_path)  

    # Save the Scaled Data
    scaled_df = pd.DataFrame(scaled_data, columns=selected_features)
    scaled_csv_path = os.path.join(SCALED_DATA_PATH, f'{ticker}_scaled.csv')
    scaled_df.to_csv(scaled_csv_path, index=False)

    print(f"Scaler and scaled data saved for {ticker}")
MODEL_SAVE_PATH = r'C:\Users\Dell\VENVs\Stock_Advisor\models\lstm_models'

LOOKBACK=30
def train_lstm_model(ticker):
    
    scaler_path = os.path.join(SCALED_DATA_PATH, f'{ticker}.pkl')
    scaled_csv_path = os.path.join(SCALED_DATA_PATH, f'{ticker}_scaled.csv')

    # Load Scaler
    if not os.path.exists(scaler_path):
        print(f"Scaler file {ticker}.pkl not found! Run scale_and_save_data() first.")
        return

    scaler = joblib.load(scaler_path)

    # Load Scaled Data
    if not os.path.exists(scaled_csv_path):
        print(f"Scaled CSV {ticker}_scaled.csv not found! Skipping {ticker}.")
        return

    df = pd.read_csv(scaled_csv_path)
    scaled_data = df.values

    # Prepare data for LSTM
    #LOOKBACK=15
    X, y = prepare_lstm_data(scaled_data, LOOKBACK=LOOKBACK)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)  # Assume function exists

        # Build and train the model
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)
    history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, verbose=1, callbacks=[early_stopping, reduce_lr])

    # Save model
    model_path = os.path.join(MODEL_SAVE_PATH, f"{ticker}.h5")
    model.save(model_path)
    print(f"Model for {ticker} saved at {model_path}")

        
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

#def plot_training_history(history):

    #plt.figure(figsize=(10, 6))
    #plt.plot(history.history['loss'], label='Training Loss')
    #plt.plot(history.history['val_loss'], label='Validation Loss')
    #plt.title('Model Training History')
    #plt.xlabel('Epochs')
    #plt.ylabel('Loss')
    #plt.legend()
    #plt.show()
def prepare_lstm_data(scaled_data, LOOKBACK):
    """
    Converts scaled data into LSTM-compatible sequences.
    """
    X, y = [], []
    for i in range(len(scaled_data) - LOOKBACK):
        X.append(scaled_data[i : i + LOOKBACK])
        y.append(scaled_data[i + LOOKBACK, 0])  # Predicting the first feature (Low)
    
    return np.array(X), np.array(y)

def predict_stock_price_lstm(ticker):
    # Load preprocessed data
    get_nifty_50_data(ticker)
    add_technical_features(ticker)
    scale_and_save_data(ticker)
    train_lstm_model(ticker)
    file_path = os.path.join(PREPROCESSED_DATA_PATH, f"{ticker}.csv")
    df = pd.read_csv(file_path)
    df.set_index("date", inplace=True)
    df.index = pd.to_datetime(df.index)
    
    # Load scaler and model
    scaler_path = os.path.join(SCALED_DATA_PATH, f"{ticker}.pkl")
    model_path = os.path.join(MODEL_SAVE_PATH, f"{ticker}.h5")
    scaler = joblib.load(scaler_path)
    model = load_model(model_path)
    
    #LOOKBACK = 15  # Number of previous days to use for predicting the next day

    # Prepare last window of data for prediction
    selected_features = ['Low', 'High', 'Daily_Return', 'Price_Diff_1', 'Price_Diff_2', 'EMA_14']
    last_data = df[selected_features].values[-LOOKBACK:]
    scaled_data = scaler.transform(last_data)
    last_data = np.expand_dims(scaled_data, axis=0)  # Shape: (1, LOOKBACK, num_features)
    
    # Predict the next 5 days
    predictions = []
    for _ in range(5):
        next_pred = model.predict(last_data)
        predictions.append(next_pred[0][0])
        
        # Update the sliding window
        next_pred_reshaped = np.tile(next_pred, (1, last_data.shape[2]))  # Repeat the value across all features
        last_data = np.append(last_data[:, 1:, :], [next_pred_reshaped], axis=1)


    # Inverse transform predictions
    extended_pred = np.concatenate((np.array(predictions).reshape(-1, 1), np.zeros((5, scaled_data.shape[1] - 1))), axis=1)
    predicted_prices = scaler.inverse_transform(extended_pred)[:, 0]
    
    #Adjust dates for input data after LOOKBACK window
    #input_dates = pd.to_datetime(df.index[LOOKBACK:])
    #input_prices = df['Low'].values[LOOKBACK:]
    #print('lastdate: ',last_date)
    #print(df.index)
    last_date = pd.to_datetime(df.index[-1])
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=5, freq='B')
    future_dates = pd.to_datetime(future_dates)
    
    prices = df["Close"].values
    last_10_days = pd.to_datetime(df.index[-10:])
    last_10_prices = prices[-10:]
    #print("last date: ", last_date)
    #print("last 10 days:", last_10_days)
    #print("future dates:", future_dates)
    #plot_predictions(last_10_days, last_10_prices, predicted_prices=predicted_prices, last_date=last_date)
    return plot_lstm(last_10_days, last_10_prices, predicted_prices=predicted_prices,future_dates=future_dates, ticker=ticker)

def plot_predictions(dates, original_prices, predicted_prices, last_date):
    # Ensure dates are in datetime format
    dates = pd.to_datetime(dates)

    # Generate the next 5 dates starting from the last date
    future_dates = pd.date_range(start=last_date, periods=6, freq='B')[1:]  # 1: to skip the last known date
    #print("future dates:", future_dates)
    #print("last date:", last_date)

    # Plot historical prices
    plt.plot(dates, original_prices, label='Historical Prices', color='blue')

    # Plot predicted prices
    plt.plot(future_dates, predicted_prices, label='Predicted Prices', color='orange', linestyle='--', marker='o')

    # Highlight predicted prices
    plt.fill_between(future_dates, predicted_prices, color='orange', alpha=0.3)

    # Labels and title
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Price Prediction')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_lstm(last_10_days, last_10_prices, predicted_prices, future_dates, ticker):
    fig= go.Figure()

    # Historical prices (last 10 days)
    fig.add_trace(go.Scatter(
        x=last_10_days,
        y=last_10_prices,
        mode='lines+markers',
        name='Historical Prices',
        line=dict(color='royalblue', width=2),
        hovertemplate='Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predicted_prices,
        mode='lines+markers',
        name='Predicted Prices',
        line=dict(color='red'),
        hovertemplate='Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
    ))

    # Continuous line for historical + future prices
    combined_dates = pd.concat([pd.Series(last_10_days), pd.Series(future_dates)], ignore_index=True)
    combined_prices = pd.concat([pd.Series(last_10_prices), pd.Series(predicted_prices)], ignore_index=True)

    fig.add_trace(go.Scatter(
        x=combined_dates,
        y=combined_prices,
        mode='lines',
        name='Continuous Prediction',
        line=dict(color='#bbbbbb', dash='dash'),
        hoverinfo='skip'
    ))

    # X-axis with all dates explicitly shown
    fig.update_layout(
        title=f'LSTM Stock Price Prediction of {ticker}',
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
        template='plotly_dark',
        plot_bgcolor="#1e1e2f",
        paper_bgcolor="#1e1e2f",
        font=dict(color="#bbbbbb"),
        margin=dict(l=40, r=40, t=60, b=40),
        height=500
    )

    return fig