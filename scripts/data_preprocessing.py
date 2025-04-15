# Data cleaning and preprocessing logic
import stock_utils.logger
import pandas as pd
import os


#RAW_DATA_PATH = '\data\raw'
#PREPROCESSED_DATA_PATH = '\data\processed'

def add_technical_features(ticker):

    logger = stock_utils.logger.get_logger(__name__)

    RAW_DATA_PATH = r'C:\Users\Dell\VENVs\Stock_Advisor\data\raw'
    PREPROCESSED_DATA_PATH = r'C:\Users\Dell\VENVs\Stock_Advisor\data\processed'
    if not os.path.exists(PREPROCESSED_DATA_PATH):
        os.makedirs(PREPROCESSED_DATA_PATH)

    file_path = os.path.join(RAW_DATA_PATH, f"{ticker}.csv")

    if not os.path.exists(file_path):
        logger.warning(f"Raw data file for {ticker} not found at {file_path}")
        return
    
    df=pd.read_csv(file_path)
    #df.set_index(["date"], inplace=True)

    if df.empty:
        logger.warning(f'No data available for {ticker}. Skipping...')

    df['ticker']=ticker

    # Moving Averages (For ARIMA, LSTM, Monte Carlo)
    df["SMA_14"] = df["Close"].rolling(window=14).mean()  # 14-day Simple Moving Average
    df["EMA_14"] = df["Close"].ewm(span=14, adjust=False).mean()  # Exponential Moving Average

    # Price Returns (For Monte Carlo Simulation)
    df["Daily_Return"] = df["Close"].pct_change()  # Percentage Change

    # Bollinger Bands (For ARIMA & LSTM)
    df["Rolling_Mean"] = df["Close"].rolling(window=20).mean()
    df["Rolling_Std"] = df["Close"].rolling(window=20).std()
    df["Upper_Band"] = df["Rolling_Mean"] + (df["Rolling_Std"] * 2)
    df["Lower_Band"] = df["Rolling_Mean"] - (df["Rolling_Std"] * 2)

    # RSI (Relative Strength Index for LSTM)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # Differencing (For ARIMA - Makes Data Stationary)
    df["Price_Diff_1"] = df["Close"].diff(1)  # 1st Order Differencing
    df["Price_Diff_2"] = df["Close"].diff(2)  # 2nd Order Differencing

    # Drop NaN values created by rolling calculations
    df.dropna(inplace=True)

    # Save Preprocessed Data
    output_path = os.path.join(PREPROCESSED_DATA_PATH, f"{ticker}.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Preprocessed data saved for {ticker} at {output_path}")