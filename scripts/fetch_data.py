import yfinance as yf
import pandas as pd
import stock_utils.logger
#import pandas_market_calendars as mcal
import sys
import os
import time
from alpha_vantage.timeseries import TimeSeries

stock_utils.logger.setup_logging_config()
logger = stock_utils.logger.get_logger(__name__)
# Nifty 50 tickers (add all 50 or use your subset for now)
#Nifty_50_tickers = ['RELIANCE.BO']
Nifty_50_tickers=['TATAMOTORS.BO','RELIANCE.BO','TCS.BO','BAJFINANCE.BO','INFY.BO','HDFCBANK.BO','TATASTEEL.BO','LT.BO','ICICIBANK.BO','INDUSINDBK.BO','SBIN.BO','M&M.BO','ZOMATO.BO','SUNPHARMA.BO','ADANIPORTS.BO','HINDUNILVR.BO','HCLTECH.BO','AXISBANK.BO','BHARTIARTL.BO','POWERGRID.BO','NTPC.BO','TECHM.BO','ITC.BO','NESTLEIND.BO','ULTRACEMCO.BO','BAJAJFINSV.BO','KOTAKBANK.BO','TITAN.BO','MARUTI.BO']
#def restructure_data(file_path):
#    data1=pd.read_csv(file_path)
#    data1.drop([data1.index[0], data1.index[1]], inplace=True)
#    data1.rename({data1.columns[0]: "Date"}, axis=1,inplace=True)
#    data1.set_index(["Date"], inplace=True)
#    for ticker in Nifty_50_tickers:
#         data1['ticker']=ticker
#    data1["Date"] = pd.to_datetime(data1["Date"])
#    return data1.to_csv(file_path)

#def get_nifty_50_data():
    #end_date = datetime.strptime(end_date, '%Y-%m-%d')
    #start_date = datetime.strptime(start_date, '%Y-%m-%d')
    #end_date = datetime.today().strftime('%Y-%m-%d')
    #start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    #Nifty_50_tickers = ['SBIN.NS', 'RELIANCE.NS']
    #for ticker in Nifty_50_tickers:
            #logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}...")
            
            #stock_data = yf.download(ticker, start=start_date, end=end_date)
            #stock_data = yf.download(ticker, period='5y', interval='1d')
            #if stock_data.empty:
                #logger.warning(f"No data available for {ticker}. Skipping...")
                #continue

            #file_path = f'./data/raw/{ticker}.csv'
            #stock_data.loc[:, ~stock_data.columns.str.contains('^Unnamed', regex=True)]
            #stock_data.to_csv(file_path)
            #time.sleep(2)
            #restructure_data(file_path)
            
            #logger.info(f"Data for {ticker} downloaded and saved to {file_path}")
            #logger.info(f"Data downloded and restructured for {ticker} and saved to {file_path}")

    #return "Data fetching complete."

ALPHA_VANTAGE_API_KEY = os.getenv('JO8G6VONBJZIPX6E')

# Paths
RAW_DATA_PATH = os.path.join('data', 'raw')
os.makedirs(RAW_DATA_PATH, exist_ok=True)

def get_nifty_50_data(ticker):
    """
    Fetches 5 years of daily stock data for Nifty 50 companies using Alpha Vantage.
    Saves each stock's data to data/raw/{ticker}.csv.
    """
    ts = TimeSeries(key='JO8G6VONBJZIPX6E', output_format='pandas')

    logger.info(f"Fetching data for {ticker}...")
        
    try:
         # Fetch daily stock data (full = all available data)
        data, _ = ts.get_daily(symbol=ticker, outputsize='full')

        # Rename columns for consistency
        data.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        }, inplace=True)

        # Convert index to datetime and filter last 5 years
        #data.index = pd.to_datetime(data.index)
        #data.index.name = 'date'
        five_years_ago = pd.Timestamp.today() - pd.DateOffset(years=5)
        data = data[data.index >= five_years_ago]

        # Skip if data is empty
        if data.empty:
            logger.warning(f"No data available for {ticker}. Skipping...")
        # Save to CSV
        #data.set_index(["date"], inplace=True)
        data.index = pd.to_datetime(data.index)
        #data.set_index("date")
        data=data.sort_index(ascending=True)
        file_path = os.path.join(RAW_DATA_PATH, f"{ticker}.csv")
        data.to_csv(file_path)
        #restructure_data(file_path)
        logger.info(f"Data for {ticker} saved to {file_path}")

        # Pause to avoid hitting rate limits (5 requests per minute for free tier)
        time.sleep(2)

    except Exception as e:
        logger.error(f"Failed to fetch data for {ticker}: {e}")