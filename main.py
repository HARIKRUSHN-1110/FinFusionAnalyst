# Main script to orchestrate the entire project
import pandas as pd
from stock_utils.logger import setup_logging_config, get_logger
from scripts.fetch_data import get_nifty_50_data
from datetime import timedelta, datetime
from scripts.data_preprocessing import add_technical_features
from scripts.lstm import train_lstm_model, scale_and_save_data, predict_stock_price_lstm
from scripts.monte_carlo import monte_carlo_simulation, predict_stock_price_monte_carlo, plot_monte_carlo
from scripts.model_evaluation import evaluate_lstm_model
#from scripts.agent import ai_stock_advisor
#from apikeytest import get_stock_insights

#setup_logging_config()

logger = get_logger(__name__)

#if __name__ == "__main__":
#    logger.info("Starting the project...")
#    logger.warning("This is a warning message.")
#    logger.error("This is an error message.")
#    logger.debug("This is a debug message.")

#if __name__ == "__main__":
    #end_date = datetime.strptime(end_date, '%Y-%m-%d')
    #start_date = datetime.strptime(start_date, '%Y-%m-%d')

    #keep this main start-end dates
    #end_date = datetime.today().strftime('%Y-%m-%d')
    #start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    #start_date = "2021-01-01"
    #end_date = "2022-01-01"

    #logger.info("Fetching Nifty 50 data...")
    #get_nifty_50_data()
    #logger.info("Fetched Nifty 50 data completely")
    
    #add_technical_features()
    #logger.info("Successfully added technical features to the data")
    
    #scale_and_save_data()
    
    #train_lstm_model()
    #logger.info("Training of LSTM model completed")

    #predict_stock_price_lstm('RELIANCE.BSE')
    #logger.info("Predicted stock price for RELiance.NS")
    #ticker = "RELIANCE.BSE"
    
    #future_dates, predicted_prices = predict_stock_price_monte_carlo(ticker)
    #print("Predicted Prices:")
    #for date, price in zip(future_dates, predicted_prices):
    #    print(f"{date.date()}: {price:.2f}")
    
    #evaluate_lstm_model()

    #company = "Reliance Industries"
    #insights = get_stock_insights(company)
    #print(f"Stock Insights for {company}:\n{insights}")