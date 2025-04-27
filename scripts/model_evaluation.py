import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scripts.lstm import predict_stock_price_lstm
#from tensorflow.keras.models import load_model
#from scripts.lstm import train_lstm_model

PREPROCESSED_DATA_PATH = r'C:\Users\Dell\VENVs\Stock_Advisor\data\processed\RELIANCE.BSE.csv'
df=pd.read_csv(PREPROCESSED_DATA_PATH)
def evaluate_lstm_model():

    y_true = df['Close'].values[-5:]
    y_pred, _= predict_stock_price_lstm('RELIANCE.BSE')
    print('Actual Stock Prices: ', y_true)
    print('Predicted Stock Prices: ', y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    print(f"MAPE: {mape:.2f}%")

    # Plot actual vs. predicted prices
    #plt.figure(figsize=(10, 6))
    #plt.plot(y_true, label='Actual Prices', color='blue')
    #plt.plot(y_pred, label='Predicted Prices', color='red')
    #plt.legend()
    #plt.title('LSTM Model Evaluation')
    #plt.show()