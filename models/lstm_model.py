import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

#def build_lstm_model(input_shape):
#    model = Sequential([
#        LSTM(150, return_sequences=True, input_shape=input_shape),
#        Dropout(0.2),
#        LSTM(75, return_sequences=False),
#        Dropout(0.2),
#        LSTM(25, return_sequences=False),
#        Dropout(0.2),
#        #Dense(25),
#        Dense(1)  # Predicting one output value (price)
#   ])
#    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error', metrics=['mae'])
#    return model

def build_lstm_model(input_shape):
    inputs = Input(shape=(input_shape[0], input_shape[1]))  # Ensure 3D input: (time_steps, num_features)
    
    x = LSTM(100, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    
    x = LSTM(50, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    
    x = LSTM(25, return_sequences=False)(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error', metrics=['mae'])
    
    return model
