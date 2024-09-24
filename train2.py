import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

# Load and clean data
data = pd.read_csv("data.csv", encoding='ISO-8859-1')
data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
data['Date'] = pd.to_datetime(data['Date'], format='%b %d, %Y')
data['Volume'] = data['Volume'].str.replace(',', '').astype(float)
price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
data[price_columns] = data[price_columns].apply(pd.to_numeric, errors='coerce')
data = data.dropna()

# Select features and target
X = data[['Open', 'High', 'Low', 'Volume']].values
y = data['Close'].values

# Feature scaling for LSTM
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Create sequences for LSTM
def create_sequences(X, y, time_step=60):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_step):
        X_seq.append(X[i:i+time_step])
        y_seq.append(y[i + time_step])
    return np.array(X_seq), np.array(y_seq)

time_step = 60
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_step)

# Train-test split
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# Reshape data to be suitable for LSTM (samples, timesteps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, X_train.shape[2])))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=64, epochs=20)

# Save the model
model.save('lstm_stock_price_model.h5')
print("LSTM model trained and saved as 'lstm_stock_price_model.h5'")

# Make predictions
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Evaluate the model
mse = mean_squared_error(scaler_y.inverse_transform(y_test), y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(scaler_y.inverse_transform(y_test), y_pred)

# Print the results
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")
