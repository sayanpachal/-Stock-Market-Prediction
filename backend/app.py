import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

start = "2020-01-01"
end = "2020-12-31"

st.title('Stock Trend Prediction')
user_input = st.text_input('Enter stock Ticker', 'AAPL')
df = yf.download(user_input, start=start, end=end)

st.subheader('Data from 2020')
st.write(df.describe())

# Visualization
st.subheader('Closing Price vs Time Chart')
ma100 = df['Close'].rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'])
plt.plot(ma100, 'r')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'])
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
st.pyplot(fig)

# Splitting data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

model = load_model('keras_model.h5')

# Testing part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Making predictions
y_predicted = model.predict(x_test)

scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final Graph
st.subheader('Prediction vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
