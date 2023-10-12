import numpy as np
import pandas as pd
#$import pandas_datareader as data
import yfinance as yf

import matplotlib.pyplot as plt
from keras.models import load_model
start = "2020-01-01"
end = "2020-12-31"

st.title('stock Trend Prediction')
user_input = st.text_input('Enter stock Ticker','AAPL')
df = data.DataReader(user_input,'yahoo',start,end)

st.subheader('Date from 2010 - 2019')
st.write(df.describe())

#visualisation
st.subheader('Closing price vs Time Chart')
ma100 = df.close.rolling(100).mean()
fig = plt.figure(figure = (12,6))
plt.plot(ma100)
plt.plot(df.close)
st.pyplot(fig)

st.subheader('Closing price vs Time Chart')
ma100 = df.close.rolling(100).mean()
ma200 = df.close.rolling(200).mean()
fig = plt.figure(figure = (12,6))
plt.plot(ma100,'r')
plt.plot(ma100, 'g')
plt.plot(df.close,'b')
st.pyplot(fig)

data_training = pd.DataFrame(fd['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][:int(int(len(df*0.70) int(len(df))))])

from sklearn.preprocessinng import MinMaxScaler
Scaler = MinMaxScaler(feature_range=(0,1))
data_training_array =Scaler.fit_traNSFORM(data_training)

x_train =[]
y_train = []

for i in range(100,data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])

x_train,y_train = np.array(x_train),np.array(y_train)

model = load_model('keras_model.h5')

x_test = []
y_test = []
for i in range(100,input_data.shape[0]):
    x_testappend(input_data[i-100:i])
    y_test.append(input_data[i,0])

'''
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def get_historical_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data


def plot_historical_prices(stock_data, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data['Close'], label=f'{ticker} Stock Price', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title(f'{ticker} Historical Stock Prices')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    ticker_symbol = "TSLA"  # Replace this with the desired stock ticker

    # Define the date range for the previous year
    end_date = "2010-01-01"
    start_date = "2020-12-31"

    # Step 1: Get historical stock price data for the previous year
    stock_data = get_historical_stock_data(ticker_symbol, start_date, end_date)

    # Step 2: Plot the historical stock prices
    plot_historical_prices(stock_data, ticker_symbol)
'''



'''
start = "2020-01-01"
end = "2020-12-31"
data = web.DataReader(name="TSLA", data_source='yahoo', start, end)
df = data.DataReader('AAPL','yahoo',start,end) 
df.head()
'''

