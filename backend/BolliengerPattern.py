import yfinance as yf
import matplotlib.pyplot as plt

def get_historical_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def calculate_bollinger_bands(data, window=20, num_std=2):
    sma = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    upper_band = sma + (num_std * rolling_std)
    lower_band = sma - (num_std * rolling_std)
    return sma, upper_band, lower_band

def plot_bollinger_bands(data, ticker, sma, upper_band, lower_band):
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label=ticker, color='blue')
    plt.plot(sma, label='SMA', color='orange')
    plt.plot(upper_band, label='Upper Band', color='green')
    plt.plot(lower_band, label='Lower Band', color='red')
    plt.fill_between(data.index, lower_band, upper_band, alpha=0.2, color='gray')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Bollinger Bands Pattern')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    ticker_symbol = 'AAPL'   # Replace this with the desired stock ticker
    start_date = '2022-01-01' # Replace this with your desired start date
    end_date = '2023-01-01'   # Replace this with your desired end date

    # Step 1: Get historical stock price data
    stock_data = get_historical_stock_data(ticker_symbol, start_date, end_date)

    # Step 2: Calculate Bollinger Bands
    sma, upper_band, lower_band = calculate_bollinger_bands(stock_data)

    # Step 3: Plot Bollinger Bands Pattern
    plot_bollinger_bands(stock_data, ticker_symbol, sma, upper_band, lower_band)
