from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
import yfinance as yf
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model('keras_model.h5')

@app.post('/predict')
async def predict(request: Request):
    data = await request.json()
    ticker = data.get('ticker')

    df = yf.download(ticker, start='2010-01-01', end='2023-01-01')
    data_testing = df['Close'].values

    past_100_days = df.iloc[-100:, 3]
    final_df = pd.concat([past_100_days, pd.Series(data_testing)])

    input_data = final_df.values.reshape(-1, 1)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    input_data = scaler.fit_transform(input_data)

    x_test = []
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i, 0])

    x_test = np.array(x_test)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    y_predicted = model.predict(x_test)
    y_predicted = scaler.inverse_transform(y_predicted)

    # Generating the graph
    plt.figure(figsize=(14, 5))
    plt.plot(data_testing, color='blue', label='Actual Stock Price')
    plt.plot(y_predicted, color='red', label='Predicted Stock Price')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    return StreamingResponse(buf, media_type='image/png')
