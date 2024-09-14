
# Stock Market Prediction

This project is designed to predict stock prices using machine learning techniques. It includes a Python backend for handling data processing and model predictions, and a React frontend for user interaction.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)

## Introduction

The Stock Market Prediction project aims to predict stock prices using historical data and machine learning models. It provides an intuitive web interface for users to select stock ticker symbols and view predicted stock prices along with graphical representations.

## Features

- Predict stock prices for selected ticker symbols.
- Display historical stock prices and predicted future prices in graphical format.
- Interactive and user-friendly web interface.
- Backend API for handling predictions.

## Technologies Used

- **Backend:** Python, FastAPI, TensorFlow, pandas, yfinance
- **Frontend:** React, React Bootstrap
- **Deployment:** Uvicorn

## Installation

### Backend

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/stock-market-prediction.git
    ```
2. Navigate to the backend directory:
    ```sh
    cd stock-market-prediction/backend
    ```
3. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
4. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```
5. Start the backend server:
    ```sh
    uvicorn api:app --reload
    ```

### Frontend

1. Navigate to the frontend directory:
    ```sh
    cd ../frontend
    ```
2. Install the required packages:
    ```sh
    npm install
    ```
3. Start the frontend server:
    ```sh
    npm start
    ```

## Usage

1. Start both the backend and frontend servers as described in the installation section.
2. Open your web browser and go to `http://localhost:3000`.
3. Select a stock ticker symbol, and click on "Predict" to view the predicted stock prices.

## API Endpoints

- `GET /`: Health check endpoint.
- `POST /predict`: Predict stock prices based on historical data.
  - **Request Body:**
    ```json
    {
      "ticker": "AAPL",
      "start_date": "2022-01-01",
      "end_date": "2023-01-01"
    }
    ```
  - **Response:**
    ```json
    {
      "predictions": [150.0, 151.2, 152.4, ...]
    }
    ```


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.
