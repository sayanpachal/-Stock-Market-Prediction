// frontend/src/components/PredictForm.js
import React, { useState } from 'react';

const PredictForm = () => {
  const [ticker, setTicker] = useState('AAPL');
  const [startDate, setStartDate] = useState('2020-01-01');
  const [endDate, setEndDate] = useState('2020-12-31');
  const [prediction, setPrediction] = useState(null);

  const handlePredict = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ ticker, start_date: startDate, end_date: endDate }),
      });

      if (response.ok) {
        const data = await response.json();
        setPrediction(data.predicted_price);
      } else {
        console.error('Failed to fetch prediction');
      }
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div>
      <h2>Stock Price Prediction</h2>
      <label>
        Ticker Symbol:
        <input
          type="text"
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
        />
      </label>
      <label>
        Start Date:
        <input
          type="date"
          value={startDate}
          onChange={(e) => setStartDate(e.target.value)}
        />
      </label>
      <label>
        End Date:
        <input
          type="date"
          value={endDate}
          onChange={(e) => setEndDate(e.target.value)}
        />
      </label>
      <button onClick={handlePredict}>Predict</button>
      {prediction !== null && <p>Predicted Price: ${prediction.toFixed(2)}</p>}
    </div>
  );
};

export default PredictForm;
