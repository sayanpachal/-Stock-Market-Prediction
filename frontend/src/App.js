import React, { useState } from 'react';
import axios from 'axios';
import { Container, Form, Button, Image } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';

function App() {
  const [ticker, setTicker] = useState('');
  const [imageUrl, setImageUrl] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://127.0.0.1:8000/predict', { ticker }, { responseType: 'blob' });
      const url = URL.createObjectURL(new Blob([response.data]));
      setImageUrl(url);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <Container>
      <h1 className="mt-5">Stock Market Prediction</h1>
      <Form onSubmit={handleSubmit}>
        <Form.Group controlId="ticker">
          <Form.Label>Enter Stock Ticker Symbol</Form.Label>
          <Form.Control type="text" value={ticker} onChange={(e) => setTicker(e.target.value)} required />
        </Form.Group>
        <Button variant="primary" type="submit" className="mt-3">
          Predict
        </Button>
      </Form>
      {imageUrl && (
        <div className="mt-5">
          <h3>Predicted Stock Price</h3>
          <Image src={imageUrl} alt="Predicted Stock Price" fluid />
        </div>
      )}
    </Container>
  );
}

export default App;
