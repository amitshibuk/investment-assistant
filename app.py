from flask import Flask, request, jsonify, render_template
from src.query_processing import intent_classification
from src.prediction_model import load_and_predict # <-- This line was missing
import json

app = Flask(__name__)

# A simple mapping from company names to ticker symbols
TICKER_MAP = {
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Google": "GOOGL",
    "Amazon": "AMZN",
    "Microsoft": "MSFT",
    "Meta": "META",
    "Netflix": "NFLX"
}

@app.route('/')
def home():
    """Render the simple HTML frontend."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.
    Takes a JSON query, processes it, and returns a stock price prediction.
    """
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Invalid request. 'query' field is required."}), 400

    query = data['query']
    
    # 1. Process the user's query to get intent and entities
    try:
        entities = intent_classification(query)
    except Exception as e:
        return jsonify({"error": f"Error in query processing: {str(e)}"}), 500

    # 2. Check if the intent is to predict a stock price
    task = entities.get('task')
    metric = entities.get('metric')
    company = entities.get('company')
    time_period = entities.get('time_period')

    if task == 'predict' and metric == 'stock price' and company and time_period:
        ticker = TICKER_MAP.get(company.capitalize())
        if not ticker:
            return jsonify({"error": f"Company '{company}' not supported."}), 400

        # Calculate days ahead from the time_period entity
        days_ahead = 1
        if time_period:
            unit = time_period.get('unit', 'day')
            value = time_period.get('value', 1)
            if 'week' in unit:
                days_ahead = value * 7
            elif 'month' in unit:
                days_ahead = value * 30
            else: # day
                days_ahead = value
        
        # 3. Load the pre-trained model and get the prediction
        try:
            predictions = load_and_predict(ticker, days_ahead)
            if predictions is None:
                 return jsonify({"error": f"Could not generate prediction for {ticker}. Model may not be trained yet."}), 500
            
            # Format the response
            response = {
                "company": company,
                "ticker": ticker,
                "days_ahead": days_ahead,
                "predictions": [
                    {"day": i+1, "predicted_close": round(price, 2)}
                    for i, price in enumerate(predictions)
                ]
            }
            return jsonify(response)

        except Exception as e:
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    else:
        # Placeholder for other intents like 'compare' or 'summarize'
        unsupported_response = {
            "message": "This MVP currently only supports stock price predictions (e.g., 'predict apple stock for 7 days').",
            "detected_entities": entities
        }
        return jsonify(unsupported_response), 400


if __name__ == '__main__':
    # Note: For production, use a proper WSGI server like Gunicorn or Waitress
    app.run(debug=True, port=5000)

