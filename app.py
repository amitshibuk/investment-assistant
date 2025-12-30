import os
import json
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content

load_dotenv()

# --- Prediction Import (Placeholder logic preserved) ---
try:
    from src.prediction_model import load_and_predict
except ImportError:
    print("Warning: Could not import 'load_and_predict'. Using placeholder.")
    def load_and_predict(ticker, days_ahead):
        # Placeholder simulation for testing
        import random
        start_price = 150.0
        return [start_price + random.uniform(-5, 5) * i for i in range(1, days_ahead + 1)]

app = Flask(__name__)

# --- Gemini SDK Configuration ---
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("Warning: API_KEY is not set.")

genai.configure(api_key=API_KEY)

# Initialize the model
# 'gemini-1.5-flash' is recommended for low-latency tasks like this
model = genai.GenerativeModel('gemini-2.5-flash-lite') 

# --------------------------------

TICKER_MAP = {
    "Apple": "AAPL", "Tesla": "TSLA", "Google": "GOOGL",
    "Amazon": "AMZN", "Microsoft": "MSFT", "Meta": "META", "Netflix": "NFLX"
}

def get_entities_with_llm(query):
    """
    Uses the Gemini SDK to extract structured entities from natural language.
    Enforces JSON output via generation_config.
    """
    system_prompt = """
    You are an expert at understanding financial queries. Extract these entities:
    - task: The user's goal (e.g., 'predict', 'compare').
    - metric: The financial metric (e.g., 'stock price').
    - company: The company name.
    - time_period: A dictionary with 'value' (int) and 'unit' (string).
    
    Return ONLY a JSON object. If an entity is missing, omit it.
    """

    try:
        response = model.generate_content(
            f"{system_prompt}\n\nQuery: {query}",
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json"
            )
        )
        
        # The SDK handles the API call; we just parse the text result
        return json.loads(response.text)

    except Exception as e:
        print(f"Error in entity extraction: {e}")
        return {"error": str(e)}

def summarize_predictions_with_llm(company, ticker, days_ahead, predictions_data):
    """
    Uses the Gemini SDK to generate a natural language summary.
    """
    # Calculate stats
    start_price = predictions_data[0]['predicted_close']
    end_price = predictions_data[-1]['predicted_close']
    change = end_price - start_price
    percent_change = (change / start_price) * 100 if start_price != 0 else 0
    
    trend = "an upward" if change > 0 else "a downward"
    if abs(percent_change) < 1: 
        trend = "a relatively stable"

    # Context for the LLM
    prediction_details = (
        f"Company: {company} ({ticker})\n"
        f"Period: {days_ahead} days\n"
        f"Trend: {trend} ({percent_change:.2f}%)\n"
        f"Start: ${start_price:.2f}, End: ${end_price:.2f}\n"
    )

    prompt = f"""
    You are a financial analyst. Summarize these stock predictions concisely:
    {prediction_details}
    
    Requirements:
    1. Mention the trend, start/end prices, and % change.
    2. Tone: Informative but cautious.
    3. MANDATORY: End with "Disclaimer: This is an AI-generated prediction and not financial advice."
    """

    try:
        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        print(f"Error in summarization: {e}")
        return "Could not generate summary."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Invalid request"}), 400

    query = data['query']
    
    # 1. Extract Entities
    entities = get_entities_with_llm(query)
    if "error" in entities:
        return jsonify(entities), 500

    task = entities.get('task')
    metric = entities.get('metric')
    company = entities.get('company')
    time_period = entities.get('time_period')

    if task == 'predict' and 'stock' in metric and company and time_period:
        ticker = TICKER_MAP.get(company.capitalize())
        if not ticker:
            return jsonify({"error": f"Company '{company}' not supported."}), 400

        # Parse days
        days_ahead = 1
        if time_period:
            unit = time_period.get('unit', 'day')
            val = time_period.get('value', 1)
            if 'week' in unit: days_ahead = val * 7
            elif 'month' in unit: days_ahead = val * 30
            else: days_ahead = val
        
        # 3. Get Prediction
        predictions_list = load_and_predict(ticker, days_ahead)
        if not predictions_list:
            return jsonify({"error": "Model failed."}), 500
        
        formatted_predictions = [
            {"day": i+1, "predicted_close": round(p, 2)}
            for i, p in enumerate(predictions_list)
        ]

        # 4. Summarize
        summary = summarize_predictions_with_llm(company, ticker, days_ahead, formatted_predictions)

        return jsonify({
            "company": company,
            "ticker": ticker,
            "days_ahead": days_ahead,
            "predictions": formatted_predictions,
            "summary": summary
        })

    return jsonify({"message": "Query not understood.", "entities": entities}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)