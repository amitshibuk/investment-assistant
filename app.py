import os
import requests
import json
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()
# The 'load_and_predict' function is still needed from your prediction model file.
# Note: Ensure you have a 'src/prediction_model.py' file with this function.
try:
    from src.prediction_model import load_and_predict
except ImportError:
    print("Warning: Could not import 'load_and_predict'. The prediction functionality will not work.")
    # Define a placeholder function if the import fails, so the app can still start.
    def load_and_predict(ticker, days_ahead):
        return None

app = Flask(__name__)

# --- Gemini API Configuration ---
API_KEY = os.environ.get("GEMINI_API_KEY")
if API_KEY == "YOUR_API_KEY":
    print("Warning: API_KEY is not set. Please set the GEMINI_API_KEY environment variable.")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={API_KEY}"
# --------------------------------

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

def get_entities_with_llm(query):
    """
    Uses the Gemini API to extract structured entities from a natural language query.
    """
    # This prompt instructs the model to act as an NLU engine and return JSON.
    system_prompt = """
    You are an expert at understanding financial queries. Your task is to extract specific pieces of information (entities) from a user's query.
    Extract the following entities:
    - task: The user's goal (e.g., 'predict', 'compare', 'summarize').
    - metric: The financial metric of interest (e.g., 'stock price', 'revenue').
    - company: The name of the company.
    - time_period: A dictionary with 'value' (integer) and 'unit' (e.g., 'day', 'week', 'month').
    
    If an entity is not present, do not include it in the output.
    Respond with ONLY a valid JSON object.
    """

    # Construct the payload for the Gemini API to get a JSON response
    payload = {
        "contents": [{"parts": [{"text": f"Query: \"{query}\""}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {"responseMimeType": "application/json"}
    }
    
    headers = {'Content-Type': 'application/json'}
    
    try:
        # Make the API request
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        
        api_response = response.json()
        
        # Safely extract the JSON string from the response
        json_text = api_response['candidates'][0]['content']['parts'][0]['text']
        entities = json.loads(json_text)
        return entities
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        return {"error": f"API request failed: {e}"}
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"Error parsing Gemini API response: {e}")
        print(f"Raw response: {response.text}")
        return {"error": "Failed to parse API response."}
    
def summarize_predictions_with_llm(company, ticker, days_ahead, predictions_data):
    """
    Uses the Gemini API to generate a natural language summary of the prediction data.
    """
    # Calculate the overall trend and percentage change
    start_price = predictions_data[0]['predicted_close']
    end_price = predictions_data[-1]['predicted_close']
    change = end_price - start_price
    percent_change = (change / start_price) * 100 if start_price != 0 else 0
    
    trend = "an upward" if change > 0 else "a downward"
    if abs(percent_change) < 1: # Consider small changes as stable
        trend = "a relatively stable"

    # The prompt provides context, data, and instructions for the LLM
    system_prompt = f"""
    You are a helpful financial analyst. Your task is to summarize stock price predictions in a concise, easy-to-understand paragraph.
    - Start with a clear introductory sentence.
    - Mention the overall trend (e.g., upward, downward, stable).
    - State the predicted price at the beginning and end of the period.
    - Mention the approximate percentage change over the period.
    - Keep the tone informative but cautious.
    - ALWAYS include this disclaimer at the end, on a new line: "Disclaimer: This is an AI-generated prediction and not financial advice."
    """

    # Create a user-friendly string from the prediction data
    prediction_details = (
        f"Company: {company} ({ticker})\n"
        f"Prediction Period: {days_ahead} days\n"
        f"Overall Trend: {trend} trend with a {percent_change:.2f}% change.\n"
        f"Starting Predicted Price: ${start_price:.2f}\n"
        f"Ending Predicted Price: ${end_price:.2f}\n"
    )

    try:
        # We want a text response, not JSON, so we adjust the payload
        payload = {
            "contents": [{
                "parts": [{"text": f"Please summarize these prediction details:\n\n{prediction_details}"}]
            }],
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": {"responseMimeType": "text/plain"}
        }
        
        headers = {'Content-Type': 'application/json'}
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=45)
        response.raise_for_status()

        api_response = response.json()
        summary = api_response['candidates'][0]['content']['parts'][0]['text']
        return summary.strip()

    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API for summarization: {e}")
        return "Could not generate a summary due to an API error."
    except (KeyError, IndexError) as e:
        print(f"Error parsing Gemini summarization response: {e}")
        return "Could not generate a summary due to a response parsing error."



@app.route('/')
def home():
    """Render the simple HTML frontend."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.
    Takes a JSON query, uses an LLM to process it, and returns a stock price prediction.
    """
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Invalid request. 'query' field is required."}), 400

    query = data['query']
    
    # 1. Process the user's query with the LLM to get structured entities
    entities = get_entities_with_llm(query)
    if "error" in entities:
        return jsonify({"error": f"Error in query processing: {entities['error']}"}), 500

    # 2. Check if the extracted entities match the prediction task
    task = entities.get('task')
    metric = entities.get('metric')
    company = entities.get('company')
    time_period = entities.get('time_period')

    if task == 'predict' and metric == 'stock price' and company and time_period:
        ticker = TICKER_MAP.get(company.capitalize())
        if not ticker:
            # Add the detected company name for better error feedback
            return jsonify({"error": f"Company '{company}' not supported.", "detected_entities": entities}), 400

        # Calculate days_ahead from the extracted time_period
        days_ahead = 1
        if time_period:
            unit = time_period.get('unit', 'day')
            value = time_period.get('value', 1)
            if 'week' in unit:
                days_ahead = value * 7
            elif 'month' in unit:
                days_ahead = value * 30
            else: # 'day'
                days_ahead = value
        
        # 3. Load the pre-trained model and get the prediction
        try:
            predictions_list = load_and_predict(ticker, days_ahead)
            
            if predictions_list is None:
                 return jsonify({"error": f"Could not generate prediction for {ticker}. The prediction model might not be available or trained."}), 500
            
            formatted_predictions = [
                {"day": i+1, "predicted_close": round(price, 2)}
                for i, price in enumerate(predictions_list)
            ]

            summary = summarize_predictions_with_llm(company, ticker, days_ahead, formatted_predictions)

            # Format the successful response
            response = {
                "company": company,
                "ticker": ticker,
                "days_ahead": days_ahead,
                "predictions": formatted_predictions,
                "summary": summary
            }
            return jsonify(response)

        except Exception as e:
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    else:
        # If the LLM couldn't extract the right entities, return a helpful message.
        unsupported_response = {
            "message": "This MVP currently only supports stock price predictions (e.g., 'predict apple stock for 7 days').",
            "detected_entities": entities
        }
        return jsonify(unsupported_response), 400

if __name__ == '__main__':
    # For production, use a proper WSGI server like Gunicorn or Waitress
    app.run(debug=True, port=5000)

