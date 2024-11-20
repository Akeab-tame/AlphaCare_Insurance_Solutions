import pandas as pd
from flask import Flask, request, render_template, jsonify
from model_load import load_model  # Adjusted import for model loading
import sys, os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.feature_engineering import DataPreprocessor  # Adjusted import statement for feature engineering
from scripts.modeling import RFMAnalysis  # Adjusted import for RFM analysis

# Initialize Flask app
app = Flask(__name__)

# Load the best model once at the start
model = load_model('model/best_model_pipeline.pkl')

@app.route('/', methods=['GET'])
def index():
    """Render the main index page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests from the web interface."""
    print("Incoming request data:", request.form)
    return handle_prediction()

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Handle prediction requests from the API."""
    return handle_prediction()

def handle_prediction():
    """Shared logic for handling prediction requests."""
    try:
        # Collect input data from the form or JSON
        input_data = {
            'TransactionId': int(request.form['TransactionId']) if 'TransactionId' in request.form else int(request.json['TransactionId']),
            'CustomerId': int(request.form['CustomerId']) if 'CustomerId' in request.form else int(request.json['CustomerId']),
            'ProductCategory': request.form['ProductCategory'] if 'ProductCategory' in request.form else request.json['ProductCategory'],
            'ChannelId': request.form['ChannelId'] if 'ChannelId' in request.form else request.json['ChannelId'],
            'Amount': float(request.form['Amount']) if 'Amount' in request.form else float(request.json['Amount']),
            'TransactionStartTime': pd.to_datetime(request.form['TransactionStartTime'], utc=True) if 'TransactionStartTime' in request.form else pd.to_datetime(request.json['TransactionStartTime'], utc=True),
            'PricingStrategy': int(request.form['PricingStrategy']) if 'PricingStrategy' in request.form else int(request.json['PricingStrategy'])
        }

        # Prepare input data as DataFrame
        input_df = pd.DataFrame([input_data])

        # Feature Engineering
        fe = DataPreprocessor()  # Assuming this class preprocesses the data
        input_df = fe.compute_aggregates(input_df)  # Compute aggregates
        input_df = fe.add_transactional_metrics(input_df)  # Add transactional metrics
        input_df = fe.generate_time_features(input_df)  # Generate time-based features

        # Encode categorical features
        categorical_cols = ['ProductCategory', 'ChannelId']
        input_df = fe.transform_categorical(input_df, categorical_cols)

        # Handle missing values and normalize features
        numeric_cols = input_df.select_dtypes(include='number').columns.tolist()
        exclude_cols = ['Amount', 'TransactionId']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

        input_df = fe.scale_numerical_features(input_df, numeric_cols, mode='standard')  # Scale numeric features

        # RFM Calculation
        rfm = RFMAnalysis(input_df.reset_index())  # Assuming RFMAnalysis is used for customer scoring
        rfm_df = rfm.compute_rfm_metrics()  # Calculate RFM metrics
        final_df = pd.merge(input_df, rfm_df, on='CustomerId', how='left')

        # Define all final features expected in the output
        final_features = [
            'PricingStrategy', 'Transaction_Count', 'Debit_Count', 'Credit_Count',
            'Debit_Credit_Ratio', 'Transaction_Month', 'Transaction_Year',
            'ProductCategory_financial_services', 'ChannelId_ChannelId_2',
            'ChannelId_ChannelId_3', 'Recency', 'Frequency'
        ]

        # Ensure all final features exist in the DataFrame and fill missing ones with 0
        final_df = final_df.reindex(columns=final_features, fill_value=0)

        # Make prediction
        prediction = model.predict(final_df)  # Use the loaded model to predict
        predicted_risk = 'Good' if prediction[0] == 0 else 'Bad'  # Assuming 0 = Good, 1 = Bad

        print(f"Predicted Risk: {predicted_risk}")
        
        # Return prediction result as JSON
        return jsonify({
            'customer_id': input_data['CustomerId'],
            'predicted_risk': predicted_risk
        })

    except ValueError as ve:
        print(f"ValueError: {str(ve)}")
        return jsonify({'error': 'Invalid input: ' + str(ve)}), 400
    except KeyError as ke:
        print(f"KeyError: {str(ke)}")
        return jsonify({'error': 'Missing input data: ' + str(ke)}), 400
    except Exception as e:
        print(f"General Exception: {str(e)}")
        return jsonify({'error': 'An error occurred: ' + str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)