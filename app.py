from flask import Flask, request, jsonify
import pandas as pd
import pickle

# Initialize Flask app
app = Flask(__name__)
import joblib

# Define the paths to the saved model and scaler files
model_path_scaled = 'logistic_regression_model_scaled.joblib'
scaler_path = 'scaler.joblib'

# Load the trained model
loaded_model = joblib.load(model_path_scaled)
print(f"Model '{model_path_scaled}' loaded successfully.")

# Load the fitted scaler
loaded_scaler = joblib.load(scaler_path)
print(f"Scaler '{scaler_path}' loaded successfully.")
# Load trained model and scaler
try:
    with open("model.pkl", "rb") as f:
        loaded_model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        loaded_scaler = pickle.load(f)

except Exception as e:
    print(f"Error loading model or scaler: {str(e)}")


@app.route('/')
def home():
    return "Diabetes Prediction API is Running!"


@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    data = request.get_json()

    if data is None:
        return jsonify({'error': 'No JSON data received'}), 400

    # Expected features (same order as training)
    expected_features = [
        'Pregnancies',
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI',
        'DiabetesPedigreeFunction',
        'Age'
    ]

    # Check for missing features
    if not all(feature in data for feature in expected_features):
        missing_features = [
            feature for feature in expected_features if feature not in data
        ]
        return jsonify({
            'error': f"Missing features in request: {', '.join(missing_features)}"
        }), 400

    try:
        # Convert JSON to DataFrame (2D format required)
        input_df = pd.DataFrame([data])

        # Ensure correct column order
        input_df = input_df[expected_features]

    except Exception as e:
        return jsonify({
            'error': f"Error converting data to DataFrame: {str(e)}"
        }), 400

    try:
        # Scale input data
        input_scaled = loaded_scaler.transform(input_df)

    except Exception as e:
        return jsonify({
            'error': f"Error scaling input data: {str(e)}"
        }), 500

    try:
        # Make prediction
        prediction = loaded_model.predict(input_scaled)
        prediction_result = int(prediction[0])

        # Convert to readable output
        if prediction_result == 1:
            result_message = "Diabetes Positive"
        else:
            result_message = "Diabetes Negative"

        return jsonify({
            'prediction': result_message
        }), 200

    except Exception as e:
        return jsonify({
            'error': f"Error during prediction: {str(e)}"
        }), 500


if __name__ == '__main__':
    app.run(debug=True)
