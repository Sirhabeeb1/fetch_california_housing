from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load("regmodel.joblib")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Linear Regression API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data
        data = request.get_json()
        features = data.get("features")  # Expecting a list of features

        if not features or not isinstance(features, list):
            return jsonify({"error": "Invalid input format. Expected a list of features."}), 400

        # Convert input to numpy array and reshape
        input_array = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_array)

        # Return the result
        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
