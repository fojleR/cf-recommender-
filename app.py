from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from tensorflow import keras
import requests
from urllib.parse import quote
import os

app = Flask(__name__)
CORS(app)

# Load the trained Keras model
model = keras.models.load_model("problem_recommender.h5")

def convert_verdict(verdict):
    return 1 if verdict == 'OK' else 0

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    handle = data.get("handle", "")

    if not handle:
        return jsonify({"error": "No handle provided"}), 400

    try:
        encoded_handle = quote(handle)
        submission_url = f"https://codeforces.com/api/user.status?handle={encoded_handle}&from=1&count=10000"
        submission_data = requests.get(submission_url).json()

        if submission_data['status'] != 'OK':
            return jsonify({"error": "Handle not found"}), 404

        user_data = pd.DataFrame(submission_data['result'])
        user_data['verdict'] = user_data['verdict'].apply(convert_verdict)
        # More preprocessing if needed...

        # Create feature array (dummy for now)
        features = np.random.rand(1, model.input_shape[1])  # Replace with actual features

        predictions = model.predict(features)
        top_index = np.argmax(predictions)
        return jsonify({"suggested_problem_id": int(top_index)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
