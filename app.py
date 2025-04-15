from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import requests
import time
from urllib.parse import quote
from utils import fetch_user_data

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the model
with open("model/problem_recommender.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    handle = data.get("handle")

    if not handle:
        return jsonify({"error": "No handle provided"}), 400

    try:
        user_df = fetch_user_data(handle)
        predictions = model.predict(user_df)  # You may need to adapt this line
        return jsonify({"recommendations": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
