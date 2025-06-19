from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import os
import logging

MODEL_FILE = "iris_model.pkl"
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

def train_and_save_model(path):
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, path)
    logging.info(f"Model saved to '{path}'")

def load_or_train_model(path):
    if not os.path.exists(path):
        train_and_save_model(path)
    return joblib.load(path)

model = load_or_train_model(MODEL_FILE)

@app.route("/", methods=["GET"])
def home():
    return jsonify(message="Iris Prediction API is running.")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        features = data.get("features")
        if not features or not isinstance(features, list):
            raise ValueError("Invalid input: 'features' must be a list of numbers.")
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)
        return jsonify(prediction=int(prediction[0]))
    except Exception as e:
        logging.exception("Prediction error")
        return jsonify(error=str(e)), 400

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
