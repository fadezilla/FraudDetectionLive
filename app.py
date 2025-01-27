from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO
import os
import joblib
import logging
import pandas as pd
from datetime import datetime

model = joblib.load("fraud_detection_model.pkl")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get('FLASK_SECRET_KEY')
socketio = SocketIO(app)

# Logging configuration
logging.basicConfig(filename="predictions.log", level=logging.INFO)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Invalid input, 'features' key is required"}), 400

    features = pd.DataFrame(data["features"])
    probabilities = model.predict_proba(features)[:, 1]
    threshold = 0.22
    predictions = (probabilities > threshold).astype(int)

    # Emit predictions via WebSocket
    for i, prob in enumerate(probabilities):
        result = {
            "Input": features.iloc[i].to_dict(),
            "Prediction": int(predictions[i]),
            "Probability": float(prob),
        }
        logging.info(result)
        socketio.emit("new_prediction", result)  

    response = {"predictions": predictions.tolist(), "probabilities": probabilities.tolist()}
    return jsonify(response)

if __name__ == "__main__":
    socketio.run(app, debug=True, port=5000)
