from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO
from dotenv import load_dotenv
import threading
import os
import joblib
import logging
import pandas as pd
from time import sleep
from datetime import datetime

load_dotenv()
model = joblib.load("fraud_detection_model.pkl")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY")
socketio = SocketIO(app)

# Logging configuration
logging.basicConfig(filename="predictions.log", level=logging.INFO)

simulation_running = False  # Global flag for simulation
simulation_paused = False

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

@socketio.on("control_simulation")
def control_simulation(data):
    global simulation_running, simulation_paused

    action = data.get("action")
    if action == "start":
        simulation_running = True
        simulation_paused = False
        socketio.emit("status", {"message": "Simulation started."})
    elif action == "pause":
        simulation_paused = True
        socketio.emit("status", {"message": "Simulation paused."})
    elif action == "restart":
        simulation_running = True
        simulation_paused = False
        socketio.emit("status", {"message": "Simulation restarted."})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port, debug=True)
