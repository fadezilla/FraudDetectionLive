from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO
from dotenv import load_dotenv
import os
import subprocess
import joblib
import logging
import pandas as pd

load_dotenv()
model = joblib.load("fraud_detection_model.pkl")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY")
socketio = SocketIO(app)

# Logging configuration
logging.basicConfig(filename="predictions.log", level=logging.INFO)

def download_dataset():
    if not os.path.exists("creditcard.csv"):
        print("Downloading dataset...")
        subprocess.run(["kaggle", "datasets", "download", "-d", "mlg-ulb/creditcardfraud", "--unzip"])
        print("Dataset downloaded and extracted.")

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

# WebSocket event handler for simulation control
@socketio.on("control_simulation")
def control_simulation(data):
    action = data.get("action")
    socketio.emit("control_simulation", {"action": action})  # Broadcast the action to the simulation script


if __name__ == "__main__":
    download_dataset()
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask app on port {port}")  # Debug logging
    socketio.run(app, host="0.0.0.0", port=port, debug=True)
