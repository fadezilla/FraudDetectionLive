from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO
from dotenv import load_dotenv
import subprocess
import os
import joblib
import logging
import pandas as pd
from datetime import datetime

load_dotenv()
model = joblib.load("fraud_detection_model.pkl")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY")
socketio = SocketIO(app)

# Logging configuration
logging.basicConfig(filename="predictions.log", level=logging.INFO)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start-simulation", methods=["POST"])
def start_simulation():
    subprocess.Popen(["Python", "simulate_live_data.py"])
    return jsonify({"message": "Simulation started"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port, debug=True)
