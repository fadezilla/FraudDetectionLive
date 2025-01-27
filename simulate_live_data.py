import pandas as pd
import requests
import time
import logging
import os
from socketio import Client

logging.basicConfig(filename="simulation.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Load dataset or download if not present
if not os.path.exists("creditcard.csv"):
    print("Downloading dataset from Kaggle...")
    os.system("kaggle datasets download -d mlg-ulb/creditcardfraud --unzip")
    print("Dataset downloaded and extracted.")

data = pd.read_csv("creditcard.csv")

# Process the dataset
data["NormalizedAmount"] = (data["Amount"] - data["Amount"].mean()) / data["Amount"].std()
data["NormalizedTime"] = (data["Time"] - data["Time"].mean()) / data["Time"].std()
data.drop(["Amount", "Time"], axis=1, inplace=True)

features = data.drop("Class", axis=1)  # Extract features for simulation
url = "https://fraud-detection-7f4v.onrender.com/predict"  # Local server URL for predictions

# SocketIO client to communicate with the Flask app
socket = Client()

# Simulation state flags
simulation_running = False
simulation_paused = False


def simulate_data():
    global simulation_running, simulation_paused

    print("Simulation process started.")
    for i in range(len(features)):
        # Wait until simulation is started and not paused
        while not simulation_running or simulation_paused:
            time.sleep(0.1)

        # Prepare and send a simulated transaction
        row = features.iloc[i:i + 1].to_dict(orient="records")
        payload = {"features": row}

        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                result = response.json()
                logging.info(f"Input: {row[0]} - Prediction: {result['predictions'][0]} - Probability: {result['probabilities'][0]:.2f}")
                print(f"Sent Input: {row[0]} | Prediction: {result['predictions'][0]}")
            else:
                logging.error(f"Error: {response.status_code} - {response.text}")
                print(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            logging.error(f"Exception occurred: {str(e)}")
            print(f"Exception occurred: {str(e)}")

        time.sleep(0.5)  # Simulate delay between transactions


# WebSocket event handlers
@socket.on("control_simulation")
def handle_control(data):
    global simulation_running, simulation_paused
    action = data.get("action")
    if action == "start":
        simulation_running = True
        simulation_paused = False
        print("Simulation started.")
    elif action == "pause":
        simulation_paused = True
        print("Simulation paused.")
    elif action == "restart":
        simulation_running = True
        simulation_paused = False
        print("Simulation restarted.")


# Connect to the WebSocket server
print("Connecting to WebSocket server...")
socket.connect("https://fraud-detection-7f4v.onrender.com")
print("Connected to WebSocket server.")

# Run the simulation
simulate_data()
