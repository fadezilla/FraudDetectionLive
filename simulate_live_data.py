import pandas as pd
import requests
import time
import os
import subprocess
from socketio import Client

if not os.path.exists("creditcard.csv"):
    print("Downloading dataset from Kaggle...")
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", "mlg-ulb/creditcardfraud", "--unzip"], check=True)
        print("Dataset downloaded and extracted.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        raise

time.sleep(10)
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
current_index = 0  # Track the current transaction index
batch_size = 20

def simulate_data():
    global simulation_running, simulation_paused, current_index

    print("Simulation process started.")
    while True:
        # Wait until simulation is started and not paused
        while not simulation_running or simulation_paused:
            time.sleep(0.1)

        # Prepare and send a simulated transaction
        batch = features.iloc[current_index : current_index + batch_size].to_dict(orient="records")
        payload = {"features": batch}

        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                result = response.json()
                for i, pred in enumerate(result["predictions"]):
                    print(f"Sent Input: {batch[i]} | Prediction: {pred}")
            else:
                print(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Exception occurred: {str(e)}")

        current_index += batch_size
        if current_index >= len(features):
            current_index = 0  # Restart from the beginning



# WebSocket event handlers
@socket.on("control_simulation")
def handle_control(data):
    global simulation_running, simulation_paused, current_index
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
        current_index = 0  # Reset to the first transaction
        print("Simulation restarted.")


# Connect to the WebSocket server
print("Connecting to WebSocket server...")
socket.connect("https://fraud-detection-7f4v.onrender.com")
print("Connected to WebSocket server.")

# Run the simulation
simulate_data()