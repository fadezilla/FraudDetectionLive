import pandas as pd
import requests
import time
import logging
import os

logging.basicConfig(filename="simulation.log", level=logging.INFO, format="%(asctime)s - %(message)s")

if not os.path.exists("creditcard.csv"):
    print("Downloading dataset from Kaggle...")
    os.system("kaggle datasets download -d mlg-ulb/creditcardfraud --unzip")
    print("Dataset downloaded and extracted.")

# Load the dataset
data = pd.read_csv("creditcard.csv")

# Process the data to match the model's requirements
data["NormalizedAmount"] = (data["Amount"] - data["Amount"].mean()) / data["Amount"].std()
data["NormalizedTime"] = (data["Time"] - data["Time"].mean()) / data["Time"].std()
data.drop(["Amount", "Time"], axis=1, inplace=True)

# Extract features for simulation
features = data.drop("Class", axis=1)

url = "https://fraud-detection-7f4v.onrender.com/predict"
simulation_running = False  # Global control flag
simulation_paused = False

def simulate_data():
    global simulation_running, simulation_paused

    print("Simulation process started...")
    for i in range(len(features)):
        while not simulation_running or simulation_paused:
            time.sleep(0.1)  # Wait until simulation is resumed

        row = features.iloc[i:i + 1].to_dict(orient="records")
        payload = {"features": row}

        response = requests.post(url, json=payload)

        if response.status_code == 200:
            logging.info(f"Input: {row[0]} - Prediction sent successfully")
        else:
            logging.error(f"Error: {response.status_code} - {response.text}")

        time.sleep(0.1)

simulate_data()
