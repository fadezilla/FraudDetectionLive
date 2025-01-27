import pandas as pd
import requests
import time
import logging

logging.basicConfig(filename="simulation.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Load the dataset
data = pd.read_csv("creditcard.csv")

# Process the data to match the model's requirements
data["NormalizedAmount"] = (data["Amount"] - data["Amount"].mean()) / data["Amount"].std()
data["NormalizedTime"] = (data["Time"] - data["Time"].mean()) / data["Time"].std()
data.drop(["Amount", "Time"], axis=1, inplace=True)

# Extract features for simulation
features = data.drop("Class", axis=1)
actual_labels = data["Class"]

url = "http://127.0.0.1:5000/predict"

print("Starting simulation...")

# Simulate sending rows one by one
for i in range(len(features)):
    try:
        row = features.iloc[i:i + 1].to_dict(orient="records")
        payload = {"features": row}

        response = requests.post(url, json=payload)

        if response.status_code == 200:
            result = response.json()
            prediction = result["predictions"][0]
            probability = result["probabilities"][0]

            logging.info(f"Input: {row[0]} - Prediction: {prediction} - Probability: {probability:.2f}")

            print(f"Sent Input: {row[0]}")
            print(f"Prediction: {'Fraudulent' if prediction else 'Non-Fraudulent'}")
            print(f"Probability: {probability:.2%}")
        else:
            logging.error(f"Error: {response.status_code} - {response.text}")
            print(f"Error: {response.status_code} - {response.text}")

    except Exception as e:
        logging.error(f"Exception occurred: {str(e)}")
        print(f"Exception occurred: {str(e)}")

    #delay to simulate real-time streaming
    time.sleep(0.1)

print("Simulation completed.")
