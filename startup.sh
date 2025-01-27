#!/bin/bash

echo "Starting the setup process..."

# Step 1: Run the model creation script
python fraudDetectionLive.py

# Step 2: Start the Flask app
echo "Starting the Flask application..."
python app.py &

# Step 3: Start the live data simulator (optional, in background)
echo "Starting the live data simulation..."
python simulate_live_data.py
