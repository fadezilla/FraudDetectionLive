#!/bin/bash
echo "Starting the Flask application..."
python app.py
sleep(5)
echo "Starting the live data simulation..."
python simulate_live_data.py