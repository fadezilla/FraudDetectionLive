#!/bin/bash
echo "Starting the Flask application..."
python app.py
echo "sending simulation.."
python simulate_live_data.py