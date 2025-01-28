#!/bin/bash
echo "Starting the Flask application..."
python app.py  # Remove the & to run in the foreground

echo "Downloading dataset if not present..."
if [ ! -f "creditcard.csv" ]; then
    kaggle datasets download -d mlg-ulb/creditcardfraud --unzip
fi
echo "Setup complete. Visit the web interface to start the simulation."