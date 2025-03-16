#!/bin/bash

# Create required directories
mkdir -p workdir data visualizations

# Setup Python environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python launch.py
