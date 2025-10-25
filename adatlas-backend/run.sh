#!/bin/bash
# Run script for AdAtlas Backend

# Activate virtual environment
source ../venv/bin/activate

# Run the application with uvicorn
cd adatlas-backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
