#!/bin/bash
set -e
cd frontend
echo "Installing frontend dependencies..."
npm install
echo "Building frontend..."
npm run build
cd ..
echo "Activating virtual environment..."
source .venv/bin/activate
echo "Installing backend dependencies..."
pip install -r requirements.txt
python app.py