#!/bin/bash
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/compile_check.py
python scripts/test_smoke_api_get.py
# deploy to fly.io
cd frontend
npm run build
cd ..
fly deploy