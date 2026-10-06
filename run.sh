#!/usr/bin/env bash
# Starts the FastAPI backend (port 8000) and the Streamlit dashboard (port 8501).
set -e
cd "$(dirname "$0")"
(cd backend && uvicorn main:app --host 0.0.0.0 --port 8000) &
API_PID=$!
trap "kill $API_PID 2>/dev/null" EXIT
sleep 3
streamlit run frontend/app.py --server.port 8501
