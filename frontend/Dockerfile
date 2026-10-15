FROM python:3.11-slim

WORKDIR /app

COPY frontend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend ./frontend
COPY data/test.csv ./data/test.csv
COPY data/test_sol.csv ./data/test_sol.csv

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]