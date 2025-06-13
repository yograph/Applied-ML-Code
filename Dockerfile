# Dockerfile

# 1) Base image with Python 3.10
FROM python:3.10-slim AS base
WORKDIR /app

# 2) Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copy all your code & model weights
COPY . .

# 4) Streamlit service
FROM base AS streamlit
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "streamlitt_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]

# 5) FastAPI service
FROM base AS api
EXPOSE 8000
CMD ["uvicorn", "fasting_api:app", "--host", "0.0.0.0", "--port", "8000"]
