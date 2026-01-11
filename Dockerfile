# -------- Stage 1: Base Python image --------
FROM python:3.10-slim
ENV DOCKER_ENV=true

# -------- Stage 2: Set working directory --------
WORKDIR /app


# -------- Stage 3: Copy required files --------
COPY src/api src/api
# COPY mlruns mlruns
# COPY mlartifacts mlartifacts

COPY artifacts artifacts
COPY data/processed/scaler.pkl data/processed/scaler.pkl

# This creates the directory structure and copies the file
COPY data/processed/scaler.pkl data/processed/scaler.pkl 
COPY requirements-inference.txt .

# -------- Stage 4: Install dependencies --------
RUN pip install --no-cache-dir -r requirements-inference.txt

# -------- Stage 5: Expose API port --------
EXPOSE 8000

# -------- Stage 6: Run FastAPI server --------
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
