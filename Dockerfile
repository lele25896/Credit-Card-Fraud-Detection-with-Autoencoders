FROM python:3.11-slim

WORKDIR /app

# Install PyTorch CPU-only in a separate layer so Docker caches it independently
# from application code changes (~250 MB wheel, avoids re-downloading on every build)
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.5.1 \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy model artifacts (creditcard.csv is excluded via .dockerignore)
COPY autoencoder.pt  .
COPY weights.npy     .
COPY scaler.pkl      .
COPY config.json     .

# Copy application code
COPY app/ ./app/

# Run as non-root user (Cloud Run best practice)
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

# Cloud Run injects $PORT (default 8080); fall back to 8080 for local runs
EXPOSE 8080

CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
