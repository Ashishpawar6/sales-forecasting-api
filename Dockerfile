FROM python:3.12-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY forecasting ./forecasting
COPY api ./api

# Train on the bundled synthetic data at build time so the image is self-contained.
# To serve your own model instead, mount trained artifacts at /app/artifacts.
RUN python -m forecasting.train --out artifacts

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
