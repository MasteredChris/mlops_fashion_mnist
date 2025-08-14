FROM python:3.11-slim

# dipendenze di base
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install requirements prima per caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia pacchetto
COPY pyproject.toml ./
COPY src ./src

# Installa pacchetto in editable (veloce)
RUN pip install -e .

# directory output e dati
RUN mkdir -p /app/outputs /app/data

# Default: training CPU 3 epoche
CMD ["python", "-m", "src.mlops_fmnist.train", "--epochs", "3", "--cpu", "--output_dir", "./outputs"]
