# Use an official lightweight Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for building C++ extensions (ChromaDB/pysqlite)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install 'uv' for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvbin/uv
ENV PATH="/uvbin:${PATH}"

# Copy dependency files first to leverage Docker cache
COPY pyproject.toml uv.lock ./

# Install dependencies into the system environment
RUN uv pip install --system .

# Copy the rest of the application code
COPY . .

# Create the data directory for the Vector DB
RUN mkdir -p data/vector_db

# Pre-download the model and build the index (Part 1 & 2)
# This ensures the container starts with data ready to query
RUN python scripts/ingest_and_cluster.py

# Expose the FastAPI port
EXPOSE 8000

# Start the service
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]