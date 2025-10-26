# Use a small official Python image
FROM python:3.10-slim

# Prevent Python from writing .pyc files & enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# System deps (optional but useful for slim image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements first (leverages Docker cache)
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY app.py .
COPY artifacts ./artifacts

# Expose the port your app uses
EXPOSE 5001

# Use gunicorn HTTP server (production-friendly)
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5001", "app:app"]