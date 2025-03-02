FROM python:3.9-slim

WORKDIR /app

# Install system dependencies required for PDF processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create data directory
RUN mkdir -p /app/data
RUN mkdir -p /app/vectorstore

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5001

# Expose the port the app runs on
EXPOSE 5001

# Command to run the application
CMD ["python", "app.py"]