FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY . .

# HF Spaces uses port 7860
ENV PORT=7860
ENV PRELOAD_MODELS=0

EXPOSE 7860

CMD ["gunicorn", "backend_api:app", "--bind", "0.0.0.0:7860", "--timeout", "300", "--workers", "1", "--threads", "2"]
