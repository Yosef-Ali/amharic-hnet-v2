# Multi-stage Dockerfile for Amharic H-Net Production Deployment
# Optimized for security, performance, and minimal attack surface

# Build stage
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash hnet

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements-production.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-production.txt

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app user
RUN useradd --create-home --shell /bin/bash --uid 1000 hnet

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY app/ ./app/
COPY --chown=hnet:hnet ../src/ ./src/

# Create necessary directories
RUN mkdir -p /app/models /app/logs /app/data && \
    chown -R hnet:hnet /app

# Copy trained model checkpoint
COPY --chown=hnet:hnet ../outputs/compact/final_model.pt ./models/final_model.pt

# Create environment file template
RUN echo "# Environment Configuration\n\
# Copy this to .env and configure for your environment\n\
ENVIRONMENT=production\n\
HOST=0.0.0.0\n\
PORT=8000\n\
WORKERS=1\n\
MODEL_PATH=/app/models/final_model.pt\n\
LOG_LEVEL=INFO\n\
ENABLE_AUTH=true\n\
API_KEY=your-secure-api-key-here\n\
REDIS_URL=redis://redis:6379/0\n\
ENABLE_METRICS=true\n\
ENABLE_CULTURAL_SAFETY=true\n\
" > /app/.env.example

# Security hardening
RUN chmod -R 755 /app && \
    chmod -R 644 /app/app/*.py && \
    chmod +x /app/app/main.py

# Switch to non-root user
USER hnet

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]