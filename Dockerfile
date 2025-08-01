# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
# This is done first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# Add the project root to the PYTHONPATH to allow for absolute imports
# e.g., from src.models import hnet_amharic
ENV PYTHONPATH="/app"

# Specify the command to run on container startup.
# This can be overridden when running the container.
# For example, to run the training pipeline:
# docker run <image_name> python training_pipeline.py
CMD ["python", "main.py"]
