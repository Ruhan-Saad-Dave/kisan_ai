# Use Python 3.9 as the base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories if they don't exist
RUN mkdir -p model dataset

# Set environment variables
ENV PORT=7860

# Expose the port
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]