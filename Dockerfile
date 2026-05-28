# OS Linux with Python 3.10
FROM python:3.10-slim

# Workdir
WORKDIR /app

# Install basic system tools
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first
COPY requirements.txt .

# Install all libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files to container
COPY . .

# Make entrypoint script executable
RUN chmod +x entrypoint.sh

# Tell Docker this app will use port 8501
EXPOSE 8501

# Run the entrypoint script
ENTRYPOINT ["./entrypoint.sh"]