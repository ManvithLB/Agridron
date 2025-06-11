#!/bin/bash

# Variables
IMAGE_NAME="agridron-flask-app"
CONTAINER_NAME="agridron-flask-container"
PORT="5000"

# Build the Docker image
echo "Building Docker image..."
docker build -t $IMAGE_NAME .

# Stop and remove any existing container with the same name
echo "Stopping and removing existing container if it exists..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Run the Docker container
echo "Running Docker container..."
docker run -d --name $CONTAINER_NAME -p $PORT:5000 $IMAGE_NAME

# Print container status
echo "Checking container status..."
docker ps -f name=$CONTAINER_NAME

# Print the public IP or hostname to access the app
echo "Flask app should be running at http://$(curl -s http://169.254.169.254/latest/meta-data/public-hostname):$PORT"
~                                                                                                                         