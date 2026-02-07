#!/bin/bash
# Setup TensorFlow Object Detection API

set -e

echo "Setting up TensorFlow Object Detection API..."

# Clone TensorFlow models repo if not exists
MODELS_DIR="$HOME/tensorflow_models"
if [ ! -d "$MODELS_DIR" ]; then
    echo "Cloning TensorFlow models repository..."
    git clone https://github.com/tensorflow/models.git "$MODELS_DIR"
fi

# Compile protobufs
echo "Compiling protocol buffers..."
cd "$MODELS_DIR/research"
protoc object_detection/protos/*.proto --python_out=.

# Install Object Detection API
echo "Installing Object Detection API..."
cd "$MODELS_DIR/research"
pip install .

echo "Setup complete!"
