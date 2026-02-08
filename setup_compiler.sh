#!/bin/bash
# Setup script for Hailo Dataflow Compiler environment

set -e

echo "Setting up Hailo Dataflow Compiler environment..."
echo "=================================================="

# Check Python version
PYTHON_VERSION=$(python3.10 --version 2>&1 | grep -oP '\d+\.\d+' || echo "not found")
if [[ "$PYTHON_VERSION" != "3.10" ]]; then
    echo "Error: Python 3.10 is required"
    echo "Install it with: sudo apt install python3.10 python3.10-venv"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment: venv_compiler"
python3.10 -m venv venv_compiler

# Activate virtual environment
source venv_compiler/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Hailo Dataflow Compiler
echo "Installing Hailo Dataflow Compiler..."
pip install hailo_dataflow_compiler-3.33.0-py3-none-linux_x86_64.whl

# Install other requirements
echo "Installing compilation dependencies..."
pip install -r requirements_compiling.txt

echo ""
echo "=================================================="
echo "Hailo Compiler environment setup complete!"
echo "=================================================="
echo ""
echo "To activate the environment:"
echo "  source venv_compiler/bin/activate"
echo ""
echo "To compile a model:"
echo "  hailo parser onnx best.onnx --hw-arch hailo8l"
