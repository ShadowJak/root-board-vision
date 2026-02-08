#!/bin/bash
# Setup script for YOLOv5n training environment

set -e

echo "Setting up YOLOv5n training environment..."
echo "=========================================="

# 1. Install System Dependencies (Required for onnxsim and OpenCV)
# Updated libgl1-mesa-glx to libgl1 for Ubuntu 24.04 compatibility
echo "Checking for system dependencies (CMake, Build-Essential)..."
sudo apt update
sudo apt install -y cmake build-essential libgl1 libglib2.0-0

# 2. Check Python version
PYTHON_VERSION=$(python3.13 --version 2>&1 | grep -oP '\d+\.\d+' || echo "not found")
if [[ "$PYTHON_VERSION" != "3.13" ]]; then
    echo "Error: Python 3.13 is required"
    echo "Install it with: sudo apt install python3.13 python3.13-venv"
    exit 1
fi

# 3. Create virtual environment
echo "Creating virtual environment: venv_yolov5n"
python3.13 -m venv venv_yolov5n

# 4. Activate virtual environment
source venv_yolov5n/bin/activate

# 5. Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# 6. Install PyTorch with CUDA 11.8
echo "Installing PyTorch with CUDA 11.8 support..."
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# 7. Install other requirements
echo "Installing YOLOv5 dependencies..."
pip install -r requirements_yolov5n.txt

# 8. Test GPU detection
echo ""
echo "Testing GPU detection..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "=========================================="
echo "YOLOv5n environment setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  source venv_yolov5n/bin/activate"