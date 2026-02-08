"""
Training script for YOLOv5n. Run this file.
"""

import sys
import torch
import subprocess
from pathlib import Path

# Check Python version
if sys.version_info < (3, 8):
    print("Error: Python 3.8 or higher is required.")
    print(f"You have Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    sys.exit(1)

def main():
    print("ROOT Board Vision - YOLOv5n-128ch Training (Hailo AI HAT+)")
    print("=" * 60)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("ERROR: No GPU detected!")
        print("YOLOv5n training requires GPU.")
        sys.exit(1)

    # Path validation using absolute paths
    root = Path(__file__).parent.absolute()
    for folder in ['train/images', 'train/labels', 'valid/images', 'valid/labels']:
        if not (root / folder).exists():
            print(f"Error: {folder} not found at {root / folder}")
            print("Export your Roboflow dataset and extract it to this folder.")
            print("The structure should be:")
            print("  train/images/  - training images")
            print("  train/labels/  - training labels")
            print("  valid/images/  - validation images")
            print("  valid/labels/  - validation labels")
            sys.exit(1)

    # Clear cache files to ensure fresh dataset scanning
    cache_files = [
        Path("train/labels.cache"),
        Path("valid/labels.cache")
    ]
    for cache_file in cache_files:
        if cache_file.exists():
            cache_file.unlink()
            print(f"Removed old cache: {cache_file}")

    # Create dataset config with absolute paths
    import os
    project_root = root  # Use the absolute path from validation
    dataset_config = f"""path: {project_root}
train: train/images
val: valid/images

nc: 9
names: ['Alliance Building', 'Alliance Token', 'Alliance Warrior', 'Bird Building', 'Bird Warrior', 'Cat Building', 'Cat Token', 'Cat Warrior', 'Clearing']
"""

    with open("dataset.yaml", "w") as f:
        f.write(dataset_config)

    print("\nCloning YOLOv5 repository...")
    import os
    yolov5_dir = Path("yolov5")
    if not yolov5_dir.exists():
        os.system("git clone https://github.com/ultralytics/yolov5.git")
    
    print("\nLoading YOLOv5n-128ch model for Hailo AI HAT+...")
    print("\nStarting training with 128-channel architecture...")
    print("This will take 2-4 hours with GPU acceleration on RTX 4080.")
    print()
    
    # --- Dynamic resolution and folder naming ---
    img_size = 1280  # Change this to 1280 for high-res training
    folder_name = f"root_yolov5n_128ch_{img_size}_"
    
    # Train using YOLOv5 train.py with subprocess for proper error handling
    try:
        subprocess.run([
            "python", "yolov5/train.py",
            "--img", str(img_size),
            "--batch", "16",
            "--epochs", "100",
            "--data", str(root / "dataset.yaml"),
            "--cfg", str(root / "yolov5n_128ch.yaml"),  # Use 128-channel config for AI HAT+
            "--weights", "",  # Train from scratch with custom architecture
            "--cache",
            "--device", "0",
            "--project", "runs/train",
            "--name", folder_name
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Training failed with exit code {e.returncode}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)

    # Export best model to ONNX
    print("\nExporting best model to ONNX format...")
    
    # Dynamically find the most recent training folder matching the resolution
    train_folders = list(Path("runs/train").glob(f"root_yolov5n_128ch_*"))
    if not train_folders:
        print("ERROR: No training output found in runs/train/")
        sys.exit(1)
    # Sort by modification time, get most recent
    latest_folder = max(train_folders, key=lambda p: p.stat().st_mtime)
    best_pt_path = latest_folder / "weights" / "best.pt"
    
    if not best_pt_path.exists():
        print(f"ERROR: {best_pt_path} not found!")
        print(f"Training completed but weights file missing in {latest_folder}")
        sys.exit(1)
    
    print(f"Found trained model: {best_pt_path}")
    
    try:
        subprocess.run([
            "python", "yolov5/export.py",
            "--weights", str(best_pt_path),
            "--include", "onnx",
            "--simplify",
            "--opset", "11",  # Hailo AI HAT+ prefers opset 11
            "--imgsz", str(img_size), str(img_size)
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Export failed with exit code {e.returncode}")
        sys.exit(1)
    
    onnx_path = latest_folder / "weights" / "best.onnx"
    print(f"\nModel exported to: {onnx_path}")
    print("\nNext steps:")
    print("1. Parse ONNX to HAR with Hailo DFC")
    print("2. Optimize with calibration data")
    print("3. Compile to HEF format")
    print("4. Copy the .hef file to your Raspberry Pi")
    print("\nSee README.md Step 4 for detailed compilation instructions.")


if __name__ == "__main__":
    main()
