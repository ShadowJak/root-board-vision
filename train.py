"""
Training script. Run this file.
"""

import sys
from ultralytics import YOLO
from pathlib import Path

# Check Python version
if sys.version_info < (3, 8):
    print("Error: Python 3.8 or higher is required.")
    print(f"You have Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    sys.exit(1)

def main():
    print("ROOT Board Vision - Training")
    print("=" * 50)

    # Check if photos and labels exist
    if not Path("train").exists() or not Path("train/images").exists():
        print("Error: 'train/images' folder not found.")
        print("Export your Roboflow dataset and extract it to this folder.")
        print("The structure should be:")
        print("  train/images/  - training images")
        print("  train/labels/  - training labels")
        print("  valid/images/  - validation images")
        print("  valid/labels/  - validation labels")
        sys.exit(1)

    if not Path("train/labels").exists():
        print("Error: 'train/labels' folder not found.")
        print("Export your Roboflow dataset and extract it to this folder.")
        sys.exit(1)    
        
    if not Path("valid").exists() or not Path("valid/images").exists():
        print("Error: 'valid/images' folder not found.")
        print("Export your Roboflow dataset and extract it to this folder.")
        sys.exit(1)

    if not Path("valid/labels").exists():
        print("Error: 'valid/labels' folder not found.")
        print("Export your Roboflow dataset and extract it to this folder.")
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

    # Create dataset config
    dataset_config = """
path: .
train: train/images
val: valid/images

nc: 9
names:
  0: Alliance Building
  1: Alliance Token
  2: Alliance Warrior
  3: Bird Building
  4: Bird Warrior
  5: Cat Building
  6: Cat Token
  7: Cat Warrior
  8: Clearing
"""

    with open("dataset.yaml", "w") as f:
        f.write(dataset_config)

    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")    
    print("\nStarting training...")
    print("This will take several hours. Training time depends on:")
    print("  - GPU: 2-4 hours (640x640 resolution)")
    print("  - CPU: 12-18 hours")
    print()
    results = model.train(
        data="dataset.yaml",
        epochs=100,
        imgsz=640,  # 640x640 - standard YOLO size, optimized for Hailo compilation
        patience=20,
        # device omitted so ultralytics auto-detects CUDA/CPU
    )

    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)

    # Export best model to ONNX
    print("\nExporting best model to ONNX format...")
    best_pt_path = Path(results.save_dir) / "weights" / "best.pt"
    best_model = YOLO(str(best_pt_path))
    best_model.export(format="onnx", simplify=True)    onnx_path = Path(results.save_dir) / "weights" / "best.onnx"
    print(f"\nModel exported to: {onnx_path}")
    print("\nNext steps:")
    print("1. Parse ONNX to HAR with Hailo DFC")
    print("2. Optimize with calibration data")
    print("3. Compile to HEF format")
    print("4. Copy the .hef file to your Raspberry Pi")
    print("\nSee README.md Step 4 for detailed compilation instructions.")


if __name__ == "__main__":
    main()
