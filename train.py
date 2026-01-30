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

# Create dataset config
dataset_config = """
path: .
train: train/images
val: valid/images

nc: 8
names:
  0: clearing
  1: marquise_warrior
  2: marquise_building
  3: eyrie_warrior
  4: eyrie_building
  5: woodland_warrior
  6: woodland_building
  7: vagabond
"""

with open("dataset.yaml", "w") as f:
    f.write(dataset_config)

print("Loading YOLO model...")
model = YOLO("yolov8m.pt")

print("\nStarting training...")
print("This will take several hours. Training time depends on:")
print("  - GPU: 4-6 hours (1280x720 resolution)")
print("  - CPU: 24+ hours")
print()

# Train
model.train(
    data="dataset.yaml",
    epochs=100,
    imgsz=(1280, 720),  # 16:9 aspect ratio - matches camera native resolution
    patience=20,
    # device omitted so ultralytics auto-detects CUDA/CPU
)

print("\n" + "=" * 50)
print("Training complete!")
print("=" * 50)

# Export best model to ONNX
print("\nExporting best model to ONNX format...")
best_model = YOLO("runs/detect/train/weights/best.pt")
best_model.export(format="onnx", simplify=True)

print("\nModel exported to: runs/detect/train/weights/best.onnx")
print("Next step: Copy best.onnx to your Raspberry Pi")
