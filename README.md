# ROOT Board Vision

## What Is Needed

- PC with GPU (for training - CPU will take 12+ hours)
- WSL2 with Ubuntu on Windows (or native Linux)
- Raspberry Pi 5 + Camera Module 3 + Hailo-8 AI Hat+ (26 TOPS)
- Python 3.13+ (for training)
- Python 3.10 (for Hailo compilation)

## Step 1: Install Dependencies (5 minutes)

Install Python 3.13 for training:

```bash
# Add deadsnakes PPA and install Python 3.13
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.13 python3.13-venv python3.13-dev

# Verify installation
python3.13 --version
```

Create a virtual environment and install packages:

```bash
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements_training.txt
```

**To reactivate the environment later:**
```bash
source venv/bin/activate
```

The `(venv)` prefix in the prompt indicates the environment is active.

## Step 2: Collect and Label Training Data (3-4 hours)

1. **Record video** of the ROOT board with a phone
   - Move pieces around during recording
   - Capture different board states
   - **Vary everything**: lighting, angles, camera height, board rotation
   - Record from different positions around the table
   - Record 5-10 minutes of footage (aim for 200-500 frames after extraction)

2. **Use Roboflow to extract frames and label**
   - Go to https://roboflow.com (free account)
   - Create new project → Upload the video
   - Roboflow will auto-extract frames (picks diverse ones, skips duplicates)
   - **Label at least 200 images** (more is better, aim for 300-500)
   - Draw boxes around both clearings AND pieces
   
**Class names to use:**
- `Clearing` - Draw a box around each clearing space on the board
- `Alliance Building` - Green alliance buildings (bases, sympathy tokens)
- `Alliance Token` - Green alliance sympathy tokens
- `Alliance Warrior` - Green alliance warriors
- `Bird Building` - Blue bird buildings (roosts)
- `Bird Warrior` - Blue bird warriors
- `Cat Building` - Orange cat buildings (sawmill, workshop, recruiter)
- `Cat Token` - Orange cat wood tokens
- `Cat Warrior` - Orange cat warriors

**Labeling tips:**
- Clearing boxes should encompass the entire clearing area
- Piece boxes should be tight around each piece

3. **Export annotations**
   - In Roboflow, go to "Generate" → Split the dataset (~80% train, ~20% valid)
   - Click "Export" → Select "YOLO v8" format → Download the zip file
   - Right-click the downloaded zip → "Extract All..." → Select the folder where `train.py` is → Click "Extract"
   - After extraction, the `train/` and `valid/` folders should appear next to `train.py`:
     ```
     train.py
     requirements.txt
     train/
       images/
       labels/
     valid/
       images/
       labels/
     ```

## Step 3: Train (2-4 hours with GPU, 12+ hours with CPU)

Ensure the virtual environment is activated, then:

```bash
python train.py
```

This will train for 100 epochs. When done, YOLO will create a `runs/` folder in the project folder. The trained model will be deeply nested (this is YOLO's default structure, not our choice):

```
project-folder/
├── train.py
├── requirements.txt
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── runs/
    └── detect/
        └── train/
            └── weights/
                ├── best.pt
                └── best.onnx  ← Copy this file
```

The path to copy is: `runs\detect\train\weights\best.onnx`

### Optional: Test Model with Webcam (Before Compiling)

Before compiling for Hailo, test the trained model on the PC using a webcam:

```bash
yolo predict model=runs/detect/train/weights/best.pt source=0 show=True conf=0.50
```

This will:
- Use the PC's default webcam (`source=0`)
- Show live detections in a window
- Only show detections with 50%+ confidence
- Press `q` to quit

If the default webcam doesn't work, try `source=1` or `source=2`.

## Step 4: Set Up Hailo Dataflow Compiler

Compile the model using the Hailo Dataflow Compiler (DFC) with Python 3.10.

### Install Compilation Environment

```bash
# Install Python 3.10 (Hailo DFC requires exactly 3.10)
# The deadsnakes PPA should already be configured from Step 1
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev

# Create virtual environment with Python 3.10
python3.10 -m venv .venv
source .venv/bin/activate

# Verify Python version (should show 3.10.x)
python --version

# Install compilation dependencies
pip install --upgrade pip
pip install -r requirements_compiling.txt
```

**Note:** This installation takes 5-10 minutes.

**To reactivate the compilation environment later:**
```bash
source .venv/bin/activate
```

### Verify installation
```bash
hailo --version
```
Expected output: Hailo DFC version (e.g., 3.33.0).

### Compilation Steps

**CRITICAL:** The 26 TOPS AI HAT+ has a full **Hailo-8** chip. Always use `--hw-arch hailo8` (NOT `hailo8l`). Using `hailo8l` will cause "Agent infeasible" compilation errors because it limits the compiler to a smaller resource pool.

#### Step 4a: Prepare Calibration Images

Before compiling, prepare calibration data for quantization:

1. **Collect calibration images** - Copy 50-100 diverse images from the training set:
   ```bash
   # Create calibration folder
   mkdir calib_images
   
   # Copy some training images (aim for 50-100 images with variety)
   cp train/images/* calib_images/
   ```

2. **Convert to numpy format** - Run the conversion script:
   ```bash
   python convert_calib.py
   ```
   
   This creates a `calib_npy/` folder with `.npy` files that Hailo uses for calibration.

#### Step 4b: Parse ONNX to HAR (2 minutes)

Convert the ONNX model to Hailo Archive (HAR) format:

```bash
source .venv/bin/activate
hailo parser onnx runs/detect/train/weights/best.onnx --hw-arch hailo8
```

This creates `best.har` in the project folder.

#### Step 4c: Optimize with Calibration Data (5-10 minutes)

Quantize the model using calibration images:

```bash
hailo optimize best.har --hw-arch hailo8 --calib-set-path calib_npy
```

This creates `best_optimized.har`.

#### Step 4d: Compile to HEF (15-30 minutes)

Compile the optimized model to Hailo Executable Format (HEF):

```bash
hailo compiler best_optimized.har --hw-arch hailo8 --output-dir ./hef_out
```

This creates `best.hef` in the `hef_out/` folder.

**Troubleshooting compilation errors:**

If "Agent infeasible" or "concat14 errors" occur, the model is too complex for single-pass compilation. Enable maximum optimization:

1. Verify `model_script.alls` exists with this content:
   ```
   performance_param(compiler_optimization_level=max)
   ```

2. Run the compiler with the optimization script:
   ```bash
   hailo compiler best_optimized.har --hw-arch hailo8 --model-script model_script.alls --output-dir ./hef_out
   ```
   
   **Note:** This can take 30-60 minutes or longer. Be patient!

#### Complete Pipeline (All Steps at Once)

To run all three compilation steps in sequence:

```bash
source .venv/bin/activate
hailo parser onnx runs/detect/train/weights/best.onnx --hw-arch hailo8 && hailo optimize best.har --hw-arch hailo8 --calib-set-path calib_npy && hailo compiler best_optimized.har --hw-arch hailo8 --output-dir ./hef_out
```

See [COMPILE_WITH_WSL2.md](COMPILE_WITH_WSL2.md) for additional troubleshooting and details.

## Step 5: Deploy to Raspberry Pi

**Control Rules:**
- Faction with the most total pieces (warriors + buildings) controls the clearing
- Ties: No one rules (except Eyrie wins ties)
- Empty clearings: No one rules
- Vagabond doesn't affect control

**Prerequisites:**
- Raspberry Pi 5 with Raspberry Pi OS installed
- Camera Module 3 connected

**Steps:**

1. **On the Raspberry Pi** - Install dependencies and create folder:
   ```bash
   sudo apt update
   sudo apt install -y hailo-all
   sudo reboot
   ```

2. **On the Linux PC or WSL** - Copy files from the project folder where `train.py` is located.
   
   Replace `<username>` with the Pi username and `<hostname>` with the Pi's IP address or hostname (e.g., `username@pi.local`).
   
   ```bash
   scp hef_out/best.hef <username>@<hostname>:~/models/root_board_vision.hef
   scp root_detect.py <username>@<hostname>:~/models/
   ```

3. **On the Raspberry Pi** - Verify files:
   ```bash
   cd ~/models
   ls -lh
   ```   Expected output: `root_board_vision.hef`, `root_detect.py`

4. **On the Raspberry Pi** - Run detection:
   ```bash
   cd ~/models
   python3 root_detect.py
   ```

**Output:** Live camera feed with bounding boxes around detected clearings and pieces showing which faction controls each clearing.

## Summary

Workflow:
1. Record video with phone
2. Label frames in Roboflow
3. Train model
4. Deploy to Raspberry Pi
