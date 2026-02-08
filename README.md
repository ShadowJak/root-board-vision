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
python3.13 -m venv venv_yolov5n
source venv_yolov5n/bin/activate
pip install --upgrade pip
pip install -r requirements_yolov5n.txt
```

**If you already created venv_yolov5n and got NumPy errors, delete it and recreate:**
```bash
rm -rf venv_yolov5n
python3.13 -m venv venv_yolov5n
source venv_yolov5n/bin/activate
pip install --upgrade pip
pip install -r requirements_yolov5n.txt
```

**To reactivate the environment later:**
```bash
source venv_yolov5n/bin/activate
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
- `Alliance Building` - Green alliance buildings (bases, sympathy tokens)
- `Alliance Token` - Green alliance sympathy tokens
- `Alliance Warrior` - Green alliance warriors
- `Bird Building` - Blue bird buildings (roosts)
- `Bird Warrior` - Blue bird warriors
- `Cat Building` - Orange cat buildings (sawmill, workshop, recruiter)
- `Cat Token` - Orange cat wood tokens
- `Cat Warrior` - Orange cat warrior
- `Clearing` - Draw a box around each clearing space on the board

**Labeling tips:**
- Clearing boxes should encompass the entire clearing area
- Piece boxes should be tight around each piece

3. **Export annotations**
   - In Roboflow, go to "Generate" → Split the dataset (~80% train, ~20% valid)
   - Click "Export" → Select "YOLOv5 PyTorch" format → Download the zip file
   - Right-click the downloaded zip → "Extract All..." → Select the folder where `yolov5n_train.py` is → Click "Extract"
   - After extraction, the `train/` and `valid/` folders should appear next to `yolov5n_train.py`:
     ```
     yolov5n_train.py
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
python yolov5n_train.py
```

This will train for 100 epochs. When done, YOLO will create a `runs/` folder in the project folder. The trained model will be deeply nested (this is YOLO's default structure, not our choice):

```
project-folder/
├── yolov5n_train.py
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

**Note:** Unless set up beforehand, WSL will not have access to the webcam.

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
python3.10 -m venv venv_compiler
source venv_compiler/bin/activate

# Verify Python version (should show 3.10.x)
python --version

# Install compilation dependencies
pip install --upgrade pip
pip install -r requirements_compiling.txt
```

**Note:** This installation takes 5-10 minutes.

**To reactivate the compilation environment later:**
```bash
source venv_compiler/bin/activate
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

```bash
mkdir -p calib_images && find train/images -type f | shuf -n 64 | xargs -I {} cp {} calib_images/
```

This randomly selects 64 images from your training set for calibration.

#### Step 4b: Optimize and Compile to HEF (15-60 minutes)

Run the optimization and compilation script:

```bash
source venv_compiler/bin/activate
python finalize.py
```

This script:
- Loads the HAR file (`best.har`)
- Optimizes the model using JPG images from `calib_images/`
- Compiles to HEF format
- Creates `best.hef` in the project folder

**Note:** This can take 15-60 minutes depending on model complexity. The script will show progress updates.

#### Complete Pipeline (All Steps at Once)

To run both compilation steps in sequence:

```bash
source venv_compiler/bin/activate
hailo parser onnx runs/detect/train/weights/best.onnx --hw-arch hailo8 --har-path best.har --end-node-names "/model.23/Concat" "/model.23/Sigmoid" && python finalize.py
```

## Step 5: Deploy to Raspberry Pi

**Control Rules:**
- Faction with the most total pieces (warriors + buildings) controls the clearing
- Ties: No one rules (except Eyrie wins ties)
- Empty clearings: No one rules
- Vagabond doesn't affect control

**Prerequisites:**
- Raspberry Pi 5 with Raspberry Pi OS installed
- Camera Module 3 connected
- AI HAT+ installed
- If Picamera2 and OpenCV are not included in hailo-all, install them with pip install picamera2 opencv-python.

**Steps:**

1. **On the Raspberry Pi** - Install dependencies and create folder:
   ```bash
   sudo apt update
   sudo apt install -y hailo-all
   sudo reboot
   ```

2. **On the Linux PC or WSL** - Copy files from the project folder where `yolov5n_train.py` is located.
   
   Replace `<username>` with the Pi username and `<hostname>` with the Pi's IP address or hostname (e.g., `username@pi.local`).
   
   ```bash
   scp best_<res>_<date stamp>.hef <username>@<hostname>:~/models/root_board_vision.hef
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
