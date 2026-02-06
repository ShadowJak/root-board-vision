# ROOT Board Vision

## What You Need

- Windows PC with GPU (for training - CPU will take 12+ hours)
- Raspberry Pi 5 + Camera Module 3 + Hailo-8 AI Hat+ (26 TOPS)
- Python 3.8+

## Step 1: Install Dependencies (5 minutes)

Create a virtual environment and install packages:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Next time you work on this project**, activate the environment:
```powershell
venv\Scripts\Activate.ps1
```

The `(venv)` prefix in your prompt indicates the environment is active.

## Step 2: Collect and Label Training Data (3-4 hours)

1. **Record video** of your ROOT board with your phone
   - Move pieces around during recording
   - Capture different board states
   - **Vary everything**: lighting, angles, camera height, board rotation
   - Record from different positions around the table
   - Record 5-10 minutes of footage (aim for 200-500 frames after extraction)

2. **Use Roboflow to extract frames and label**
   - Go to https://roboflow.com (free account)
   - Create new project → Upload your video
   - Roboflow will auto-extract frames (picks diverse ones, skips duplicates)
   - **Label at least 200 images** (more is better, aim for 300-500)
   - Draw boxes around both clearings AND pieces
   
**Class names to use:**
- `clearing` - Draw a box around each clearing space on the board
- `marquise_warrior` - Orange cat warriors
- `marquise_building` - Orange cat buildings (sawmill, workshop, recruiter)
- `eyrie_warrior` - Blue bird warriors  
- `eyrie_building` - Blue bird buildings (roosts)
- `woodland_warrior` - Green alliance warriors
- `woodland_building` - Green alliance buildings (bases, sympathy tokens)
- `vagabond` - The vagabond pawn (doesn't affect control)

**Labeling tips:**
- Label all **visible** clearings in each frame (you won't see all 12 at once)
- Clearing boxes should encompass the entire clearing area
- Piece boxes should be tight around each piece
- This dual labeling allows the system to map pieces to clearings

3. **Export annotations**
   - In Roboflow, go to "Generate" → Split your dataset (70% train, 20% valid, 10% test)
   - Click "Export" → Select "YOLO v8" format → Download the zip file
   - Right-click the downloaded zip → "Extract All..." → Select the folder where `train.py` is → Click "Extract"
   - After extraction, you should see `train/` and `valid/` folders next to `train.py`:
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

Make sure your virtual environment is activated, then:

```powershell
python train.py
```

This will train for 100 epochs. When done, YOLO will create a `runs/` folder in your project folder. The trained model will be deeply nested (this is YOLO's default structure, not our choice):

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

Before compiling for Hailo, you can test your trained model on your PC using a webcam:

```powershell
yolo predict model=runs/detect/train/weights/best.pt source=0 show=True conf=0.50
```

This will:
- Use your PC's default webcam (`source=0`)
- Show live detections in a window
- Only show detections with 50%+ confidence
- Press `q` to quit

If your webcam is not the default, try `source=1` or `source=2`.

## Step 4: Set Up Hailo Dataflow Compiler (WSL2)

To deploy your model to the Hailo-8 AI HAT+, you need to compile it using the Hailo Dataflow Compiler (DFC). This runs in WSL2 (Windows Subsystem for Linux).

### Prerequisites

1. **Install WSL2 with Ubuntu**
   ```powershell
   wsl --install
   ```
   Restart your computer if prompted. Open a new PowerShell window and verify:
   ```powershell
   wsl --status
   ```

2. **Install Hailo Dataflow Compiler in WSL2**
   
   Open WSL2 terminal:
   ```powershell
   wsl
   ```
     Then inside WSL2, run:
   ```bash
   cd /c/Users/<your-username>/source/repos/root-board-vision
   
   # Install Python 3.10 (Hailo DFC requires exactly 3.10, not 3.11 or 3.12)
   sudo apt update
   sudo apt install -y software-properties-common
   sudo add-apt-repository ppa:deadsnakes/ppa -y
   sudo apt update
   sudo apt install -y python3.10 python3.10-venv python3.10-dev
   
   # Create virtual environment with Python 3.10
   python3.10 -m venv .venv
   source .venv/bin/activate
     # Verify Python version (should show 3.10.x)
   python --version
   
   # Install Hailo Dataflow Compiler from the downloaded .whl file
   # First, download hailo_dataflow_compiler-3.33.0-py3-none-linux_x86_64.whl from:
   # https://hailo.ai/developer-zone/software-downloads/
   # (Save it to your Downloads folder)
     pip install --upgrade pip
   pip install /c/Users/<your-username>/Downloads/hailo_dataflow_compiler-3.33.0-py3-none-linux_x86_64.whl
   ```
   
   **Note:** This installation takes 5-10 minutes as it extracts and installs all dependencies.

3. **Verify installation**
   ```bash
   hailo --version
   ```
   You should see the Hailo DFC version (e.g., 3.33.0).

### Compilation Steps

**CRITICAL:** Your 26 TOPS AI HAT+ has a full **Hailo-8** chip. Always use `--hw-arch hailo8` (NOT `hailo8l`). Using `hailo8l` will cause "Agent infeasible" compilation errors because it limits the compiler to a smaller resource pool.

#### Step 4a: Prepare Calibration Images

Before compiling, prepare calibration data for quantization:

1. **Collect calibration images** - Copy 50-100 diverse images from your training set:
   ```powershell
   # Create calibration folder
   mkdir calib_images
   
   # Copy some training images (aim for 50-100 images with variety)
   copy train\images\* calib_images\
   ```

2. **Convert to numpy format** - Run the conversion script:
   ```powershell
   python convert_calib.py
   ```
   
   This creates a `calib_npy/` folder with `.npy` files that Hailo uses for calibration.

#### Step 4b: Parse ONNX to HAR (2 minutes)

Convert your ONNX model to Hailo Archive (HAR) format:

```powershell
wsl bash -lc 'cd /c/Users/thesh/source/repos/root-board-vision && . .venv/bin/activate && hailo parser onnx runs/detect/train/weights/best.onnx --hw-arch hailo8'
```

This creates `best.har` in your project folder.

#### Step 4c: Optimize with Calibration Data (5-10 minutes)

Quantize the model using calibration images:

```powershell
wsl bash -lc 'cd /c/Users/thesh/source/repos/root-board-vision && . .venv/bin/activate && hailo optimize best.har --hw-arch hailo8 --calib-set-path calib_npy'
```

This creates `best_optimized.har`.

#### Step 4d: Compile to HEF (15-30 minutes)

Compile the optimized model to Hailo Executable Format (HEF):

```powershell
wsl bash -lc 'cd /c/Users/thesh/source/repos/root-board-vision && . .venv/bin/activate && hailo compiler best_optimized.har --hw-arch hailo8 --output-dir ./hef_out'
```

This creates `best.hef` in the `hef_out/` folder.

**If you get "Agent infeasible" or "concat14 errors":**

The model is too complex for single-pass compilation. Enable maximum optimization:

1. Verify `model_script.alls` exists with this content:
   ```
   performance_param(compiler_optimization_level=max)
   ```

2. Run the compiler with the optimization script:
   ```powershell
   wsl bash -lc 'cd /c/Users/thesh/source/repos/root-board-vision && . .venv/bin/activate && hailo compiler best_optimized.har --hw-arch hailo8 --model-script model_script.alls --output-dir ./hef_out'
   ```
   
   **Note:** This can take 30-60 minutes or longer. Be patient!

#### Complete Pipeline (All Steps at Once)

To run all three compilation steps in sequence:

```powershell
wsl bash -lc 'cd /c/Users/thesh/source/repos/root-board-vision && . .venv/bin/activate && hailo parser onnx runs/detect/train/weights/best.onnx --hw-arch hailo8 && hailo optimize best.har --hw-arch hailo8 --calib-set-path calib_npy && hailo compiler best_optimized.har --hw-arch hailo8 --output-dir ./hef_out'
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

1. **On your Raspberry Pi** - Install dependencies and create folder:
   ```bash
   sudo apt update
   sudo apt install -y python3-onnxruntime python3-opencv python3-picamera2
   mkdir -p ~/models
   ```

2. **On your Windows PC** - Copy files from the project folder where `train.py` is located.
   
   Replace `<username>` with your Pi username and `<hostname>` with your Pi's IP address or hostname (e.g., `username@pi.local`).
   
   ```powershell
   scp runs\detect\train\weights\best.onnx <username>@<hostname>:~/models/root_board_vision.onnx
   scp root_detect.py <username>@<hostname>:~/models/
   ```
   
   If `scp` command not found, install OpenSSH Client:
   - Settings → Apps → Optional Features → Add a feature
   - Search "OpenSSH Client" → Install   - Restart PowerShell

3. **On your Raspberry Pi** - Verify files:
   ```bash
   cd ~/models
   ls -lh
   ```   Expected output: `root_board_vision.onnx`, `root_detect.py`

4. **On your Raspberry Pi** - Run detection:
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
