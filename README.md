# ROOT Board Vision

Real-time ROOT board game piece detection and clearing control visualization using YOLOv8 on Raspberry Pi 5 with Hailo-8 AI accelerator.

## What You Need

- Windows PC with GPU (for training - RTX 4080 Super: 4-6 hours, CPU: 24+ hours)
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

1. **Record video** of your ROOT board
   - **Use 16:9 aspect ratio** (1920x1080 or 1280x720) to match training resolution
   - Move pieces around during recording
   - Capture different board states
   - **Vary everything**: lighting, angles, camera height, board rotation
   - Record from different positions around the table
   - Record 5-10 minutes of footage (aim for 200-500 frames after extraction)

2. **Use Roboflow to extract frames and label**
   - Go to https://roboflow.com (free account)
   - Create new project → Upload your video
   - Roboflow will auto-extract frames (picks diverse ones, skips duplicates)
   - **Label at least 70+ images** (more is better)
   - Draw boxes around both clearings AND pieces
   - Use Roboflow's auto-labeler to speed up the process, then manually verify/correct
   
**Class names to use (9 classes):**
- `Alliance Building` - Green alliance bases
- `Alliance Token` - Green alliance sympathy tokens (counted but don't affect control)
- `Alliance Warrior` - Green alliance warriors
- `Bird Building` - Blue bird roosts
- `Bird Warrior` - Blue bird warriors
- `Cat Building` - Orange cat buildings (sawmill, workshop, recruiter)
- `Cat Token` - Orange cat keep tokens (counted but don't affect control)
- `Cat Warrior` - Orange cat warriors
- `Clearing` - Draw a box around each clearing space on the board

**Labeling tips:**
- Label all **visible** clearings in each frame
- Clearing boxes should encompass the entire clearing area
- Piece boxes should be tight around each piece
- Tokens are counted separately from warriors/buildings for control calculation

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

## Step 3: Train (4-6 hours with RTX 4080 Super)

Make sure your virtual environment is activated, then:

```powershell
python train.py
```

This will:
- Train YOLOv8m model for 100 epochs at 1280x720 resolution (16:9 aspect ratio)
- Automatically export the best model to ONNX format when complete
- Save results in `runs/detect/train/`

After training completes, you'll find:
```
runs/
└── detect/
    └── train/
        ├── weights/
        │   ├── best.pt          ← PyTorch model
        │   └── best.onnx        ← Export this to Pi
        └── results.png          ← Training metrics graph
```

## Step 4: Deploy to Raspberry Pi

**Control Rules (ROOT game logic):**
- Faction with the most pieces (warriors + buildings only) controls the clearing
- Tokens are counted but **don't affect control**
- Ties: Bird wins ties over other factions
- Empty clearings: Shown as gray

**Visualization:**
- Only clearing boxes are drawn (no individual piece boxes)
- Clearing box color indicates controlling faction:
  - Green = Alliance controlled
  - Blue = Bird controlled  
  - Orange = Cat controlled
  - Gray = No control
- Label shows piece counts: "Cat: 3, Alliance: 1"

**Prerequisites:**
- Raspberry Pi 5 with Raspberry Pi OS installed
- Camera Module 3 connected
- Hailo-8 AI HAT+ with drivers installed
  - Installation guide: https://www.raspberrypi.com/documentation/accessories/ai-kit.html

**Steps:**

1. **On your Windows PC** - Copy files to Pi:
   
   Replace `<pi-ip>` with your Pi's IP address or hostname (default user: `shadowjak`).
   
   ```powershell
   # Copy trained model
   scp runs\detect\train\weights\best.onnx shadowjak@<pi-ip>:/home/shadowjak/models/root_board_vision.onnx
   
   # Copy detection script
   scp root_detect.py shadowjak@<pi-ip>:/home/shadowjak/root-board-vision/
   ```
   
   If `scp` command not found, install OpenSSH Client:
   - Settings → Apps → Optional Features → Add a feature
   - Search "OpenSSH Client" → Install
   - Restart PowerShell

2. **On your Raspberry Pi** - Create models directory (if it doesn't exist):
   ```bash
   mkdir -p /home/shadowjak/models
   ```

3. **On your Raspberry Pi** - Convert ONNX to HEF (Hailo format):
   ```bash
   cd /home/shadowjak/models
   hailomz compile yolov8m root_board_vision.onnx --hw-arch hailo8 --output root_board_vision.hef
   ```
   This optimizes the model for Hailo-8 hardware.

4. **On your Raspberry Pi** - Install Python dependencies:
   ```bash
   sudo apt update
   sudo apt install python3-opencv python3-picamera2 python3-numpy
   pip3 install pyhailort
   ```

5. **On your Raspberry Pi** - Run detection:
   ```bash
   cd /home/shadowjak/root-board-vision
   python3 root_detect.py
   ```
   
   Press `q` to quit.

**Output:** Live camera feed showing clearing boxes colored by controlling faction with piece count labels.

## Summary

**Workflow:**
1. Record video of ROOT board (16:9 aspect ratio)
2. Upload to Roboflow, use auto-labeler, manually add clearing annotations
3. Export dataset in YOLO v8 format
4. Run `python train.py` (4-6 hours on RTX 4080 Super)
5. Copy `best.onnx` and `root_detect.py` to Raspberry Pi
6. Convert ONNX to HEF using Hailo compiler
7. Run `python3 root_detect.py` on Pi for live detection

**Files:**
- `train.py` - Training script (Windows)
- `root_detect.py` - Detection script with ROOT game logic (Raspberry Pi)
- `dataset.yaml` - Auto-generated during training
- `requirements.txt` - Python dependencies

**Model:**
- YOLOv8m at 1280x720 resolution
- 9 classes (3 factions × 3 piece types + clearings)
- Trained for 100 epochs with early stopping
