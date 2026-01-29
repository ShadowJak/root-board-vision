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

## Step 4: Deploy to Raspberry Pi

**Control Rules:**
- Faction with the most total pieces (warriors + buildings) controls the clearing
- Ties: No one rules (except Eyrie wins ties)
- Empty clearings: No one rules
- Vagabond doesn't affect control

**Prerequisites:**
- Raspberry Pi 5 with Raspberry Pi OS installed
- Camera Module 3 connected
- Hailo-8 AI HAT+ (26 TOPS) with AI Kit software installed (includes drivers, rpicam-apps, and Hailo compiler)
  - Installation guide: https://www.raspberrypi.com/documentation/accessories/ai-kit.html

**Steps:**

1. **On your Raspberry Pi** - Create project folder:
   ```bash
   mkdir -p ~/root-board-vision
   cd ~/root-board-vision
   ```

2. **On your Windows PC** - Copy files from the project folder where `train.py` is located.
   
   Replace `<username>` with your Pi username and `<hostname>` with your Pi's IP address or hostname.
   
   ```powershell
   scp runs\detect\train\weights\best.onnx <username>@<hostname>:~/root-board-vision/
   scp root_rule_calc.json <username>@<hostname>:~/root-board-vision/
   scp rule_calculator.py <username>@<hostname>:~/root-board-vision/
   ```
   
   If `scp` command not found, install OpenSSH Client:
   - Settings → Apps → Optional Features → Add a feature
   - Search "OpenSSH Client" → Install
   - Restart PowerShell

3. **On your Raspberry Pi** - Verify files:
   ```bash
   ls -lh
   ```
   Expected output: `best.onnx`, `root_rule_calc.json`, `rule_calculator.py`

4. **On your Raspberry Pi** - Convert ONNX to HEF:
   ```bash
   hailo parser onnx best.onnx
   hailo compiler best.har
   ```
   This creates `best.hef` optimized for Hailo-8 hardware.

5. **On your Raspberry Pi** - Organize files:
   ```bash
   mkdir -p models
   mv best.hef models/root_detector_h8.hef
   ```

6. **On your Raspberry Pi** - Run detection:
   ```bash
   rpicam-hello -t 0 --post-process-file root_rule_calc.json
   ```

**Output:** Live camera feed with bounding boxes around detected clearings and pieces.

## Summary

Workflow:
1. Record video with phone
2. Label frames in Roboflow
3. Train model
4. Deploy to Raspberry Pi
