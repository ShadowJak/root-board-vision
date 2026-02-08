import os
import numpy as np
from PIL import Image
from hailo_sdk_client import ClientRunner

def run_finalize():
    # SETTINGS - Adjust these to match your setup
    onnx_path = 'runs/train/root_yolov5n_128ch_1280_/weights/best.onnx'
    har_path = 'best.har'
    hw_arch = 'hailo8l'  # Use 'hailo8l' for Raspberry Pi AI Kit
    
    # 1. Initialize Runner
    runner = ClientRunner(hw_arch=hw_arch)

    # 2. Generate HAR if missing
    if not os.path.exists(har_path):
        print(f"Translating {onnx_path} to HAR at 1280px...")
        runner.translate_onnx_model(
            onnx_path,
            'root_board_model',
            net_input_shapes={'images': [1, 3, 1280, 1280]},
            # Add these end node names to skip the unsupported Detect head
            end_node_names=[
                '/model.24/Sigmoid', 
                '/model.24/Sigmoid_1', 
                '/model.24/Sigmoid_2'
            ]
        )
        runner.save_har(har_path)
    else:
        print(f"Loading existing HAR: {har_path}")
        runner.load_har(har_path)

    # 3. Prepare Calibration Data
    img_dir = 'calib_images'
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))][:64]
    
    calib_dataset = np.zeros((len(image_files), 1280, 1280, 3), dtype=np.float32)

    print(f"Loading {len(image_files)} calibration images...")
    for i, f in enumerate(image_files):
        img = Image.open(os.path.join(img_dir, f)).convert('RGB')
        img = img.resize((1280, 1280), Image.BILINEAR)
        calib_dataset[i] = np.array(img).astype(np.float32)

    # 4. Model Script (ALLS)
    model_script = """
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
model_optimization_flavor(optimization_level=2)
performance_param(compiler_optimization_level=max)
nms_postprocess(meta_arch=yolov5, engine=cpu)
"""
    runner.load_model_script(model_script)

    # 5. Run Optimization
    print("Starting optimization (Level 4 - AdaRound)...")
    runner.optimize(calib_dataset)

    # 6. Compile and Save
    print("Compiling to HEF...")
    hef = runner.compile()
    
    output_hef = 'best_1280.hef'
    with open(output_hef, 'wb') as f:
        f.write(hef)
    
    print(f"\nSUCCESS: {output_hef} created for {hw_arch}.")

if __name__ == "__main__":
    run_finalize()