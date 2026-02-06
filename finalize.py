import os
import numpy as np
from PIL import Image
from hailo_sdk_client import ClientRunner

def run_finalize():
    runner = ClientRunner(hw_arch='hailo8')
    runner.load_har('clean_v2.har')

    # Load Calibration Data
    img_dir = 'calib_images'
    images = [np.array(Image.open(os.path.join(img_dir, f)).convert('RGB').resize((640, 640))) 
              for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))][:64]
    calib_dataset = np.array(images).astype(np.float32)

    # Simplified Model Script - No NMS commands to avoid conflicts
    model_script = "model_optimization_flavor(optimization_level=2)"
    runner.load_model_script(model_script)

    print("Starting optimization...")
    runner.optimize(calib_dataset)

    print("Compiling to HEF...")
    hef = runner.compile()
    with open('best.hef', 'wb') as f:
        f.write(hef)
    
    print("\nSUCCESS: best.hef created with 6 raw output tensors.")

if __name__ == "__main__":
    run_finalize()