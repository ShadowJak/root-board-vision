import os
import numpy as np
from PIL import Image
from hailo_sdk_client import ClientRunner

def run_finalize():
    runner = ClientRunner(hw_arch='hailo8')
    runner.load_har('best.har')

    # Load Calibration Data
    img_dir = 'calib_images'
    if not os.path.exists(img_dir):
        print(f"Error: Directory {img_dir} not found.")
        return

    images = [np.array(Image.open(os.path.join(img_dir, f)).convert('RGB').resize((640, 640))) 
              for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))][:64]
    
    if len(images) == 0:
        print("Error: No images found in calib_images folder.")
        return
        
    calib_dataset = np.array(images).astype(np.float32)

    # Use a single multi-line string (triple quotes), NO list, NO trailing period.
    model_script = """
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
model_optimization_flavor(optimization_level=4)
performance_param(compiler_optimization_level=max)
"""
    runner.load_model_script(model_script)

    print("Starting optimization...")
    runner.optimize(calib_dataset)

    print("Compiling to HEF...")
    hef = runner.compile()
    with open('best.hef', 'wb') as f:
        f.write(hef)
    
    print("\nSUCCESS: best.hef created.")

if __name__ == "__main__":
    run_finalize()