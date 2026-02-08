#!/usr/bin/env python3
"""
ROOT Board Vision - Live Detection
Optimized for Hailo-8 Hardware NMS and 1280x1280 Resolution
"""

import cv2
import numpy as np
import os
import time
from picamera2 import Picamera2
from hailo_platform import (HEF, VDevice, InferVStreams, InputVStreamParams, 
                            OutputVStreamParams, FormatType)

# Class IDs (must match train.py)
CLASS_NAMES = {
    0: 'Alliance Building',
    1: 'Alliance Token',
    2: 'Alliance Warrior',
    3: 'Bird Building',
    4: 'Bird Warrior',
    5: 'Cat Building',
    6: 'Cat Token',
    7: 'Cat Warrior',
    8: 'Clearing'
}

# Faction colors (BGR for OpenCV)
FACTION_COLORS = {
    'alliance': (0, 255, 0),       # Green
    'bird': (255, 0, 0),           # Blue
    'cat': (0, 128, 255),          # Orange
    'none': (128, 128, 128)        # Gray
}

def bbox_overlap(box1, box2):
    """Check if two bounding boxes overlap"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)

def calculate_control(clearing_detections, piece_detections):
    """Calculate which faction controls each clearing"""
    control_results = []
    for clearing in clearing_detections:
        clearing_box = clearing['bbox']
        faction_counts = {'alliance': 0, 'bird': 0, 'cat': 0}
        
        for piece in piece_detections:
            # Skip tokens (class 1 = Alliance Token, class 6 = Cat Token)
            if piece['class'] in [1, 6]:
                continue
            
            if bbox_overlap(piece['bbox'], clearing_box):
                class_name = CLASS_NAMES[piece['class']]
                if 'Alliance' in class_name:
                    faction_counts['alliance'] += 1
                elif 'Bird' in class_name:
                    faction_counts['bird'] += 1
                elif 'Cat' in class_name:
                    faction_counts['cat'] += 1
        
        max_count = max(faction_counts.values())
        if max_count == 0:
            controlling_faction = 'none'
        else:
            factions_with_max = [f for f, c in faction_counts.items() if c == max_count]
            # Tie - Bird wins ties
            if 'bird' in factions_with_max:
                controlling_faction = 'bird'
            elif len(factions_with_max) == 1:
                controlling_faction = factions_with_max[0]
            else:
                controlling_faction = 'none'
        
        control_results.append({
            'bbox': clearing_box,
            'faction': controlling_faction,
            'counts': faction_counts
        })
    return control_results

def postprocess_detections(raw_output, frame_shape, conf_threshold=0.25):
    """Parses Hardware NMS output: [ymin, xmin, ymax, xmax, confidence, class_id]"""
    detections = []
    frame_h, frame_w = frame_shape[:2]
    
    # Hailo NMS output is typically [1, 100, 6]
    for det in raw_output[0]:
        confidence = det[4]
        if confidence >= conf_threshold:
            ymin, xmin, ymax, xmax = det[0], det[1], det[2], det[3]
            detections.append({
                'class': int(det[5]),
                'bbox': [xmin * frame_w, ymin * frame_h, xmax * frame_w, ymax * frame_h],
                'confidence': float(confidence)
            })
    return detections

def main():
    # 1. Configuration - Matched to your 1280px HAR info
    INFERENCE_SIZE = (1280, 1280)
    CAMERA_SIZE = (1280, 720)
    CONF_THRESHOLD = 0.4 
    MODEL_PATH = os.path.expanduser("~/models/root_board_vision.hef")

    # 2. Initialize Hardware
    hef = HEF(MODEL_PATH)
    devices = VDevice.scan()
    if not devices:
        raise RuntimeError("No Hailo device found.")
    vdevice = VDevice(device_ids=devices)
    
    network_group = hef.get_network_groups()[0]
    
    # 3. Stream Params - UINT8 for maximum Raspberry Pi performance
    input_vstreams_params = InputVStreamParams.make_from_network_group(
        network_group, quantized=True, format_type=FormatType.UINT8)
    output_vstreams_params = OutputVStreamParams.make_from_network_group(
        network_group, quantized=True, format_type=FormatType.UINT8)

    # 4. Camera Setup
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": CAMERA_SIZE, "format": "RGB888"})
    picam2.configure(config)
    picam2.start()

    print(f"Starting ROOT Vision at {INFERENCE_SIZE[0]}px...")

    try:
        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            while True:
                start_time = time.time()
                frame = picam2.capture_array()
                
                # Resize to model input
                input_data = cv2.resize(frame, INFERENCE_SIZE)
                
                # Run Inference
                input_dict = {infer_pipeline.input_vstreams[0].name: np.expand_dims(input_data, axis=0)}
                output_dict = infer_pipeline.infer(input_dict)
                raw_output = list(output_dict.values())[0]
                
                # Postprocess (Hardware handles NMS)
                detections = postprocess_detections(raw_output, frame.shape, CONF_THRESHOLD)
                
                # Business Logic
                clearings = [d for d in detections if d['class'] == 8]
                pieces = [d for d in detections if d['class'] != 8]
                control_results = calculate_control(clearings, pieces)
                
                # Visualization
                for res in control_results:
                    bx = list(map(int, res['bbox']))
                    color = FACTION_COLORS[res['faction']]
                    cv2.rectangle(frame, (bx[0], bx[1]), (bx[2], bx[3]), color, 3)
                    
                    label = f"{res['faction'].upper()}"
                    cv2.putText(frame, label, (bx[0], bx[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Show frame (Convert RGB to BGR for OpenCV)
                cv2.imshow("ROOT Board Vision", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()