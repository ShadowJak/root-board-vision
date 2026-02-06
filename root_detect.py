#!/usr/bin/env python3
"""
ROOT Board Vision - Live Detection
Run on Raspberry Pi with Hailo AI HAT+ to detect pieces and show clearing control
"""

import cv2
import numpy as np
import os
from picamera2 import Picamera2
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
                            InputVStreamParams, OutputVStreamParams, FormatType)
import time

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
    
    if x1_max < x2_min or x2_max < x1_min:
        return False
    if y1_max < y2_min or y2_max < y1_min:
        return False
    
    return True


def calculate_control(clearing_detections, piece_detections):
    """
    Calculate which faction controls each clearing
    
    Returns: list of dicts with faction and counts for each clearing
    """
    control_results = []
    
    for clearing in clearing_detections:
        clearing_box = clearing['bbox']
        
        # Count pieces in this clearing
        faction_counts = {
            'alliance': 0,
            'bird': 0,
            'cat': 0
        }
        
        for piece in piece_detections:
            piece_class = piece['class']
            piece_box = piece['bbox']
            
            # Skip tokens (class 1 = Alliance Token, class 6 = Cat Token)
            if piece_class == 1 or piece_class == 6:
                continue
            
            # Check if piece is in this clearing
            if bbox_overlap(piece_box, clearing_box):
                class_name = CLASS_NAMES[piece_class]
                
                if 'Alliance' in class_name:
                    faction_counts['alliance'] += 1
                elif 'Bird' in class_name:
                    faction_counts['bird'] += 1
                elif 'Cat' in class_name:
                    faction_counts['cat'] += 1
        
        # Determine control
        max_count = max(faction_counts.values())
        
        if max_count == 0:
            controlling_faction = 'none'
        else:
            factions_with_max = [f for f, c in faction_counts.items() if c == max_count]
            
            if len(factions_with_max) == 1:
                controlling_faction = factions_with_max[0]
            else:
                # Tie - Bird wins ties
                if 'bird' in factions_with_max:
                    controlling_faction = 'bird'
                else:
                    controlling_faction = 'none'
        
        control_results.append({
            'bbox': clearing_box,
            'faction': controlling_faction,
            'counts': faction_counts
        })
    
    return control_results


def draw_visualization(frame, clearing_detections, piece_detections):
    """Draw clearing boxes with control information"""
    
    # Calculate control
    control_results = calculate_control(clearing_detections, piece_detections)
    
    # Draw each clearing
    for control_info in control_results:
        bbox = control_info['bbox']
        faction = control_info['faction']
        counts = control_info['counts']
        
        # Get color
        color = FACTION_COLORS[faction]
        
        # Draw box
        x_min, y_min, x_max, y_max = map(int, bbox)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)
        
        # Create label
        sorted_factions = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        label_parts = [f"{f.capitalize()}: {c}" for f, c in sorted_factions if c > 0]
        
        if not label_parts:
            label = "Empty"
        else:
            label = ", ".join(label_parts)
        
        # Draw label background and text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        label_y = y_min - 10
        if label_y < text_height + 10:
            label_y = y_min + text_height + 10
        
        # Background
        cv2.rectangle(frame, 
                     (x_min, label_y - text_height - 5), 
                     (x_min + text_width + 10, label_y + 5), 
                     color, -1)
        
        # Text
        cv2.putText(frame, label, (x_min + 5, label_y), 
                   font, font_scale, (255, 255, 255), thickness)
    
    return frame


def preprocess_frame(frame, target_size=(640, 640)):
    """Preprocess frame for Hailo inference"""
    # Resize to model input size
    resized = cv2.resize(frame, target_size)
    
    # Hailo expects RGB uint8 format (0-255), no normalization needed
    # Already in RGB from Picamera2
    
    return resized


def postprocess_detections(raw_output, input_shape, frame_shape, conf_threshold=0.25, iou_threshold=0.45):
    """
    Convert ONNX raw output to detection format
    
    YOLOv8 ONNX output format: [batch, 4+num_classes, num_boxes]
    Transposed to: [batch, num_boxes, 4+num_classes]
    where each detection is [x_center, y_center, width, height, class_scores...]
    """
    detections = []
    
    # YOLOv8 output is [1, 13, 8400] - transpose to [1, 8400, 13]
    if len(raw_output.shape) == 3 and raw_output.shape[1] == 13:
        raw_output = np.transpose(raw_output, (0, 2, 1))
    
    batch_size, num_boxes, box_data_size = raw_output.shape
    
    # Calculate scale factors
    input_h, input_w = input_shape
    frame_h, frame_w = frame_shape[:2]
    scale_x = frame_w / input_w
    scale_y = frame_h / input_h
    
    for i in range(num_boxes):
        detection = raw_output[0, i, :]
        
        # Extract box coordinates (normalized 0-1)
        x_center, y_center, width, height = detection[0:4]
        
        # Extract class scores (no objectness in YOLOv8)
        class_scores = detection[4:]
        
        # Get best class
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]
        
        if confidence > conf_threshold:
            # Convert from normalized center coords to frame coords
            x_min = int((x_center - width / 2) * input_w * scale_x)
            y_min = int((y_center - height / 2) * input_h * scale_y)
            x_max = int((x_center + width / 2) * input_w * scale_x)
            y_max = int((y_center + height / 2) * input_h * scale_y)
            
            # Clip to frame bounds
            x_min = max(0, min(x_min, frame_w))
            y_min = max(0, min(y_min, frame_h))
            x_max = max(0, min(x_max, frame_w))
            y_max = max(0, min(y_max, frame_h))
            
            detections.append({
                'class': int(class_id),
                'bbox': [x_min, y_min, x_max, y_max],
                'confidence': float(confidence)
            })
    
    # Apply NMS (Non-Maximum Suppression) to remove duplicate detections
    detections = apply_nms(detections, iou_threshold)
    
    return detections


def apply_nms(detections, iou_threshold):
    """Apply Non-Maximum Suppression to remove overlapping boxes"""
    if len(detections) == 0:
        return []
    
    # Sort by confidence
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    while len(detections) > 0:
        # Keep highest confidence detection
        best = detections.pop(0)
        keep.append(best)
        
        # Remove detections that overlap significantly
        detections = [
            d for d in detections
            if not (d['class'] == best['class'] and 
                   calculate_iou(d['bbox'], best['bbox']) > iou_threshold)
        ]
    
    return keep


def calculate_iou(box1, box2):
    """Calculate Intersection over Union for two boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)
    
    if x_max < x_min or y_max < y_min:
        return 0.0
    
    intersection = (x_max - x_min) * (y_max - y_min)
    
    # Calculate union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def main():
    print("ROOT Board Vision - Starting...")
    print("=" * 50)
    
    # Configuration
    INFERENCE_SIZE = (640, 640)
    CAMERA_SIZE = (1280, 720)
    CONF_THRESHOLD = 0.25
    
    # Initialize camera
    print("Initializing camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": CAMERA_SIZE, "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)  # Camera warm-up
    
    # Initialize Hailo
    print("Loading Hailo model...")
    model_path = os.path.expanduser("~/models/root_board_vision.hef")
    
    # Load HEF file
    hef = HEF(model_path)
    
    # Get VDevice (the Hailo accelerator)
    devices = VDevice.scan()
    if not devices:
        raise RuntimeError("No Hailo device found! Make sure AI HAT+ is connected.")
    
    print(f"Found Hailo device: {devices[0]}")
    vdevice = VDevice(device_ids=devices)
    
    # Configure network
    network_group = hef.get_network_groups()[0]
    network_group_params = vdevice.create_configure_params(hef)
    
    # Configure input/output streams
    input_vstreams_params = InputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
    output_vstreams_params = OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
    
    print(f"Model input shape: {input_vstreams_params[0].shape}")
    print(f"Using inference size: {INFERENCE_SIZE}")
    print("Model loaded. Starting detection...")
    print("Press 'q' to quit")
    print()
    
    # FPS tracking
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0.0
    
    try:
        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            while True:
                # Capture frame
                frame = picam2.capture_array()
                
                # Preprocess frame
                input_data = preprocess_frame(frame, INFERENCE_SIZE)
                
                # Run inference on Hailo NPU
                inference_start = time.time()
                input_dict = {infer_pipeline.input_vstreams[0].name: np.expand_dims(input_data, axis=0).astype(np.float32)}
                output_dict = infer_pipeline.infer(input_dict)
                raw_output = list(output_dict.values())[0]
                inference_time = time.time() - inference_start
                
                # Postprocess to get detections
                detections = postprocess_detections(
                    raw_output, 
                    INFERENCE_SIZE, 
                    frame.shape, 
                    conf_threshold=CONF_THRESHOLD
                )
                
                # Separate clearings and pieces
                clearing_detections = [d for d in detections if d['class'] == 8]
                piece_detections = [d for d in detections if d['class'] != 8]
                
                # Draw visualization
                annotated_frame = draw_visualization(frame, clearing_detections, piece_detections)
                
                # Calculate FPS
                fps_frame_count += 1
                elapsed = time.time() - fps_start_time
                if elapsed > 1.0:
                    current_fps = fps_frame_count / elapsed
                    fps_frame_count = 0
                    fps_start_time = time.time()
                
                # Display stats
                cv2.putText(annotated_frame, f"FPS: {current_fps:.1f} | Inference: {inference_time*1000:.0f}ms | HAILO NPU", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Detections: {len(detections)}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display
                cv2.imshow("ROOT Board Vision", annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
