#!/usr/bin/env python3
"""
ROOT Board Vision - Live Detection
Run on Raspberry Pi to detect pieces and show clearing control
"""

import cv2
import numpy as np
from picamera2 import Picamera2
from pyhailort import HEFModel, ConfigureParams, VDevice, FormatType
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


def preprocess_frame(frame):
    """Preprocess frame for Hailo inference"""
    # Resize to model input size (should match training: 1280x720)
    resized = cv2.resize(frame, (1280, 720))
    
    # Normalize to 0-1 range (YOLO expects this)
    normalized = resized.astype(np.float32) / 255.0
    
    # Convert to NCHW format (batch, channels, height, width)
    input_data = np.transpose(normalized, (2, 0, 1))  # HWC -> CHW
    input_data = np.expand_dims(input_data, axis=0)   # Add batch dimension
    
    return input_data


def postprocess_detections(raw_output, conf_threshold=0.25, iou_threshold=0.45):
    """
    Convert Hailo raw output to detection format
    
    YOLO output format is typically [batch, num_detections, 5+num_classes]
    where each detection is [x_center, y_center, width, height, objectness, class_scores...]
    """
    detections = []
    
    # Extract detections from raw output
    # Note: Exact format depends on your YOLO model's output layer
    # This is a typical YOLO v8 format
    if len(raw_output.shape) == 3:
        batch_size, num_boxes, box_data_size = raw_output.shape
        
        for i in range(num_boxes):
            detection = raw_output[0, i, :]
            
            # Extract box coordinates
            x_center, y_center, width, height = detection[0:4]
            
            # Extract confidence and class scores
            objectness = detection[4]
            class_scores = detection[5:]
            
            # Get best class
            class_id = np.argmax(class_scores)
            class_conf = class_scores[class_id]
            
            # Combined confidence
            confidence = objectness * class_conf
            
            if confidence > conf_threshold:
                # Convert from center coords to corner coords
                x_min = int((x_center - width / 2) * 1280)
                y_min = int((y_center - height / 2) * 720)
                x_max = int((x_center + width / 2) * 1280)
                y_max = int((y_center + height / 2) * 720)
                
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
    
    # Initialize camera
    print("Initializing camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (1280, 720), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
      # Initialize Hailo
    print("Loading Hailo model...")
    model_path = "/home/shadowjak/models/root_board_vision.hef"
    
    try:
        with VDevice() as vdevice:
            # Load HEF model
            hef = HEFModel(model_path)
            
            # Configure network group
            configure_params = ConfigureParams.create_from_hef(hef, interface=FormatType.FLOAT32)
            network_group = vdevice.configure(hef, configure_params)[0]
            
            # Get input/output virtual streams
            network_group_params = network_group.create_params()
            input_vstreams_params = network_group_params.input_vstream_params
            output_vstreams_params = network_group_params.output_vstream_params
            
            print("Model loaded. Starting detection...")
            print("Press 'q' to quit")
            print()
            
            with network_group.activate(network_group_params):
                # Create input and output virtual streams
                input_vstream_info = hef.get_input_vstream_infos()[0]
                output_vstream_info = hef.get_output_vstream_infos()[0]
                
                while True:
                    # Capture frame
                    frame = picam2.capture_array()
                    
                    # Preprocess frame
                    input_data = preprocess_frame(frame)
                    
                    # Run inference
                    with network_group.activate(network_group_params):
                        # Send input
                        input_dict = {input_vstream_info.name: input_data}
                        
                        # Get output
                        output_dict = network_group.infer(input_dict)
                        
                        # Extract output tensor
                        raw_output = list(output_dict.values())[0]
                        
                        # Postprocess to get detections
                        detections = postprocess_detections(raw_output)
                    
                    # Separate clearings and pieces
                    clearing_detections = [d for d in detections if d['class'] == 8]
                    piece_detections = [d for d in detections if d['class'] != 8]
                    
                    # Draw visualization
                    annotated_frame = draw_visualization(frame, clearing_detections, piece_detections)
                    
                    # Display FPS
                    cv2.putText(annotated_frame, f"Detections: {len(detections)}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
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
