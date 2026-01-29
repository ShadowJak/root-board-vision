"""
ROOT Board Vision - Rule Calculator
Post-processing script for rpicam-apps to calculate clearing control
"""

import json
import sys

# Class IDs (must match train.py)
CLASS_NAMES = {
    0: 'clearing',
    1: 'marquise_warrior',
    2: 'marquise_building',
    3: 'eyrie_warrior',
    4: 'eyrie_building',
    5: 'woodland_warrior',
    6: 'woodland_building',
    7: 'vagabond'
}

# Faction colors (BGR for OpenCV)
FACTION_COLORS = {
    'marquise': (0, 128, 255),    # Orange
    'eyrie': (255, 0, 0),          # Blue
    'woodland': (0, 255, 0),       # Green
    'none': (128, 128, 128)        # Gray
}


def bbox_overlap(box1, box2):
    """Calculate if a piece box overlaps with a clearing box"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Check if boxes overlap
    if x1_max < x2_min or x2_max < x1_min:
        return False
    if y1_max < y2_min or y2_max < y1_min:
        return False
    
    return True


def calculate_control(clearing_detections, piece_detections):
    """
    Calculate which faction controls each clearing
    
    Returns: dict mapping clearing_id -> {'faction': str, 'counts': dict}
    """
    control_results = {}
    
    for clearing_idx, clearing in enumerate(clearing_detections):
        clearing_box = clearing['bbox']  # [x_min, y_min, x_max, y_max]
        
        # Count pieces in this clearing
        faction_counts = {
            'marquise': 0,
            'eyrie': 0,
            'woodland': 0
        }
        
        for piece in piece_detections:
            piece_class = piece['class']
            piece_box = piece['bbox']
            
            # Skip vagabond (doesn't affect control)
            if piece_class == 7:
                continue
            
            # Check if piece is in this clearing
            if bbox_overlap(piece_box, clearing_box):
                class_name = CLASS_NAMES[piece_class]
                
                if 'marquise' in class_name:
                    faction_counts['marquise'] += 1
                elif 'eyrie' in class_name:
                    faction_counts['eyrie'] += 1
                elif 'woodland' in class_name:
                    faction_counts['woodland'] += 1
        
        # Determine control based on ROOT rules
        max_count = max(faction_counts.values())
        
        if max_count == 0:
            # Empty clearing - no one rules
            controlling_faction = 'none'
        else:
            # Find factions with max count
            factions_with_max = [f for f, c in faction_counts.items() if c == max_count]
            
            if len(factions_with_max) == 1:
                # Clear winner
                controlling_faction = factions_with_max[0]
            else:
                # Tie - check if Eyrie is involved
                if 'eyrie' in factions_with_max:
                    controlling_faction = 'eyrie'
                else:
                    # Tie without Eyrie - no one rules
                    controlling_faction = 'none'
        
        control_results[clearing_idx] = {
            'faction': controlling_faction,
            'counts': faction_counts,
            'bbox': clearing_box
        }
    
    return control_results


def process_detections(detections_json):
    """
    Main processing function called by rpicam-apps
    
    Input: JSON with detection results from YOLO
    Output: JSON with added control information
    """
    try:
        detections = json.loads(detections_json)
        
        # Separate clearings from pieces
        clearing_detections = []
        piece_detections = []
        
        for det in detections.get('detections', []):
            if det['class'] == 0:  # clearing
                clearing_detections.append(det)
            else:  # piece
                piece_detections.append(det)
        
        # Calculate control
        control_results = calculate_control(clearing_detections, piece_detections)
        
        # Add control info to output
        detections['control'] = control_results
        
        return json.dumps(detections)
    
    except Exception as e:
        print(f"Error in rule_calculator: {e}", file=sys.stderr)
        return detections_json


if __name__ == "__main__":
    # Read from stdin (rpicam-apps pipes detection data)
    input_data = sys.stdin.read()
    output_data = process_detections(input_data)
    print(output_data)
