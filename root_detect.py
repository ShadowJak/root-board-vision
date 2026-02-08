#!/usr/bin/env python3
import cv2
import numpy as np
import os
from picamera2 import Picamera2
from hailo_platform import (HEF, VDevice, InferVStreams, InputVStreamParams, 
                            OutputVStreamParams, FormatType, ConfigureParams, 
                            HailoStreamInterface)

MODEL_PATH = os.path.expanduser("~/models/root_board_vision.hef")
INFERENCE_SIZE = (1280, 1280)
CAMERA_SIZE = (1280, 720)

CLASS_NAMES = {
    0: 'Alliance Building', 1: 'Alliance Token', 2: 'Alliance Warrior',
    3: 'Bird Building', 4: 'Bird Warrior', 5: 'Cat Building',
    6: 'Cat Token', 7: 'Cat Warrior', 8: 'Clearing'
}

def postprocess(output_list, frame_shape):
    detections = []
    h, w = frame_shape[:2]
    for class_id, class_detections in enumerate(output_list):
        if not class_detections:
            continue
        # Normalize: if single detection (1D array or list), wrap in list
        if isinstance(class_detections, (np.ndarray, list)) and (
            (isinstance(class_detections, np.ndarray) and class_detections.ndim == 1)
            or (isinstance(class_detections, list) and len(class_detections) > 0 and not isinstance(class_detections[0], (np.ndarray, list)))
        ):
            class_detections = [class_detections]
        for det in class_detections:
            det = np.array(det).flatten()
            if det.shape[0] < 5:
                continue
            conf = float(det[4])
            if conf < 0.4:
                continue
            detections.append({
                'class': class_id,
                'bbox': [
                    det[1] * w,
                    det[0] * h,
                    det[3] * w,
                    det[2] * h
                ],
                'conf': conf
            })
    return detections

def main():
    hef = HEF(MODEL_PATH)
    vdevice = VDevice()
    
    params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = vdevice.configure(hef, params)[0]
    
    in_params = InputVStreamParams.make_from_network_group(network_group, format_type=FormatType.UINT8)
    out_params = OutputVStreamParams.make_from_network_group(network_group, format_type=FormatType.FLOAT32)

    in_name = hef.get_input_vstream_infos()[0].name
    out_name = hef.get_output_vstream_infos()[0].name

    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": CAMERA_SIZE, "format": "BGR888"}))
    picam2.start()

    try:
        with network_group.activate():
            with InferVStreams(network_group, in_params, out_params) as pipeline:
                while True:
                    frame = picam2.capture_array()
                    
                    input_img = cv2.resize(frame, INFERENCE_SIZE)
                    input_tensor = np.expand_dims(input_img, axis=0).astype(np.uint8)
                    
                    results = pipeline.infer({in_name: input_tensor})
                    
                    # results[out_name] is the inhomogeneous list causing the previous errors
                    detections = postprocess(results[out_name], frame.shape)
                    
                    for d in detections:
                        b = [int(v) for v in d['bbox']]
                        label = f"{CLASS_NAMES.get(d['class'], 'Object')} {d['conf']:.2f}"
                        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                        cv2.putText(frame, label, (b[0], b[1]-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    cv2.imshow("Hailo ROOT Detect", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
