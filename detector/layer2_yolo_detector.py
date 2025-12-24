"""
Temporary mask detector using color detection
This is a workaround until you get a proper PPE YOLO model
"""

import cv2
import numpy as np
from ultralytics import YOLO

class YOLODetectorWithColorMask:
    def __init__(self, model_path="yolov8n.pt", classes=None):
        """
        Enhanced YOLO detector with color-based mask detection
        model_path: pre-trained YOLO model
        classes: list of class names to detect
        """
        self.model = YOLO(model_path)
        self.classes = classes

    def detect_mask_by_color(self, frame, person_bbox):
        """
        Detect if person is wearing a mask based on color in face region
        Returns True if mask detected
        """
        x1, y1, x2, y2 = person_bbox
        
        # Extract upper body region (likely contains face)
        height = y2 - y1
        face_y1 = y1
        face_y2 = y1 + int(height * 0.4)  # Upper 40% of person
        
        if face_y2 > y2:
            face_y2 = y2
            
        face_region = frame[face_y1:face_y2, x1:x2]
        
        if face_region.size == 0:
            return False
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for common mask colors
        # White masks
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        
        # Blue masks
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        # Create masks for each color
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(white_mask, blue_mask)
        
        # Calculate percentage of mask pixels
        mask_percentage = (cv2.countNonZero(combined_mask) / combined_mask.size) * 100
        
        # If more than 15% of face region is mask color, consider it a mask
        return mask_percentage > 15

    def detect(self, frame):
        """
        Detect objects in a single frame.
        Returns list of detections including color-based mask detection.
        """
        results = self.model(frame)
        detections = []

        for res in results:
            for box in res.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = self.model.names[cls_id]

                if self.classes and class_name not in self.classes:
                    continue

                detection = {
                    "bbox": [x1, y1, x2, y2],
                    "class": class_name,
                    "confidence": conf
                }
                
                detections.append(detection)
                
                # If it's a person, check for mask using color detection
                if class_name == "person":
                    has_mask = self.detect_mask_by_color(frame, [x1, y1, x2, y2])
                    if has_mask:
                        # Add a synthetic "mask" detection
                        mask_y1 = y1
                        mask_y2 = y1 + int((y2 - y1) * 0.3)
                        detections.append({
                            "bbox": [x1, mask_y1, x2, mask_y2],
                            "class": "mask",
                            "confidence": 0.7  # Lower confidence for color-based
                        })

        return detections