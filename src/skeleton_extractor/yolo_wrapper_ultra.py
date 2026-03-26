# src/skeleton_extractor/yolo_detector.py
from ultralytics import YOLO
import numpy as np
from typing import List, Tuple

class YOLOPersonDetector:
    """
    Person detector using Ultralytics YOLO.
    Returns boxes in multiple formats compatible with different trackers.
    Optimized for CPU inference with vectorized post-processing.
    """

    def __init__(self, model_path=r"src/models/yolo12n.pt", device="cpu"):
        self.model = YOLO(model_path)
        self.device = device

    def detect(self, image: np.ndarray, conf: float = 0.3) -> Tuple[List, List]:
        """
        Detect persons in an image.

        Args:
            image: RGB numpy image
            conf: confidence threshold

        Returns:
            tuple: (boxes_xywh, boxes_xyxy) where:
                - boxes_xywh: list of [x, y, w, h] for tracker
                - boxes_xyxy: list of [x1, y1, x2, y2] for ViTPose
        """
        # Run inference
        results = self.model.predict(
            source=image,
            conf=conf,
            device=self.device,
            verbose=False
        )[0]

        # If no detections, return empty lists
        if results.boxes is None:
            return [], []

        # Get all boxes and class labels as numpy arrays (already on CPU)
        boxes_xyxy = results.boxes.xyxy.numpy()  # shape: (N, 4)
        classes = results.boxes.cls.numpy().astype(int)  # shape: (N,)

        # Filter only person class (class 0)
        person_mask = classes == 0
        if not np.any(person_mask):
            return [], []

        person_boxes = boxes_xyxy[person_mask]  # shape: (M, 4)

        # Convert to list of lists for compatibility (original interface)
        boxes_xyxy_list = person_boxes.tolist()

        # Convert xyxy to xywh (x, y, w, h) using vectorized operations
        x1 = person_boxes[:, 0]
        y1 = person_boxes[:, 1]
        x2 = person_boxes[:, 2]
        y2 = person_boxes[:, 3]
        w = x2 - x1
        h = y2 - y1
        boxes_xywh = np.column_stack((x1, y1, w, h))  # shape: (M, 4)
        boxes_xywh_list = boxes_xywh.tolist()

        return boxes_xywh_list, boxes_xyxy_list