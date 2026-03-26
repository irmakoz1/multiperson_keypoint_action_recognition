# src/skeleton_extractor/vitpose_wrapper.py
from transformers import AutoProcessor, VitPoseForPoseEstimation
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ViTPoseWrapper:
    def __init__(self, model_name="usyd-community/vitpose-base-coco-aic-mpii", device="cpu"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading ViTPose model on {self.device}...")

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = VitPoseForPoseEstimation.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info("ViTPose model loaded successfully")

        # COCO skeleton connections (17 joints)
        self.skeleton_connections = [
            (0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,7),(7,9),
            (6,8),(8,10),(5,6),(5,11),(6,12),(11,12),(11,13),
            (13,15),(12,14),(14,16)
        ]

        self.joint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]

    def _heatmaps_to_keypoints_batch(self, heatmaps: torch.Tensor, boxes: List[List[float]],
                                      image_size: tuple) -> List[np.ndarray]:
        """
        Convert batch of heatmaps to keypoint coordinates using vectorized operations.

        Args:
            heatmaps: Tensor of shape (P, J, H, W) where P = number of persons, J = 17 joints
            boxes: List of P boxes in xyxy format [x1, y1, x2, y2] in original image coordinates
            image_size: (height, width) of original image (unused but kept for consistency)

        Returns:
            List of numpy arrays, each of shape (J, 3) with (x, y, confidence)
        """
        # ---- FIX: Ensure heatmaps is 4D ----
        if heatmaps.dim() != 4:
            if heatmaps.dim() == 5 and heatmaps.size(0) == 1:
                heatmaps = heatmaps.squeeze(0)
            else:
                raise ValueError(f"Expected 4D heatmaps, got {heatmaps.shape}")
        # ------------------------------------

        P, J, H, W = heatmaps.shape

        # Flatten spatial dimensions: (P, J, H*W)
        flat_heatmaps = heatmaps.view(P, J, -1)

        # Get max values and indices along spatial dimension
        max_vals, flat_indices = torch.max(flat_heatmaps, dim=2)

        # Convert flat indices to (h, w) coordinates
        h_idx = flat_indices // W  # (P, J)
        w_idx = flat_indices % W   # (P, J)

        # Normalize coordinates to [0, 1] range
        norm_x = w_idx.float() / (W - 1) if W > 1 else torch.zeros_like(w_idx.float())
        norm_y = h_idx.float() / (H - 1) if H > 1 else torch.zeros_like(h_idx.float())

        # Convert to CPU and numpy for coordinate calculation
        max_vals_np = max_vals.cpu().numpy()          # (P, J)
        norm_x_np = norm_x.cpu().numpy()              # (P, J)
        norm_y_np = norm_y.cpu().numpy()              # (P, J)

        # Precompute box dimensions
        boxes_np = np.array(boxes, dtype=np.float32)  # (P, 4)
        x1 = boxes_np[:, 0:1]   # (P, 1)
        y1 = boxes_np[:, 1:2]
        w_boxes = boxes_np[:, 2:3] - x1
        h_boxes = boxes_np[:, 3:4] - y1

        # Broadcast: (P, J) + (P, 1) -> (P, J)
        x_orig = x1 + norm_x_np * w_boxes
        y_orig = y1 + norm_y_np * h_boxes

        # Stack into (P, J, 3)
        keypoints_batch = np.stack([x_orig, y_orig, max_vals_np], axis=2)

        # Return as list of arrays (one per person)
        return [keypoints_batch[i] for i in range(P)]

    def infer(self, image: np.ndarray, boxes_xyxy: List[List[float]], conf_threshold: float = 0.3) -> List[Dict]:
        """
        Run pose estimation on an image.

        Args:
            image: RGB numpy array (H, W, 3)
            boxes_xyxy: List of bounding boxes [x1, y1, x2, y2]
            conf_threshold: Minimum confidence for keypoints (keypoints below this will be set to 0)

        Returns:
            List of dict with keys "keypoints" (np.array shape (17,3)) and "bbox"
        """
        if not boxes_xyxy:
            return []

        # Validate and clip boxes to image bounds
        h, w = image.shape[:2]
        valid_boxes = []
        for box in boxes_xyxy:
            if len(box) != 4:
                continue
            x1 = max(0, min(box[0], w-1))
            y1 = max(0, min(box[1], h-1))
            x2 = max(x1+1, min(box[2], w))
            y2 = max(y1+1, min(box[3], h))
            if x2 > x1 and y2 > y1:
                valid_boxes.append([x1, y1, x2, y2])

        if not valid_boxes:
            return []

        # Convert numpy array to PIL (required by processor)
        pil_img = Image.fromarray(image)

        try:
            # Prepare inputs: the processor expects a list of boxes (one per image)
            inputs = self.processor(
                images=pil_img,
                boxes=[valid_boxes],  # List of boxes for this single image
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            if not hasattr(outputs, 'heatmaps'):
                logger.warning("No heatmaps in model outputs")
                return []

            heatmaps = outputs.heatmaps

            # ---- FIX: Handle 5D heatmaps from model (batch, num_boxes, joints, H, W) ----
            # The model output shape can be (1, num_boxes, num_joints, H, W)
            if heatmaps.dim() == 5 and heatmaps.size(0) == 1:
                heatmaps = heatmaps[0]  # (num_boxes, num_joints, H, W)
            # ----------------------------------------------------------------------------

            # Now heatmaps should be 4D (num_boxes, num_joints, H, W)
            # Convert heatmaps to keypoints in one go
            keypoints_batch = self._heatmaps_to_keypoints_batch(heatmaps, valid_boxes, (h, w))

            # Apply confidence threshold and build results
            results = []
            for i, keypoints in enumerate(keypoints_batch):
                # keypoints shape: (17, 3)
                # Apply threshold: set low-confidence points to 0 (or we could set to 0,0,0)
                keypoints[keypoints[:, 2] < conf_threshold] = 0
                results.append({
                    "keypoints": keypoints,
                    "bbox": valid_boxes[i]
                })

            logger.info(f"Processed {len(results)} poses")
            return results

        except Exception as e:
            logger.error(f"Error in pose inference: {e}")
            import traceback
            traceback.print_exc()
            return []

    def draw_skeletons(self, image: np.ndarray, skeletons: List[Dict],
                       draw_boxes: bool = True, conf_threshold: float = 0.3) -> np.ndarray:
        """
        Draw skeletons on the image using PIL (no OpenCV).
        """
        # Convert numpy array to PIL Image (RGB)
        pil_img = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_img)

        # Colors (RGB) for different body parts
        colors = {
            "head": (255, 255, 0),      # Yellow
            "upper": (255, 0, 0),       # Red
            "lower": (0, 0, 255)        # Blue
        }

        if not skeletons:
            return np.array(pil_img)

        for person_idx, person in enumerate(skeletons):
            # Draw bounding box
            if draw_boxes and person.get("bbox") is not None:
                x1, y1, x2, y2 = person["bbox"]
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
                # Add person index text
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
                draw.text((x1, y1-10), f"P{person_idx}", fill=(0, 255, 0), font=font)

            keypoints = person["keypoints"]
            if isinstance(keypoints, (list, tuple)):
                keypoints = np.array(keypoints)

            # Draw bones (lines)
            for i, j in self.skeleton_connections:
                if (i < len(keypoints) and j < len(keypoints) and
                    keypoints[i, 2] > conf_threshold and
                    keypoints[j, 2] > conf_threshold):
                    x1, y1 = int(keypoints[i, 0]), int(keypoints[i, 1])
                    x2, y2 = int(keypoints[j, 0]), int(keypoints[j, 1])

                    # Choose color based on joint index
                    if i < 5 or j < 5:
                        color = colors["head"]
                    elif i < 11 or j < 11:
                        color = colors["upper"]
                    else:
                        color = colors["lower"]

                    draw.line([(x1, y1), (x2, y2)], fill=color, width=2)

            # Draw joints (circles)
            for i, (x, y, conf) in enumerate(keypoints):
                if conf > conf_threshold:
                    x, y = int(x), int(y)
                    if i < 5:
                        color = colors["head"]
                    elif i < 11:
                        color = colors["upper"]
                    else:
                        color = colors["lower"]
                    draw.ellipse([x-4, y-4, x+4, y+4], fill=color, outline=(255,255,255), width=1)

        # Convert back to numpy array (RGB)
        return np.array(pil_img)

    def get_keypoint_names(self):
        """Return list of keypoint names"""
        return self.joint_names