#!/usr/bin/env python3
"""
Real‑time multi‑person pose classification with temporal models.
Supports:
  - GraphSAGE (with MPOSEFeatureExtractor) – expects window of raw keypoints (x,y,conf)
  - Temporal transformer (raw) – expects window of normalized keypoints (x,y,conf)
Maintains a sliding window per person and runs inference in a background thread.

python src/pipeline/temporal_live_pipeline.py --video 0 --video "PC-LM1E Camera" --model_path outputs/temporal/graphsage_temporal/best_model.pth --class_info data/mpose/class_info.json --temporal_model graphsage --window_size 20
"""

import argparse
import sys
import time
import numpy as np
import torch
import json
from pathlib import Path
import logging
import pygame
import imageio.v3 as iio
from PIL import Image
from typing import List, Dict, Optional, Tuple
from collections import deque
import os
import threading

# Add project root to path
script_path = Path(__file__).resolve()
current = script_path.parent
while current.name != 'src' and current != current.parent:
    current = current.parent
project_root = current.parent
sys.path.insert(0, str(project_root))

# Import core modules
from src.skeleton_extractor.yolo_wrapper_ultra import YOLOPersonDetector
from src.skeleton_extractor.multiperson_tracker import SimpleMPT
from src.skeleton_extractor.vitpose_wrapper import ViTPoseWrapper

# Import temporal model classes (adjust imports to your actual files)
# For GraphSAGE (with extractor):
try:
    from src.evaluation.temporal.graphsage_with_preprocessing import SpatioTemporalGraphSAGEWithExtractor
    from src.encoder.mpose_encoder import MPOSEFeatureExtractor
    GRAPHSAGE_AVAILABLE = True
except ImportError:
    GRAPHSAGE_AVAILABLE = False
    print("Warning: GraphSAGE model not available. Install required modules.")

# For transformer (raw):
try:
    from src.evaluation.temporal.transformer_raw import TemporalTransformerClassifier
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    print("Warning: Raw transformer model not available.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PyAV for camera capture
try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False
    logger.warning("PyAV not installed. Camera name capture may not work. Install with 'pip install av'")


# ----------------------------------------------------------------------
# Torso normalization (for raw transformer)
# ----------------------------------------------------------------------
def normalize_frame(keypoints, min_torso_height=10.0, max_norm_range=5.0):
    """
    Apply torso‑centric normalization to a single frame (J, 3).
    Returns normalized (x,y,conf) array of shape (J, 3).
    """
    left_shoulder = 5
    right_shoulder = 6
    left_hip = 11
    right_hip = 12

    pos = keypoints[:, :2]
    conf = keypoints[:, 2]

    # Torso normalization
    if (conf[left_shoulder] > 0 and conf[right_shoulder] > 0 and
        conf[left_hip] > 0 and conf[right_hip] > 0):
        shoulder_mid = (pos[left_shoulder] + pos[right_shoulder]) / 2
        hip_mid = (pos[left_hip] + pos[right_hip]) / 2
        torso_height = np.linalg.norm(shoulder_mid - hip_mid)
        if torso_height > min_torso_height:
            center = (shoulder_mid + hip_mid) / 2
            norm_pos = (pos - center) / torso_height
            norm_pos = np.clip(norm_pos, -max_norm_range, max_norm_range)
            return np.concatenate([norm_pos, conf[:, np.newaxis]], axis=1)

    # Fallback: bounding‑box normalization
    min_xy = np.min(pos, axis=0)
    max_xy = np.max(pos, axis=0)
    bbox_center = (min_xy + max_xy) / 2
    bbox_size = np.max(max_xy - min_xy)
    if bbox_size > 0:
        norm_pos = (pos - bbox_center) / bbox_size
        norm_pos = np.clip(norm_pos, -max_norm_range, max_norm_range)
        return np.concatenate([norm_pos, conf[:, np.newaxis]], axis=1)
    else:
        return keypoints


# ----------------------------------------------------------------------
# Temporal Model Wrapper
# ----------------------------------------------------------------------
class TemporalPoseClassifier:
    def __init__(self, model_path: str, class_names: List[str], device: str,
                 window_size: int, model_type: str = 'graphsage',
                 hidden_dim: int = 128, num_heads: int = 8, num_layers: int = 4):
        self.device = torch.device(device)
        self.class_names = class_names
        self.window_size = window_size
        self.model_type = model_type

        checkpoint = torch.load(model_path, map_location=device)

        if model_type == 'graphsage':
            if not GRAPHSAGE_AVAILABLE:
                raise ImportError("GraphSAGE model not available.")
            # Instantiate GraphSAGE model (same architecture as training)
            self.model = SpatioTemporalGraphSAGEWithExtractor(
                num_joints=17,
                joint_embedding_dim=64,
                graphsage_hidden_dims=[128, 256, 128],
                num_actions=len(class_names),
                dropout=0.3,
                temporal_window=window_size,
                num_attention_heads=4,
                skeleton_connections=None  # uses default COCO
            )
            self.model.load_state_dict(checkpoint)
            self.model.to(device).eval()
            self.extractor = None
            self.transformer = None
            logger.info("Loaded GraphSAGE model with extractor.")

        elif model_type == 'transformer':
            if not TRANSFORMER_AVAILABLE:
                raise ImportError("Raw transformer model not available.")
            # Instantiate raw transformer (no extractor)
            input_dim = 17 * 3  # 51
            self.model = TemporalTransformerClassifier(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=len(class_names),
                seq_len=window_size,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=0.3
            )
            self.model.load_state_dict(checkpoint)
            self.model.to(device).eval()
            self.extractor = None
            self.transformer = None
            logger.info("Loaded raw temporal transformer model.")

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def predict_sequence(self, keypoints_seq: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        keypoints_seq: (T, J, 3) raw keypoints (x,y,conf)
        Returns (class_idx, confidence, all_scores)
        """
        if self.model_type == 'graphsage':
            # GraphSAGE expects raw keypoints (no normalization; extractor does it)
            seq_tensor = torch.from_numpy(keypoints_seq).float().to(self.device).unsqueeze(0)  # (1, T, J, 3)
            with torch.no_grad():
                out = self.model(seq_tensor)
                logits = out['logits']
        else:  # transformer
            # Apply torso normalization to each frame
            norm_seq = []
            for frame in keypoints_seq:
                norm_frame = normalize_frame(frame)
                norm_seq.append(norm_frame.flatten())
            norm_seq = np.stack(norm_seq, axis=0)  # (T, 51)
            seq_tensor = torch.from_numpy(norm_seq).float().to(self.device).unsqueeze(0)  # (1, T, 51)
            with torch.no_grad():
                logits = self.model(seq_tensor)

        scores = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        class_idx = int(np.argmax(scores))
        confidence = float(scores[class_idx])
        return class_idx, confidence, scores


# ----------------------------------------------------------------------
# Drawing utilities (Pygame)
# ----------------------------------------------------------------------
class DrawingStyle:
    def __init__(self):
        self.bbox_color = (0, 255, 0)
        self.bbox_thickness = 2
        self.skeleton_colors = {'head': (255, 255, 0), 'upper': (255, 0, 0), 'lower': (0, 0, 255)}
        self.joint_radius = 4
        self.label_font = None
        self.label_bg = (0, 0, 0, 128)
        self.label_color = (255, 255, 255)
        self.label_font_size = 16

def draw_pygame_skeleton(screen, keypoints, style, offset=(0,0)):
    connections = [
        (0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,7),(7,9),
        (6,8),(8,10),(5,6),(5,11),(6,12),(11,12),(11,13),
        (13,15),(12,14),(14,16)
    ]
    for i, j in connections:
        if keypoints[i,2] > 0.3 and keypoints[j,2] > 0.3:
            pt1 = (int(keypoints[i,0]) + offset[0], int(keypoints[i,1]) + offset[1])
            pt2 = (int(keypoints[j,0]) + offset[0], int(keypoints[j,1]) + offset[1])
            if i < 5 or j < 5:
                color = style.skeleton_colors['head']
            elif i < 11 or j < 11:
                color = style.skeleton_colors['upper']
            else:
                color = style.skeleton_colors['lower']
            pygame.draw.line(screen, color, pt1, pt2, 2)

    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.3:
            pos = (int(x) + offset[0], int(y) + offset[1])
            if i < 5:
                color = style.skeleton_colors['head']
            elif i < 11:
                color = style.skeleton_colors['upper']
            else:
                color = style.skeleton_colors['lower']
            pygame.draw.circle(screen, color, pos, style.joint_radius)
            pygame.draw.circle(screen, (255,255,255), pos, style.joint_radius-1, 1)

def draw_pygame_label(screen, text, position, style):
    if style.label_font is None:
        style.label_font = pygame.font.Font(None, style.label_font_size)
    text_surf = style.label_font.render(text, True, style.label_color)
    if style.label_bg:
        bg_rect = text_surf.get_rect(topleft=position)
        bg_rect.inflate_ip(4, 2)
        pygame.draw.rect(screen, style.label_bg[:3], bg_rect)
        if len(style.label_bg) == 4:
            bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surf.fill(style.label_bg)
            screen.blit(bg_surf, bg_rect.topleft)
    screen.blit(text_surf, position)


# ----------------------------------------------------------------------
# Live Pipeline
# ----------------------------------------------------------------------
class LiveTemporalPipeline:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.conf_thresh = args.conf_thresh
        self.iou_thresh = args.iou_thresh
        self.max_lost = args.max_lost
        self.draw_skeleton = args.draw_skeleton
        self.draw_boxes = args.draw_boxes
        self.draw_labels = args.draw_labels
        self.skip_frames = args.skip_frames
        self.input_size = tuple(args.input_size) if args.input_size else None
        self.window_size = args.window_size
        self.temporal_model_type = args.temporal_model

        # Load models
        self.detector = YOLOPersonDetector(device=self.device)
        self.tracker = SimpleMPT(iou_threshold=self.iou_thresh, max_lost=self.max_lost)
        self.pose_estimator = ViTPoseWrapper(device=self.device)
        self.class_names = load_class_names(args.class_info)

        # Temporal classifier
        self.classifier = TemporalPoseClassifier(
            model_path=args.model_path,
            class_names=self.class_names,
            device=self.device,
            window_size=self.window_size,
            model_type=self.temporal_model_type,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers
        )

        # Per‑track buffers
        self.keypoint_buffers = {}   # track_id -> deque of keypoints (maxlen=window_size)
        self.last_results = {}       # track_id -> (label, confidence)

        # Threading
        self.running = True
        self.latest_frame = None
        self.latest_info = {}
        self.frame_ready = threading.Condition()
        self.lock = threading.Lock()
        self.frame_counter = 0
        self.processing = False

        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()

        self.fps_counter = deque(maxlen=30)
        self.writer = None
        if args.output:
            self.writer = iio.imopen(args.output, 'w', format='mp4')
            self.writer.init_video_writer(codec='h264', fps=30)

        self.drawing_style = DrawingStyle()

    def process_frame(self, frame):
        """Detect, track, pose, update buffers."""
        # 1. Detect
        if self.input_size:
            pil_frame = Image.fromarray(frame)
            pil_resized = pil_frame.resize(self.input_size, Image.Resampling.LANCZOS)
            frame_det = np.array(pil_resized)
        else:
            frame_det = frame

        _, boxes_xyxy_resized = self.detector.detect(frame_det, conf=self.conf_thresh)

        if self.input_size:
            h_orig, w_orig = frame.shape[:2]
            scale_x = w_orig / self.input_size[0]
            scale_y = h_orig / self.input_size[1]
            boxes_xyxy_orig = []
            for box in boxes_xyxy_resized:
                x1, y1, x2, y2 = box
                boxes_xyxy_orig.append([x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y])
        else:
            boxes_xyxy_orig = boxes_xyxy_resized

        # 2. Update tracker
        if boxes_xyxy_orig:
            tracks = self.tracker.update(boxes_xyxy_orig)
        else:
            self.tracker.update([])
            tracks = []

        # 3. Pose estimation
        if tracks:
            boxes_for_pose = [t.bbox.tolist() for t in tracks]
            try:
                pose_results = self.pose_estimator.infer(frame, boxes_for_pose, conf_threshold=0.3)
            except Exception as e:
                logger.error(f"Pose inference error: {e}")
                pose_results = [{'keypoints': np.zeros((17,3))} for _ in tracks]
        else:
            pose_results = []

        keypoints_dict = {}
        for i, res in enumerate(pose_results):
            track_id = tracks[i].track_id
            keypoints_dict[track_id] = res['keypoints']

        # 4. Update buffers
        with self.lock:
            for track_id, kp in keypoints_dict.items():
                if track_id not in self.keypoint_buffers:
                    self.keypoint_buffers[track_id] = deque(maxlen=self.window_size)
                self.keypoint_buffers[track_id].append(kp)
            # Remove old tracks
            current_ids = set(keypoints_dict.keys())
            for tid in list(self.keypoint_buffers.keys()):
                if tid not in current_ids:
                    del self.keypoint_buffers[tid]
                    if tid in self.last_results:
                        del self.last_results[tid]

        # Build info for drawing
        info = {
            'tracks': tracks,
            'keypoints': keypoints_dict,
            'labels': {tid: self.last_results.get(tid, ('unknown', 0.0))[0] for tid in keypoints_dict.keys()},
            'confidences': {tid: self.last_results.get(tid, ('unknown', 0.0))[1] for tid in keypoints_dict.keys()}
        }
        return frame, info

    def _inference_loop(self):
        """Background thread: processes full buffers."""
        while self.running:
            time.sleep(0.01)
            if not self.running:
                break
            with self.lock:
                buffers = dict(self.keypoint_buffers)
            for track_id, buffer in buffers.items():
                if len(buffer) == self.window_size:
                    seq = np.stack(list(buffer), axis=0)  # (T, J, 3)
                    try:
                        class_idx, conf, _ = self.classifier.predict_sequence(seq)
                        label = self.class_names[class_idx]
                        with self.lock:
                            self.last_results[track_id] = (label, conf)
                    except Exception as e:
                        logger.error(f"Classification error for track {track_id}: {e}")

    def draw_frame(self, frame, info):
        frame_surface = pygame.surfarray.make_surface(frame.transpose(1,0,2))
        tracks = info.get('tracks', [])
        keypoints_dict = info.get('keypoints', {})
        labels = info.get('labels', {})
        confidences = info.get('confidences', {})

        for track in tracks:
            bbox = track.bbox.astype(int)
            if self.draw_boxes:
                rect = pygame.Rect(bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1])
                pygame.draw.rect(frame_surface, self.drawing_style.bbox_color, rect, self.drawing_style.bbox_thickness)
                draw_pygame_label(frame_surface, f"ID:{track.track_id}",
                                  (bbox[0], bbox[1]-15), self.drawing_style)

            if track.track_id in keypoints_dict and self.draw_skeleton:
                draw_pygame_skeleton(frame_surface, keypoints_dict[track.track_id],
                                     self.drawing_style, offset=(0,0))

            if track.track_id in labels and self.draw_labels:
                label_text = f"{labels[track.track_id]} ({confidences[track.track_id]:.2f})"
                x = bbox[2] + 5
                y = bbox[1]
                draw_pygame_label(frame_surface, label_text, (x, y), self.drawing_style)
        return frame_surface

    def run_display(self, cap, using_pyav, fps, container=None):
        pygame.init()
        screen_info = pygame.display.Info()
        window_size = (min(screen_info.current_w, 1280), min(screen_info.current_h, 720))
        screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
        pygame.display.set_caption("Multi‑Person Pose Classification (Temporal)")
        clock = pygame.time.Clock()

        if self.writer:
            self.writer.init_video_writer(codec='h264', fps=fps)

        frame_count = 0

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.VIDEORESIZE:
                    window_size = event.size
                    screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)

            try:
                if using_pyav:
                    frame_av = next(cap)
                    frame = frame_av.to_ndarray(format='rgb24')
                else:
                    frame = cap.read()
                    if frame is None:
                        break
            except StopIteration:
                break
            except Exception as e:
                logger.error(f"Frame read error: {e}")
                break

            if frame.ndim == 2:
                frame = np.stack([frame]*3, axis=2)
            elif frame.shape[2] == 4:
                frame = frame[:,:,:3]

            processed, info = self.process_frame(frame)
            drawn = self.draw_frame(processed, info)

            h, w = processed.shape[:2]
            scale = min(window_size[0]/w, window_size[1]/h)
            new_w, new_h = int(w*scale), int(h*scale)
            scaled = pygame.transform.scale(drawn, (new_w, new_h))
            screen.fill((0,0,0))
            screen.blit(scaled, ((window_size[0]-new_w)//2, (window_size[1]-new_h)//2))

            self.fps_counter.append(clock.get_fps())
            if self.fps_counter:
                avg_fps = np.mean(self.fps_counter)
                draw_pygame_label(screen, f"FPS: {avg_fps:.1f}", (10,10), self.drawing_style)

            pygame.display.flip()
            clock.tick(fps)

            if self.writer:
                frame_out = pygame.surfarray.array3d(scaled)
                frame_out = frame_out.transpose(1,0,2)
                self.writer.write_frame(frame_out)

            frame_count += 1
            if self.args.max_frames and frame_count >= self.args.max_frames:
                break

        if using_pyav and container:
            container.close()
        else:
            cap.close()
        if self.writer:
            self.writer.close()
        pygame.quit()

    def run(self):
        if self.args.image:
            logger.error("Single image mode not supported.")
            return

        is_file = os.path.isfile(self.args.video) if self.args.video else False
        is_camera = not is_file

        using_pyav = False
        cap = None
        container = None
        fps = 30

        if is_camera and HAS_AV:
            try:
                video_input = f'video={self.args.video}' if not self.args.video.isdigit() else f'video={self.args.video}'
                container = av.open(video_input, format='dshow', options={'rtbufsize': '1000000000'})
                video_stream = next((s for s in container.streams if s.type == 'video'), None)
                if video_stream is None:
                    raise RuntimeError("No video stream found")
                cap = iter(container.decode(video=0))
                using_pyav = True
                logger.info(f"Opened camera with PyAV: {video_input}")
            except Exception as e:
                logger.warning(f"PyAV open failed: {e}, falling back to imageio")

        if not using_pyav:
            if is_camera:
                if self.args.video.isdigit():
                    cap = iio.imopen(f'<video{self.args.video}>', 'r')
                else:
                    raise ValueError(f"Imageio does not support camera names; use a numeric index or install PyAV. Got: {self.args.video}")
                fps = 30
            else:
                cap = iio.imopen(self.args.video, 'r')
                fps = cap.metadata.get('fps', 30) if hasattr(cap, 'metadata') else 30
                using_pyav = False

        self.run_display(cap, using_pyav, fps, container)


def load_class_names(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data.get('class_names', [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='0',
                        help='Video file, camera index, or camera name')
    parser.add_argument('--image', type=str, help='Single image file (overrides video)')
    parser.add_argument('--output', type=str, help='Output video file')
    parser.add_argument('--model_path', required=True, help='Path to best_model.pth')
    parser.add_argument('--class_info', required=True, help='Path to class_info.json')
    parser.add_argument('--temporal_model', type=str, default='graphsage', choices=['graphsage', 'transformer'])
    parser.add_argument('--window_size', type=int, default=30, help='Number of frames in temporal window')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--conf_thresh', type=float, default=0.3)
    parser.add_argument('--iou_thresh', type=float, default=0.3)
    parser.add_argument('--max_lost', type=int, default=5)
    parser.add_argument('--draw_skeleton', action='store_true', default=True)
    parser.add_argument('--draw_boxes', action='store_true', default=True)
    parser.add_argument('--draw_labels', action='store_true', default=True)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--max_frames', type=int)
    parser.add_argument('--skip_frames', type=int, default=20, help='Process every N frames (not used for temporal)')
    parser.add_argument('--input_size', type=int, nargs=2, default=[320,240])
    args = parser.parse_args()

    pipeline = LiveTemporalPipeline(args)
    pipeline.run()

if __name__ == '__main__':
    main()