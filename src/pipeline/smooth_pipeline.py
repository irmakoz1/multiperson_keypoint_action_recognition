
#!/usr/bin/env python3
"""
Smooth live action recognition with a single camera capture thread.
- Capture thread: reads frames at full FPS, puts them into a queue.
- Display thread: consumes frames from the queue and draws the latest state.
- Processing thread: consumes frames at reduced rate, runs detection, pose, temporal model.
python src/pipeline/smooth_pipeline.py --video "PC-LM1E Camera" --model_path outputs/temporal/transformer_raw_mpose/best_model.pth --class_info data/mpose/class_info.json --temporal_model transformer --window_size 20 --det_input_size 480 360 --process_interval 3     --draw_skeleton --draw_boxes --draw_labels


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
import queue

# Add project root to path
script_path = Path(__file__).resolve()
current = script_path.parent
while current.name != 'src' and current != current.parent:
    current = current.parent
project_root = current.parent
sys.path.insert(0, str(project_root))

from src.skeleton_extractor.yolo_wrapper_ultra import YOLOPersonDetector
from src.skeleton_extractor.multiperson_tracker import SimpleMPT
from src.skeleton_extractor.vitpose_wrapper import ViTPoseWrapper

# Import temporal model classes (adjust imports)
try:
    from src.evaluation.temporal.graphsage_with_preprocessing import SpatioTemporalGraphSAGEWithExtractor
    GRAPHSAGE_AVAILABLE = True
except ImportError:
    GRAPHSAGE_AVAILABLE = False
try:
    from src.evaluation.temporal.transformer_raw import TemporalTransformerClassifier
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False

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
    """Torso normalization for a single frame (J, 3)."""
    left_shoulder = 5
    right_shoulder = 6
    left_hip = 11
    right_hip = 12

    pos = keypoints[:, :2]
    conf = keypoints[:, 2]

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

    # Fallback
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
# Temporal Model Wrapper (same as before)
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
            self.model = SpatioTemporalGraphSAGEWithExtractor(
                num_joints=17, joint_embedding_dim=64,
                graphsage_hidden_dims=[128, 256, 128],
                num_actions=len(class_names), dropout=0.3,
                temporal_window=window_size, num_attention_heads=4,
                skeleton_connections=None
            )
            self.model.load_state_dict(checkpoint)
            self.model.to(device).eval()
        elif model_type == 'transformer':
            if not TRANSFORMER_AVAILABLE:
                raise ImportError("Raw transformer model not available.")
            input_dim = 17 * 3
            self.model = TemporalTransformerClassifier(
                input_dim=input_dim, hidden_dim=hidden_dim,
                num_classes=len(class_names), seq_len=window_size,
                num_heads=num_heads, num_layers=num_layers, dropout=0.3
            )
            self.model.load_state_dict(checkpoint)
            self.model.to(device).eval()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def predict_sequence(self, keypoints_seq):
        if self.model_type == 'graphsage':
            seq_tensor = torch.from_numpy(keypoints_seq).float().to(self.device).unsqueeze(0)
            with torch.no_grad():
                out = self.model(seq_tensor)
                logits = out['logits']
        else:
            norm_seq = []
            for frame in keypoints_seq:
                norm_seq.append(normalize_frame(frame).flatten())
            seq_tensor = torch.from_numpy(np.stack(norm_seq, axis=0)).float().to(self.device).unsqueeze(0)
            with torch.no_grad():
                logits = self.model(seq_tensor)
        scores = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        class_idx = int(np.argmax(scores))
        return class_idx, scores[class_idx], scores


# ----------------------------------------------------------------------
# Drawing utilities (unchanged)
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
# Smooth Pipeline with Responsive Label Updates
# ----------------------------------------------------------------------
class SmoothLivePipeline:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.conf_thresh = args.conf_thresh
        self.iou_thresh = args.iou_thresh
        self.max_lost = args.max_lost
        self.draw_skeleton = args.draw_skeleton
        self.draw_boxes = args.draw_boxes
        self.draw_labels = args.draw_labels
        self.window_size = args.window_size
        self.temporal_model_type = args.temporal_model

        # Input size for detection (scaled down)
        self.det_input_size = tuple(args.det_input_size) if args.det_input_size else None
        # Interval for processing (frames)
        self.process_interval = args.process_interval

        # Models
        self.detector = YOLOPersonDetector(device=self.device)
        self.tracker = SimpleMPT(iou_threshold=self.iou_thresh, max_lost=self.max_lost,nms_threshold=0.5)
        self.pose_estimator = ViTPoseWrapper(device=self.device)
        self.class_names = load_class_names(args.class_info)
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

        # Shared state
        self.lock = threading.Lock()
        self.latest_tracks = []              # list of track objects
        self.latest_keypoints = {}           # track_id -> (17,3) numpy array
        self.latest_labels = {}              # track_id -> (label, confidence)
        self.buffers = {}                    # track_id -> deque of keypoints
        self.pred_buffers = {}               # track_id -> deque of (class_idx, confidence) for smoothing
        self.running = True

        # Queue for frames (maxsize to avoid memory blow)
        self.frame_queue = queue.Queue(maxsize=30)

        # Start threads
        self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.process_thread = threading.Thread(target=self._process_worker, daemon=True)
        self.display_thread = threading.Thread(target=self._display_worker, daemon=True)

        self.capture_thread.start()
        self.process_thread.start()
        self.display_thread.start()

        # For video saving
        self.writer = None
        if args.output:
            self.writer = iio.imopen(args.output, 'w', format='mp4')
            self.writer.init_video_writer(codec='h264', fps=30)

    def _capture_worker(self):
        """Capture frames from camera/video and put them into a queue."""
        # Determine capture method
        is_file = os.path.isfile(self.args.video) if self.args.video else False
        is_camera = not is_file

        using_pyav = False
        cap = None
        container = None
        fps = 30

        if is_camera and HAS_AV:
            try:
                video_input = f'video={self.args.video}' if not self.args.video.isdigit() else f'video={self.args.video}'
                container = av.open(video_input, format='dshow', options={'rtbufsize': '80000000'})
                video_stream = next((s for s in container.streams if s.type == 'video'), None)
                if video_stream is None:
                    raise RuntimeError("No video stream found")
                cap = iter(container.decode(video=0))
                using_pyav = True
                logger.info(f"Capture worker: opened camera with PyAV: {video_input}")
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

        frame_idx = 0
        while self.running:
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
                logger.error(f"Capture worker frame read error: {e}")
                break

            if frame.ndim == 2:
                frame = np.stack([frame]*3, axis=2)
            elif frame.shape[2] == 4:
                frame = frame[:,:,:3]

            # Put frame into queue (block if queue full)
            try:
                self.frame_queue.put(frame, timeout=1)
            except queue.Full:
                # Drop frame if queue is full
                pass

            frame_idx += 1

        # Cleanup
        if using_pyav and container:
            container.close()
        else:
            cap.close()
        logger.info("Capture worker stopped.")

    def _process_worker(self):
        """Consume frames from queue, process at reduced rate."""
        frame_counter = 0
        while self.running:
        # Get a frame (block until available)
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            frame_counter += 1
            if frame_counter % self.process_interval != 0:
                continue  # skip this frame

        # 1. Detection on downscaled frame
            if self.det_input_size:
                pil_frame = Image.fromarray(frame)
                pil_resized = pil_frame.resize(self.det_input_size, Image.Resampling.LANCZOS)
                frame_det = np.array(pil_resized)
            else:
                frame_det = frame

            _, boxes_xyxy_resized = self.detector.detect(frame_det, conf=self.conf_thresh)

        # Scale boxes back to original coordinates
            if self.det_input_size:
                h_orig, w_orig = frame.shape[:2]
                scale_x = w_orig / self.det_input_size[0]
                scale_y = h_orig / self.det_input_size[1]
                boxes_xyxy_orig = []
                for box in boxes_xyxy_resized:
                    x1, y1, x2, y2 = box
                    boxes_xyxy_orig.append([x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y])
            else:
                boxes_xyxy_orig = boxes_xyxy_resized

        # 2. Update tracker (now returns tracks and matched detection boxes)
            if boxes_xyxy_orig:
                tracks, matched_boxes = self.tracker.update(boxes_xyxy_orig)
            else:
                tracks, matched_boxes = self.tracker.update([])

        # 3. Pose estimation using raw detection boxes (or fallback to Kalman boxes)
            boxes_for_pose = []
            valid_tracks = []
            for i, track in enumerate(tracks):
            # Use raw detection if available, otherwise use Kalman‑filtered box
                if i < len(matched_boxes) and matched_boxes[i] is not None:
                    boxes_for_pose.append(matched_boxes[i])
                else:
                    boxes_for_pose.append(track.bbox.tolist())
                valid_tracks.append(track)

            if boxes_for_pose:
                try:
                    pose_results = self.pose_estimator.infer(frame, boxes_for_pose, conf_threshold=0.3)
                except Exception as e:
                    logger.error(f"Pose inference error: {e}")
                    pose_results = [{'keypoints': np.zeros((17,3))} for _ in boxes_for_pose]
            else:
                pose_results = []

            keypoints_dict = {}
            for i, res in enumerate(pose_results):
                track_id = valid_tracks[i].track_id
                keypoints_dict[track_id] = res['keypoints']

        # 4. Update buffers and run temporal model if buffer full
            with self.lock:
            # Update latest tracks and keypoints for drawing
                self.latest_tracks = tracks
                self.latest_keypoints = keypoints_dict

            # Update keypoint buffers
                for track_id, kp in keypoints_dict.items():
                    if track_id not in self.buffers:
                        self.buffers[track_id] = deque(maxlen=self.window_size)
                    self.buffers[track_id].append(kp)

            # Remove old tracks
                current_ids = set(keypoints_dict.keys())
                for tid in list(self.buffers.keys()):
                    if tid not in current_ids:
                        del self.buffers[tid]
                        if tid in self.latest_labels:
                            del self.latest_labels[tid]
                        if tid in self.pred_buffers:
                            del self.pred_buffers[tid]

            # Run temporal model on full buffers
                for track_id, buf in self.buffers.items():
                    if len(buf) == self.window_size:
                        seq = np.stack(list(buf), axis=0)  # (T, J, 3)
                        try:
                            class_idx, conf, _ = self.classifier.predict_sequence(seq)
                        # Update prediction buffer for smoothing
                            if track_id not in self.pred_buffers:
                                self.pred_buffers[track_id] = deque(maxlen=3)
                            self.pred_buffers[track_id].append((class_idx, conf))

                        # Compute smoothed label (weighted average confidence)
                            class_scores = {}
                            for c_idx, c_conf in self.pred_buffers[track_id]:
                                class_scores[c_idx] = class_scores.get(c_idx, 0) + c_conf
                            smoothed_idx = max(class_scores, key=class_scores.get)
                            avg_conf = class_scores[smoothed_idx] / len(self.pred_buffers[track_id])

                        # Immediate update if confidence > threshold, otherwise wait for smoothed change
                            current_label, _ = self.latest_labels.get(track_id, (None, 0.0))
                            if (conf > 0.8) or (smoothed_idx != current_label and avg_conf > 0.5):
                                label = self.class_names[smoothed_idx]
                                self.latest_labels[track_id] = (label, avg_conf)
                        except Exception as e:
                            logger.error(f"Temporal model error for track {track_id}: {e}")

        logger.info("Process worker stopped.")

    def _display_worker(self):
        """Consume frames from queue and draw the latest state."""
        pygame.init()
        screen_info = pygame.display.Info()
        window_size = (min(screen_info.current_w, 1280), min(screen_info.current_h, 720))
        screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
        pygame.display.set_caption("Smooth Action Recognition")
        clock = pygame.time.Clock()
        fps_counter = deque(maxlen=30)
        drawing_style = DrawingStyle()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.VIDEORESIZE:
                    window_size = event.size
                    screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)

            # Get a frame for display (non‑blocking)
            try:
                frame = self.frame_queue.get_nowait()
            except queue.Empty:
                # No new frame, just continue (maybe draw last frame)
                continue

            # Get latest state
            with self.lock:
                tracks = self.latest_tracks
                keypoints = self.latest_keypoints
                labels = self.latest_labels

            # Draw
            frame_surface = pygame.surfarray.make_surface(frame.transpose(1,0,2))
            for track in tracks:
                bbox = track.bbox.astype(int)
                if self.draw_boxes:
                    rect = pygame.Rect(bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1])
                    pygame.draw.rect(frame_surface, drawing_style.bbox_color, rect, drawing_style.bbox_thickness)
                    draw_pygame_label(frame_surface, f"ID:{track.track_id}", (bbox[0], bbox[1]-15), drawing_style)

                if track.track_id in keypoints and self.draw_skeleton:
                    draw_pygame_skeleton(frame_surface, keypoints[track.track_id], drawing_style, offset=(0,0))

                if track.track_id in labels and self.draw_labels:
                    label, conf = labels[track.track_id]
                    draw_pygame_label(frame_surface, f"{label} ({conf:.2f})", (bbox[2]+5, bbox[1]), drawing_style)

            # Scale to window
            h, w = frame.shape[:2]
            scale = min(window_size[0]/w, window_size[1]/h)
            new_w, new_h = int(w*scale), int(h*scale)
            scaled = pygame.transform.scale(frame_surface, (new_w, new_h))
            screen.fill((0,0,0))
            screen.blit(scaled, ((window_size[0]-new_w)//2, (window_size[1]-new_h)//2))

            # FPS
            fps_counter.append(clock.get_fps())
            if fps_counter:
                draw_pygame_label(screen, f"FPS: {np.mean(fps_counter):.1f}", (10,10), drawing_style)

            pygame.display.flip()
            clock.tick(30)  # limit to 30 fps for display

            if self.writer:
                frame_out = pygame.surfarray.array3d(scaled)
                frame_out = frame_out.transpose(1,0,2)
                self.writer.write_frame(frame_out)

        pygame.quit()
        if self.writer:
            self.writer.close()
        logger.info("Display worker stopped.")

    def run(self):
        """Wait for threads to finish (they run until self.running is False)."""
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Interrupted, shutting down...")
            self.running = False
        # Wait for threads to finish (give them time)
        self.capture_thread.join(timeout=2)
        self.process_thread.join(timeout=2)
        self.display_thread.join(timeout=2)

def load_class_names(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data.get('class_names', [])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='0')
    parser.add_argument('--image', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--class_info', required=True)
    parser.add_argument('--temporal_model', type=str, default='graphsage', choices=['graphsage', 'transformer'])
    parser.add_argument('--window_size', type=int, default=30)
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
    parser.add_argument('--det_input_size', type=int, nargs=2, default=[480, 360],
                        help='Input size for detection (width height)')
    parser.add_argument('--process_interval', type=int, default=3,
                        help='Process every N frames (1 = all frames)')
    args = parser.parse_args()

    if args.image:
        logger.error("Single image mode not supported.")
        return

    pipeline = SmoothLivePipeline(args)
    pipeline.run()

if __name__ == '__main__':
    main()