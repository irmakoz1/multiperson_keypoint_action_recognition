#!/usr/bin/env python3
"""
Real‑time multi‑person pose classification pipeline

Uses:
- Pygame for window and drawing (fast, customisable)
- ImageIO for video capture/writing
- YOLO for person detection
- SimpleMPT for tracking (Kalman + IOU)
- ViTPose for keypoints
-transformer

    python src/pipeline/live_pipeline.py
    --video "PC-LM1E Camera"
    --model_path outputs/transformer_20260323_034725/best_model.pth
    --class_info data/mpii/class_info.json


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
from PIL import Image, ImageDraw, ImageFont
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

from src.skeleton_extractor.yolo_wrapper_ultra import YOLOPersonDetector
from src.skeleton_extractor.multiperson_tracker import SimpleMPT
from src.skeleton_extractor.vitpose_wrapper import ViTPoseWrapper
from src.encoder.skeleton_encoder import JointFeatureExtractor
from src.evaluation.transformer_emg import TransformerClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PyAV for camera capture (optional, but preferred)
try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False
    logger.warning("PyAV not installed. Camera name capture may not work. Install with 'pip install av'")


class PoseClassifier:
    """Wraps extractor + transformer for inference."""
    def __init__(self, model_path: str, class_names: List[str], device: str, use_extractor: bool = True,
                 hidden_dim: int = 128, num_heads: int = 8, num_layers: int = 4, dropout: float = 0.3):
        self.device = torch.device(device)
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.use_extractor = use_extractor

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Build transformer
        self.transformer = TransformerClassifier(
            input_dim=64,  # extractor output dim
            hidden_dim=hidden_dim,
            num_classes=self.num_classes,
            seq_len=17,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            recon_weight=0.0,
            capture_attention=False
        ).to(self.device)

        if use_extractor:
            if isinstance(checkpoint, dict) and 'extractor' in checkpoint and 'transformer' in checkpoint:
                self.extractor = JointFeatureExtractor(
                    num_joints=17,
                    joint_type_embedding_dim=16,
                    max_angles=3,
                    use_angles=True,
                    use_velocities=False,
                    use_relative_pos=True,
                    use_confidence=True,
                    use_temporal=False,
                    output_dim=64,
                    normalize='torso',
                    augmentations={},
                    min_torso_height=10.0,
                    max_norm_range=5.0,
                    fallback_to_bbox=True,
                ).to(self.device)
                self.extractor.load_state_dict(checkpoint['extractor'])
                self.transformer.load_state_dict(checkpoint['transformer'])
                self.extractor.eval()
                self.transformer.eval()
                logger.info("Loaded extractor + transformer (on‑the‑fly mode).")
            else:
                raise ValueError("Checkpoint does not contain 'extractor' and 'transformer' keys.")
        else:
            self.transformer.load_state_dict(checkpoint)
            self.transformer.eval()
            self.extractor = None
            logger.info("Loaded transformer only (pre‑computed embeddings mode).")

    def predict(self, keypoints: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        keypoints: numpy array (17,3) with (x,y,confidence)
        Returns: (class_idx, confidence, all_scores)
        """
        if self.use_extractor:
            # Convert to tensor
            kp_tensor = torch.from_numpy(keypoints).float().to(self.device).unsqueeze(0)  # (1,17,3)
            with torch.no_grad():
                embeddings = self.extractor(kp_tensor)
                if isinstance(embeddings, dict):
                    embeddings = embeddings['joint_embeddings']
                logits = self.transformer(embeddings)
                scores = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        else:
            # Pre‑computed embeddings mode: expects embeddings directly
            raise NotImplementedError("Pre‑computed mode not implemented for live inference.")
        class_idx = int(np.argmax(scores))
        confidence = float(scores[class_idx])
        return class_idx, confidence, scores


class DrawingStyle:
    """Customisable drawing styles."""
    def __init__(self):
        self.bbox_color = (0, 255, 0)
        self.bbox_thickness = 2
        self.skeleton_colors = {
            'head': (255, 255, 0),   # yellow
            'upper': (255, 0, 0),    # red
            'lower': (0, 0, 255)     # blue
        }
        self.joint_radius = 4
        self.label_font = None
        self.label_bg = (0, 0, 0, 128)  # semi-transparent background
        self.label_color = (255, 255, 255)
        self.label_font_size = 16
        self.track_id_font_size = 12


def draw_pygame_skeleton(screen, keypoints, style, offset=(0,0)):
    """Draw skeleton using Pygame."""
    # Skeleton connections (COCO order)
    connections = [
        (0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,7),(7,9),
        (6,8),(8,10),(5,6),(5,11),(6,12),(11,12),(11,13),
        (13,15),(12,14),(14,16)
    ]
    for i, j in connections:
        if keypoints[i,2] > 0.3 and keypoints[j,2] > 0.3:
            pt1 = (int(keypoints[i,0]) + offset[0], int(keypoints[i,1]) + offset[1])
            pt2 = (int(keypoints[j,0]) + offset[0], int(keypoints[j,1]) + offset[1])
            # Choose colour based on joint index
            if i < 5 or j < 5:
                color = style.skeleton_colors['head']
            elif i < 11 or j < 11:
                color = style.skeleton_colors['upper']
            else:
                color = style.skeleton_colors['lower']
            pygame.draw.line(screen, color, pt1, pt2, 2)

    # Draw joints
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
    """Draw text with optional background."""
    if style.label_font is None:
        style.label_font = pygame.font.Font(None, style.label_font_size)
    text_surf = style.label_font.render(text, True, style.label_color)
    if style.label_bg:
        bg_rect = text_surf.get_rect(topleft=position)
        bg_rect.inflate_ip(4, 2)
        pygame.draw.rect(screen, style.label_bg[:3], bg_rect)
        if len(style.label_bg) == 4:
            # alpha not directly supported in Pygame rect, but we can use a surface
            bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surf.fill(style.label_bg)
            screen.blit(bg_surf, bg_rect.topleft)
    screen.blit(text_surf, position)


class LivePosePipeline:
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

        # Load models
        self.detector = YOLOPersonDetector(device=self.device)
        self.tracker = SimpleMPT(iou_threshold=self.iou_thresh, max_lost=self.max_lost)
        self.pose_estimator = ViTPoseWrapper(device=self.device)
        self.class_names = load_class_names(args.class_info)
        self.classifier = PoseClassifier(
            model_path=args.model_path,
            class_names=self.class_names,
            device=self.device,
            use_extractor=True,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
        self.drawing_style = DrawingStyle()

        # Threading state
        self.running = True
        self.latest_frame = None          # raw frame for inference
        self.latest_info = {}             # results from last inference
        self.frame_ready = threading.Condition()
        self.lock = threading.Lock()
        self.frame_counter = 0            # for frame skipping
        self.processing = False           # prevent overlapping inference

        # Start inference thread
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()

        # For FPS calculation
        self.fps_counter = deque(maxlen=30)
        self.last_time = time.time()

        # For video saving
        self.writer = None
        if args.output:
            self.writer = iio.imopen(args.output, 'w', format='mp4')
            self.writer.init_video_writer(codec='h264', fps=30)  # will update later

    def process_frame(self, frame):
        """
        Full processing (detection, tracking, pose, classification).
        Returns (processed_frame, info_dict).
        """
        # 1. Detect persons (maybe on resized frame)
        if self.input_size:
            pil_frame = Image.fromarray(frame)
            pil_resized = pil_frame.resize(self.input_size, Image.Resampling.LANCZOS)
            frame_det = np.array(pil_resized)
        else:
            frame_det = frame

        _, boxes_xyxy_resized = self.detector.detect(frame_det, conf=self.conf_thresh)

        # Scale boxes back to original coordinates if needed
        if self.input_size:
            h_orig, w_orig = frame.shape[:2]
            scale_x = w_orig / self.input_size[0]
            scale_y = h_orig / self.input_size[1]
            boxes_xyxy_orig = []
            for box in boxes_xyxy_resized:
                x1, y1, x2, y2 = box
                boxes_xyxy_orig.append([
                    x1 * scale_x, y1 * scale_y,
                    x2 * scale_x, y2 * scale_y
                ])
        else:
            boxes_xyxy_orig = boxes_xyxy_resized

        # 2. Update tracker
        if boxes_xyxy_orig:
            tracks = self.tracker.update(boxes_xyxy_orig)
        else:
            self.tracker.update([])
            tracks = []

        # 3. Pose estimation (only if we have boxes)
        if tracks:
            boxes_for_pose = [t.bbox.tolist() for t in tracks]
            try:
                pose_results = self.pose_estimator.infer(frame, boxes_for_pose, conf_threshold=0.3)
            except Exception as e:
                logger.error(f"Pose inference error: {e}")
                pose_results = [{'keypoints': np.zeros((17,3))} for _ in tracks]
        else:
            pose_results = []

        # Map track_id to keypoints
        keypoints_dict = {}
        for i, res in enumerate(pose_results):
            track_id = tracks[i].track_id
            keypoints_dict[track_id] = res['keypoints']

        # 4. Classification
        labels = {}
        confidences = {}
        for track_id, kp in keypoints_dict.items():
            try:
                class_idx, conf, _ = self.classifier.predict(kp)
                labels[track_id] = self.class_names[class_idx]
                confidences[track_id] = conf
            except Exception as e:
                logger.warning(f"Classification failed for track {track_id}: {e}")
                labels[track_id] = "unknown"
                confidences[track_id] = 0.0

        return frame, {'tracks': tracks, 'keypoints': keypoints_dict,
                       'labels': labels, 'confidences': confidences}

    def _inference_loop(self):
        """Background thread: waits for a new frame, processes it, updates results."""
        while self.running:
            with self.frame_ready:
                self.frame_ready.wait()           # wait for a new frame
                if not self.running:
                    break
                frame = self.latest_frame

            if frame is None or self.processing:
                continue

            self.processing = True
            try:
                _, info = self.process_frame(frame)
                with self.lock:
                    self.latest_info = info
            except Exception as e:
                logger.error(f"Inference thread error: {e}")
            finally:
                self.processing = False

    def draw_frame(self, frame, info):
        """Draw on frame using Pygame (modifies frame in place)."""
        frame_surface = pygame.surfarray.make_surface(frame.transpose(1,0,2))

        tracks = info.get('tracks', [])
        keypoints_dict = info.get('keypoints', {})
        labels = info.get('labels', {})
        confidences = info.get('confidences', {})
        print(f"Drawing: tracks={len(tracks)}, labels={len(labels)}")


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
    # Position at top-right of bounding box
                x = bbox[2] + 5
                y = bbox[1]
                draw_pygame_label(frame_surface, label_text, (x, y), self.drawing_style)
        return frame_surface

    def run_display(self, cap, using_pyav, fps, container=None):
        """Main display loop (runs in main thread)."""
        pygame.init()
        screen_info = pygame.display.Info()
        window_size = (min(screen_info.current_w, 1280), min(screen_info.current_h, 720))
        screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
        pygame.display.set_caption("Multi‑Person Pose Classification")
        clock = pygame.time.Clock()

        if self.writer:
            self.writer.init_video_writer(codec='h264', fps=fps)

        running = True
        frame_count = 0

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    window_size = event.size
                    screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)

            # Get frame
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

            # ---- Decide whether to trigger inference ----
            frame_count += 1
            if frame_count % self.skip_frames == 0:
                with self.lock:
                    self.latest_frame = frame.copy()
                with self.frame_ready:
                    self.frame_ready.notify()

            # ---- Get latest results ----
            with self.lock:
                info = self.latest_info.copy()   # avoid race while drawing

            # ---- Draw ----
            drawn = self.draw_frame(frame, info)

            # Scale to window
            h, w = frame.shape[:2]
            scale = min(window_size[0]/w, window_size[1]/h)
            new_w, new_h = int(w*scale), int(h*scale)
            scaled = pygame.transform.scale(drawn, (new_w, new_h))
            screen.fill((0,0,0))
            screen.blit(scaled, ((window_size[0]-new_w)//2, (window_size[1]-new_h)//2))

            # Show FPS
            self.fps_counter.append(clock.get_fps())
            if self.fps_counter:
                avg_fps = np.mean(self.fps_counter)
                draw_pygame_label(screen, f"FPS: {avg_fps:.1f}", (10,10), self.drawing_style)

            pygame.display.flip()
            clock.tick(fps)

            # Save frame if output
            if self.writer:
                frame_out = pygame.surfarray.array3d(scaled)
                frame_out = frame_out.transpose(1,0,2)
                self.writer.write_frame(frame_out)

            if self.args.max_frames and frame_count >= self.args.max_frames:
                break

        # Cleanup
        if using_pyav and container:
            container.close()
        else:
            cap.close()
        if self.writer:
            self.writer.close()
        pygame.quit()

    def run(self):
        """Entry point: set up capture and start display loop."""
        if self.args.image:
            # Single image mode (simple, no threading)
            frame = iio.imread(self.args.image)
            if frame.ndim == 2:
                frame = np.stack([frame]*3, axis=2)
            elif frame.shape[2] == 4:
                frame = frame[:,:,:3]
            processed, info = self.process_frame(frame)
            drawn = self.draw_frame(processed, info)
            self.display_single(drawn)
            return

        # --- Capture setup  ---
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

    def display_single(self, surface):
        """Show a single image in pygame window."""
        pygame.init()
        screen = pygame.display.set_mode(surface.get_size())
        pygame.display.set_caption("Result")
        screen.blit(surface, (0,0))
        pygame.display.flip()
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    waiting = False
        pygame.quit()


def load_class_names(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data.get('class_names', [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='0',
                        help='Video file, camera index (0, 1, ...), or camera name (e.g., "PC-LM1E Camera")')
    parser.add_argument('--image', type=str, help='Single image file (overrides video)')
    parser.add_argument('--output', type=str, help='Output video file (e.g., output.mp4)')
    parser.add_argument('--model_path', required=True, help='Path to best_model.pth')
    parser.add_argument('--class_info', required=True, help='Path to class_info.json')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--conf_thresh', type=float, default=0.3, help='Detection confidence')
    parser.add_argument('--iou_thresh', type=float, default=0.3, help='Tracker IOU threshold')
    parser.add_argument('--max_lost', type=int, default=5, help='Tracker max lost frames')
    parser.add_argument('--draw_skeleton', action='store_true', default=True)
    parser.add_argument('--draw_boxes', action='store_true', default=True)
    parser.add_argument('--draw_labels', action='store_true', default=True)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--max_frames', type=int, help='Max frames to process (for testing)')
    # Performance
    parser.add_argument('--skip_frames', type=int, default=15,
                        help='Process inference every N frames (1 = all frames). Higher = faster display.')
    parser.add_argument('--input_size', type=int, nargs=2, default=[640, 480],
                        help='Resize detection input to (width height).')

    args = parser.parse_args()
    pipeline = LivePosePipeline(args)
    pipeline.run()


if __name__ == '__main__':
    main()
