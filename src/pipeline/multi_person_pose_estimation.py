# scripts/live_multiperson_pose.py
import torch
import numpy as np
import cv2
from pathlib import Path
import json
import yaml
import sys
import os
from collections import defaultdict
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pose_classification.graphsage import GraphSAGEPoseClassifier
from src.skeleton_extractor.skeleton_extractor import SkeletonExtractor

class LiveMultiPersonPoseClassifier:
    def __init__(self, model_dir, device='cpu'):
        print("📦 Initializing Live Multi-Person Pose Classifier...")

        self.device = torch.device(device)
        self.model_dir = Path(model_dir)

        # Load model and class info
        self.model, self.idx_to_pose, self.config = self.load_model()

        # Initialize skeleton extractor (handles multi-person)
        self.skeleton_extractor = SkeletonExtractor(device=device)

        # For tracking people across frames
        self.person_tracks = {}
        self.next_id = 0

        # Performance metrics
        self.fps = 0
        self.frame_times = []

        print(" Ready for live multi-person pose classification!")

    def load_model(self):
        """Load trained model and class info."""
        with open(self.model_dir / 'class_info.json', 'r') as f:
            class_info = json.load(f)

        with open(self.model_dir / 'config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        checkpoint = torch.load(self.model_dir / 'best_model.pth', map_location=self.device)
        num_classes = checkpoint['model_state_dict']['pose_classifier.3.weight'].shape[0]

        model = GraphSAGEPoseClassifier(
            num_joints=17,
            in_features=3,
            joint_embedding_dim=config['joint_embedding_dim'],
            graphsage_hidden_dims=config['graphsage_hidden_dims'],
            num_pose_classes=num_classes
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        # Create pose mapping
        pose_categories = sorted(set(class_info['class_names'].values()))
        idx_to_pose = {i: pose for i, pose in enumerate(pose_categories)}

        print(f"  - Loaded model with {num_classes} pose classes")
        print(f"  - Pose categories: {list(idx_to_pose.values())}")

        return model, idx_to_pose, config

    def predict_person(self, keypoints):
        """Predict pose for a single person."""
        # Convert to tensor
        keypoints_tensor = torch.from_numpy(keypoints).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(keypoints_tensor)
            probs = outputs['pose_probs'][0].cpu().numpy()
            pred_idx = probs.argmax()
            confidence = probs[pred_idx]

            # Get top 3
            top3_idx = probs.argsort()[-3:][::-1]
            top3 = [(self.idx_to_pose[idx], probs[idx]) for idx in top3_idx]

        return {
            'pose': self.idx_to_pose[pred_idx],
            'pose_idx': int(pred_idx),
            'confidence': float(confidence),
            'top3': top3,
            'joint_embeddings': outputs['joint_embeddings'][0].cpu().numpy(),
            'all_probs': {self.idx_to_pose[i]: float(probs[i]) for i in range(len(probs))}
        }

    def process_frame(self, frame):
        """Process a single frame with multiple people."""
        start_time = time.time()

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Extract skeletons for all people
        results = self.skeleton_extractor.process_frame(frame_rgb)

        people_data = []

        # Process each person
        for i, skeleton in enumerate(results['skeletons']):
            keypoints = skeleton['keypoints']  # (17, 3)
            bbox = skeleton.get('bbox', None)
            track_id = skeleton.get('track_id', i)

            # Predict pose
            prediction = self.predict_person(keypoints)

            people_data.append({
                'track_id': track_id,
                'bbox': bbox,
                'keypoints': keypoints,
                'prediction': prediction
            })

        # Calculate FPS
        self.frame_times.append(time.time() - start_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        self.fps = len(self.frame_times) / sum(self.frame_times)

        return people_data

    def draw_predictions(self, frame, people_data):
        """Draw bounding boxes and predictions on frame."""
        for person in people_data:
            bbox = person['bbox']
            prediction = person['prediction']

            if bbox:
                x1, y1, x2, y2 = [int(v) for v in bbox]

                # Draw bounding box
                color = (0, 255, 0)  # Green
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw pose text
                text = f"ID: {person['track_id']}"
                text += f" | {prediction['pose']}: {prediction['confidence']:.0%}"

                # Text background
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1-25), (x1+w, y1), color, -1)

                # Text
                cv2.putText(frame, text, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Draw FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"People: {len(people_data)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

    def run_on_webcam(self, camera_id=0):
        """Run live on webcam."""
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print("❌ Could not open webcam")
            return

        print("\n🎥 Live webcam started. Press 'q' to quit, 's' to save frame")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            people_data = self.process_frame(frame)

            # Draw results
            frame = self.draw_predictions(frame, people_data)

            # Show frame
            cv2.imshow('Multi-Person Pose Classification', frame)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = int(time.time())
                cv2.imwrite(f'frame_{timestamp}.jpg', frame)
                print(f"💾 Saved frame_{timestamp}.jpg")

        cap.release()
        cv2.destroyAllWindows()

    def run_on_video(self, video_path, output_path=None):
        """Run on video file."""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0

        print(f"\n🎬 Processing video: {video_path}")
        print(f"  - Resolution: {width}x{height}")
        print(f"  - FPS: {fps}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Process frame
            people_data = self.process_frame(frame)

            # Draw results
            frame = self.draw_predictions(frame, people_data)

            # Write frame
            if writer:
                writer.write(frame)

            # Show progress
            if frame_count % 30 == 0:
                print(f"  Processed {frame_count} frames, {len(people_data)} people in current frame")

        cap.release()
        if writer:
            writer.release()

        print(f"\n✅ Video processing complete! {frame_count} frames processed")
        if output_path:
            print(f"   Output saved to: {output_path}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Live Multi-Person Pose Classification')
    parser.add_argument('--mode', type=str, choices=['webcam', 'video'], default='webcam',
                       help='Mode: webcam or video')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source: camera ID (e.g., 0) or video path')
    parser.add_argument('--output', type=str, help='Output video path (for video mode)')
    parser.add_argument('--model_dir', type=str,
                       default='outputs/mpii_training/graphsage_20260311_184423',
                       help='Path to model directory')
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    # Initialize classifier
    classifier = LiveMultiPersonPoseClassifier(args.model_dir, args.device)

    # Run
    if args.mode == 'webcam':
        camera_id = int(args.source) if args.source.isdigit() else 0
        classifier.run_on_webcam(camera_id)
    else:
        classifier.run_on_video(args.source, args.output)

if __name__ == '__main__':
    main()
#python scripts/live_multiperson_pose.py --mode webcam
#python scripts/live_multiperson_pose.py --mode video --source path/to/video.mp4
#python scripts/live_multiperson_pose.py --mode video --source video.mp4 --output output.mp4
