[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10+-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-00a8ff.svg)](https://github.com/ultralytics/ultralytics)
[![HuggingFace](https://img.shields.io/badge/Transformers-f9a8d4.svg)](https://huggingface.co/)
[![PyAV](https://img.shields.io/badge/PyAV-1.0+-yellow.svg)](https://github.com/PyAV-Org/PyAV)

# Real‑Time Multi‑Person Action Recognition from 2D Pose
  This project provides a complete pipeline for real‑time multi‑person action recognition from 2D skeletal keypoints. It combines state‑of‑the‑art detection (YOLO), tracking (Kalman + IOU), pose estimation (ViTPose), and temporal action models (GraphSAGE / Transformer) to classify actions like walking, jumping, waving, etc. The system runs in real‑time on a standard CPU/GPU and supports live webcam input, video files, and output recording.

### Features
- Real‑time multi‑person tracking with Kalman filter and IOU association (NMS to reduce duplicates).

- Pose estimation using ViTPose (COCO 17 joints) – accurate and robust.

- Rich feature extractor that computes biomechanical angles, relative positions, velocities, and confidence encoding.

### Two temporal architectures:

- Spatio‑temporal GraphSAGE – unified graph that processes space and time together.

  <img width="3000" height="1000" alt="Azure Databricks Data Lake-2026-03-24-003809" src="https://github.com/user-attachments/assets/659ae709-1f9a-46d0-be59-dd5b28d6b2fe" />

- Temporal Transformer – frame‑level transformer with optional raw keypoints.
<img width="3000" height="1000" alt="GraphSAGE Action-2026-03-26-152937" src="https://github.com/user-attachments/assets/5d31c033-11e8-4ea5-b784-cddefcec7b6b" />


  Sliding window inference – buffers per‑person keypoints over the last 20‑30 frames.

  Smooth live display – three‑thread design separates capture, heavy processing, and display.

  Customisable drawing – bounding boxes, skeletons, action labels with confidence.

  Supports camera by name (Windows DirectShow via PyAV) or index.

  Video saving to MP4.

## Architecture



<img width="800" height="1600" alt="Untitled diagram-2026-03-25-165006" src="https://github.com/user-attachments/assets/05dac7f7-d076-4585-9f87-94b439c3358c" />


















**Capture thread:** reads frames from camera/video and pushes them into a queue.

**Process thread:** consumes frames at a reduced rate (e.g., every 3rd frame), runs YOLO on a downscaled image, updates tracker (NMS + Kalman), runs ViTPose on raw detection boxes, maintains per‑person sliding windows, and runs the temporal model when a window is full.

**Display thread:** consumes frames from the same queue as fast as possible and draws the latest tracks, skeletons, and action labels.


## Installation
Clone the repository:
```
bash
git clone https://github.com/irmakoz1/multiperson_keypoint_action_recognition.git
cd multiperson_keypoint_action_recognition
```
Create a virtual environment (optional) and install requirements:

```
bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Training
PyGeometric custom GraphSAGE (Unified Spatio‑Temporal with embedding)
```
bash
python src/evaluation/temporal/graphsage_with_preprocessing.py \
    --window_size 20 \
    --batch_size 64 \
    --epochs 50 \
    --split 1 \
    --output_dir ./models/graphsage
```


PyTorch custom Temporal Transformer (with embedding)
```
bash
python src/evaluation/temporal/temporal_transformer_mpose.py \
    --seq_len 20 \
    --batch_size 64 \
    --epochs 50 \
    --split 1 \
    --output_dir ./models/transformer
```

After training, the best model will be saved as best_model.pth in the respective output directory. In this repo, the models are already trained and saved.

## Live Inference
With a camera by name (Windows DirectShow)
```
bash
python src/pipeline/smooth_live_pipeline.py 
    --video "name_of_your_camera" 
    --model_path ./models/graphsage/best_model.pth 
    --class_info data/mpose/class_info.json 
    --temporal_model graphsage 
    --window_size 20 
    --det_input_size 480 360 
    --process_interval 3 
    --draw_skeleton --draw_boxes --draw_labels
```

With a video file:
```
bash
python src/pipeline/smooth_live_pipeline.py 
    --video path/to/video.mp4 
    --model_path ./models/transformer/best_model.pth 
    --class_info data/mpose/class_info.json 
    --temporal_model transformer 
    --window_size 20 
    --det_input_size 480 360 
    --process_interval 3
```

### Argument	Description
  --video	Camera index (e.g., 0) or camera name (e.g., "PC-LM1E Camera") or video file path.

  --model_path	Path to trained best_model.pth.

  --class_info	Path to class_info.json.

  --temporal_model	Either graphsage or transformer.

  --window_size	Number of frames in sliding window (must match training).

  --det_input_size	Downscaled size for YOLO detection (e.g., 480 360).

  --process_interval	Process every N frames (higher = faster but slower reaction).

  --draw_skeleton	Draw skeleton lines and joints.

  --draw_boxes	Draw bounding boxes.

  --draw_labels	Draw action label and confidence.

  --output	Save output video (MP4).

### Models: 

| **Component** | **Model** | **Version / Variant** | **Source / Link** |
|---------------|-----------|-----------------------|-------------------|
| **Object Detection** | YOLO | `yolov12n.pt` (nano) | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| **Pose Estimation** | ViTPose | `vitpose-base-coco-aic-mpii` | [ViTPose on HuggingFace](https://huggingface.co/usyd-community/vitpose-base-coco-aic-mpii) |
| **Temporal (GraphSAGE)** | Spatio‑temporal GraphSAGE | Custom made in pygeometric– 3 layers, hidden dims [128,256,128] | [GraphSAGE paper](https://arxiv.org/abs/1706.02216) |
| **Temporal (Transformer)** | Temporal Transformer | Custom mace in pytorch – 4 layers, 8 heads, hidden dim 128 | [Transformer paper](https://arxiv.org/abs/1706.03762) |
| **Feature Extractor** | MPOSEFeatureExtractor | Custom – includes angles, velocities, relative positions | – |
| **Tracker** | SimpleMPT | Kalman filter + IOU with NMS | Custom implementation |

# Optimisation Tips:
  - Increase --process_interval (e.g., 4 or 5) to reduce processing load.

  - Lower --det_input_size to 320 240 for faster detection.

  - Use GPU (--device cuda) if available – this is the biggest speedup.

  - Quantise models with ONNX or TorchScript for faster CPU inference.

  - Disable skeleton drawing if not needed (--draw_skeleton False).

### Troubleshooting
  “No such file: …” – ensure you have class_info.json.

  PyAV cannot find camera – use camera index (e.g., 0) or install PyAV and provide the exact name as shown in Windows Device Manager.

  Low FPS – increase --process_interval, lower --det_input_size, and ensure you are using GPU.

  Duplicate tracks – adjust --iou_thresh and --max_lost in the tracker; the pipeline already applies NMS.

  Poor keypoint accuracy – increase detection confidence (--conf_thresh), use raw detection boxes (the pipeline does this by default after the tracker update), or expand bounding   boxes slightly in the code.

### Acknowledgements
This project uses several open‑source libraries:

[Ultralytics YOLO](https://github.com/ultralytics/ultralytics) – object detection.

[ViTPose](https://github.com/ViTAE-Transformer/ViTPose) – pose estimation.

[MPOSE2021](https://github.com/PIC4SeR/MPOSE2021_Dataset) – dataset.

[PyTorch Geometric](https://pytorch-geometric.com) – graph neural networks.

[https://huggingface.co/docs/transformers/index](https://huggingface.co/docs/transformers/index) – transformer models.

[Pygame](https://www.pygame.org/news) – display.

[PyAV](https://github.com/PyAV-Org/PyAV) – camera capture.

### Citation
If you use this work, please cite the relevant papers:

MPOSE2021: Mazzia, V. et al., Action Transformer: A Self‑Attention Model for Short‑Time Pose‑Based Human Action Recognition, Pattern Recognition 2021.

GraphSAGE: Hamilton, W. et al., Inductive Representation Learning on Large Graphs, NeurIPS 2017.

ViTPose: Xu, Y. et al., ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation, NeurIPS 2022.

YOLO: Jocher, G. et al., Ultralytics YOLOv8.
