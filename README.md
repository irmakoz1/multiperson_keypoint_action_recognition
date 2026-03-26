[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10+-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-00a8ff.svg)](https://github.com/ultralytics/ultralytics)
[![HuggingFace](https://img.shields.io/badge/Transformers-f9a8d4.svg)](https://huggingface.co/)
[![PyAV](https://img.shields.io/badge/PyAV-1.0+-yellow.svg)](https://github.com/PyAV-Org/PyAV)

# Real‑Time Multi‑Person Action Recognition from 2D Pose
  This project provides a complete pipeline for real‑time multi‑person action recognition from 2D skeletal keypoints. It combines state‑of‑the‑art detection (YOLO), tracking (Kalman + IOU), pose estimation (ViTPose), and temporal action models (GraphSAGE / Transformer) to classify actions like walking, jumping, waving, etc. The system runs in real‑time on a CPU (for now) and supports live webcam input, video files, and output recording.

### Features
- YOLO person bounding box detection.

- Real‑time multi‑person tracking with Kalman filter and IOU association (NMS to reduce duplicates).

- Pose estimation using ViTPose (COCO 17 joints) – accurate and robust.

- Rich feature extractor that computes biomechanical angles, relative positions, velocities, and confidence encoding.

- Custom temporal models for action recognition in 20 categories.
  
## Folder Structure
  ```

├── README.md
├── requirements.txt
├── LICENSE
├──   .env      #env variables
├── contraints.txt
├── setup.py
├── setup.bat
├── data/
│   ├── mpose/
│   │   ├── class_info.json      #MPOSE class information that models trained on
│   │   └── mpose_explore.py     #dataset exploration and visualisation
├── outputs/ (training outputs)
│    ├── embedding_vis   #PCA/t‑SNE embedding plots
│    ├── temporal
│    │   ├──graphsage_raw
│    │   │     ├── best_model.pth   # trained model (in every temporal model folder there is the trained model and the training history)
│    │   │     ├── history.json     #training history (in every temporal model folder there is the trained model and the training history)
│    │   ├──graphsage_temporal
│    │   ├──temporal_transformer_mpose
│    │   ├──transformer_raw_mpose
├── src/
│   ├── skeleton_extractor/
│   │   ├── yolo_wrapper_ultra.py      #YOLO person detection class
│   │   ├── multiperson_tracker.py     #Kalman filter, IoU , NMS class
│   │   └── vitpose_wrapper.py         #VitPose keypoint estimation class
│   ├── encoder/
│   │   ├── skeleton_encoder.py        #a general extractor class for adapting to other datasets
│   │   └── mpose_encoder.py           #extractor class adapted to mpose dataset.
│   ├── evaluation/
│   │   └── temporal/
│   │       ├── graphsage_with_preprocessing.py      #GraphSAGE model with embedding  
│   │       ├── temporal_transformer_mpose.py        #Transformer model with embedding
│   │       ├── transformer_raw.py                   #Transformer model without embedding
│   │       └── graphsage_noemb.py                   #GraphSAGE model without embedding
│   │       └── graphsage_embedding_extract.py       #GraphSAGE model that extracts embedding for exploration.
│   │       └── lstm_temp.py                         #Template for a GraphSAGE + LSTM model (not trained)
│   ├── models/   
│           ├── yolo12n.pt
│   ├── pipeline/
│   │   └── smooth_pipeline.py             #main real‑time inference script
│   ├── utils/
│   │   ├── evaluation_table.py                #summarise training results into CSV
│   │   ├── embedding_extract_temporal.py  #extract embedding visualisations from best model 
│   │   └── plot_training.py                #plot loss/accuracy curves
│   │   └── class_info.py        #extract class info from mpose dataset.
│   └── features/
│       └── joint_features.py      #coco joint feature mapping
```

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
**Optional:**

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

After training, the best model will be saved as best_model.pth in the respective output directory. In this repo, the models are already trained and saved. They are stored in the outputs directory.

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
```
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
```
### Two custom temporal architectures:

- **Spatio‑temporal GraphSAGE** – unified graph that processes space and time together (per joint, per frame).

  <img width="3000" height="1000" alt="Azure Databricks Data Lake-2026-03-24-003809" src="https://github.com/user-attachments/assets/659ae709-1f9a-46d0-be59-dd5b28d6b2fe" />

- **Temporal Transformer** – frame‑level transformer with optional raw keypoints.
<img width="3000" height="1000" alt="GraphSAGE Action-2026-03-26-152937" src="https://github.com/user-attachments/assets/5d31c033-11e8-4ea5-b784-cddefcec7b6b" />



### Models: 

| **Component** | **Model** | **Version / Variant** | **Source / Link** |
|---------------|-----------|-----------------------|-------------------|
| **Object Detection** | YOLO | `yolov12n.pt` (nano) | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| **Pose Estimation** | ViTPose | `vitpose-base-coco-aic-mpii` | [ViTPose on HuggingFace](https://huggingface.co/usyd-community/vitpose-base-coco-aic-mpii) |
| **Temporal (GraphSAGE)** | Spatio‑temporal GraphSAGE with embedding | Custom made in pygeometric– 3 layers, hidden dims [128,256,128] | [GraphSAGE paper](https://arxiv.org/abs/1706.02216) |
| **Temporal (Transformer)** | Temporal Transformer with embedding| Custom made in pytorch – 4 layers, 8 heads, hidden dim 128 | [Transformer paper](https://arxiv.org/abs/1706.03762) |
| **Temporal (GraphSAGE)** | Temporal GraphSAGE no embedding | Same GraphSAGE architecture as above without features from extractor |- |
| **Temporal (Transformer)** | Temporal Transformer no embedding | Same Transformer architecture without features from extractor |-  |
| **Feature Extractor** | MPOSEFeatureExtractor | Custom – includes angles, velocities, relative positions | – |
| **Tracker** | SimpleMPT | Kalman filter + IOU with NMS | Custom implementation |


## Temporal Model Performances:

| Model Type                     | Window Size | Batch Size | Hidden Dim | Num Layers | Num Heads | Dropout | Best Train Acc | Best Val Acc | Best Val Bal Acc | Best Val Loss |
|--------------------------------|-------------|------------|------------|------------|-----------|---------|----------------|--------------|------------------|---------------|
| Unified GraphSAGE (extractor)  | 20          | 64         | 128        | 3          | 4         | 0.3     | 0.8500         | 0.8001       | 0.7477           | 0.6301        |
| Temporal Transformer (raw)     | 20          | 64         | 128        | 4          | 8         | 0.3     | 0.9189         | 0.7862       | 0.7227           | 0.8847        |
| Unified GraphSAGE (raw)        | 20          | 64         | 128        | 3          | 4         | 0.3     | 0.8237         | 0.7729       | 0.7163           | 0.7238        |
| Temporal Transformer (extractor)| 20          | 64         | 128        | 4          | 8         | 0.3     | 0.7438         | 0.7192       | 0.6558           | 0.9680        |



## Architecture

<img width="480" height="1200" alt="Untitled diagram-2026-03-25-165006" src="https://github.com/user-attachments/assets/05dac7f7-d076-4585-9f87-94b439c3358c" />

**Capture thread:** reads frames from camera/video and pushes them into a queue.

**Process thread:** consumes frames at a reduced rate (e.g., every 3rd frame), runs YOLO on a downscaled image, updates tracker (NMS + Kalman), runs ViTPose on raw detection boxes, maintains per‑person sliding windows, and runs the temporal model when a window is full.

**Display thread:** consumes frames from the same queue as fast as possible and draws the latest tracks, skeletons, and action labels.


### Optimisation Tips:
  - Increase --process_interval (e.g., 4 or 5) to reduce processing load.

  - Lower --det_input_size to 320 240 for faster detection.

  - Use GPU (--device cuda) if available.

  - Quantise models with ONNX or TorchScript for faster CPU inference.

  - Disable skeleton drawing if not needed (--draw_skeleton False).

### Troubleshooting
    “No such file: …” – ensure you have class_info.json.

    PyAV cannot find camera – use camera index (e.g., 0) or install PyAV and provide the exact name as shown in Windows Device Manager.

    Low FPS – increase --process_interval, lower --det_input_size, and ensure you are using GPU.

    Duplicate tracks – adjust --iou_thresh and --max_lost in the tracker; the pipeline already applies NMS.

    Poor keypoint accuracy – increase detection confidence (--conf_thresh).

### Acknowledgements
This project uses several open‑source libraries:

  [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) – object detection.

  [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) – pose estimation.

  [MPOSE2021](https://github.com/PIC4SeR/MPOSE2021_Dataset) – dataset.

  [PyTorch Geometric](https://pytorch-geometric.com) – graph neural networks.

  [https://huggingface.co/docs/transformers/index](https://huggingface.co/docs/transformers/index) – transformer models.

  [Pygame](https://www.pygame.org/news) – display.

  [PyAV](https://github.com/PyAV-Org/PyAV) – camera capture.


