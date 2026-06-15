# 🚬 Cigarette Defect Detection — YOLOv8 | ITC Limited

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![Status](https://img.shields.io/badge/Status-Production%20Deployed-brightgreen)
![Domain](https://img.shields.io/badge/Domain-Industrial%20QA-blueviolet)
![Internship](https://img.shields.io/badge/Internship-ITC%20Limited-red)

> Real-time multi-class cigarette defect detection system deployed on a live
> production line at ITC Limited's manufacturing facility, Munger, Bihar (Jun–Aug 2025).

## Overview

Cigarette manufacturing at scale requires continuous quality monitoring across
thousands of units per minute. Manual inspection is error-prone and slow.

This project builds a **production-deployed computer vision pipeline** that
detects structural and cosmetic defects in cigarettes in **real time** on the
factory floor using YOLOv8 — enabling live quality control with zero human
intervention in the detection loop.

### Problem
Traditional QA methods flagged defects post-production, leading to material
waste and rework costs. The goal was to catch defects *on the line*, instantly.

### Solution
A custom-trained YOLOv8 object detection model integrated into the production
line camera feed, classifying defects across multiple classes in real time.


## Defect Classes Detected

| Class | Description |
|-------|-------------|
| `tobacco_spillage` | Loose tobacco leaking from the tip |
| `crushed_tip`      | Deformed or collapsed cigarette tip |
| `crushed_bottom`   | Deformed or collapsed filter end |
| `wrinkle`          | Paper wrinkles on the cigarette body |
| `shape_deformation`| Non-cylindrical / bent shape |
| `good`             | No defect — passes QA |


## Pipeline Architecture

```
Camera Feed (Production Line)
        │
        ▼
Frame Extraction & Preprocessing
  └── Resize → 640×640
  └── Normalize pixel values
  └── Batch inference
        │
        ▼
YOLOv8 Inference Engine
  └── Backbone: CSPDarknet
  └── Neck: PANet (multi-scale)
  └── Head: Detection (boxes + classes)
        │
        ▼
Post-Processing
  └── NMS (IoU threshold: 0.45)
  └── Confidence filtering (threshold: 0.25)
  └── Class label assignment
        │
        ▼
Real-time Output
  └── Bounding boxes drawn on frame
  └── Defect class + confidence shown
  └── Alert triggered for flagged units
```


## Dataset

- **Source:** Custom-curated and annotated dataset — collected from the ITC
  production facility under real lighting and industrial conditions
- **Annotation format:** YOLO format (`.txt` bounding box files)
- **Annotation tool:** Roboflow / LabelImg
- **Classes:** 6 (5 defect types + 1 good class)
- **Augmentations applied:**
  - Horizontal flip
  - Brightness & contrast variation (simulates lighting changes)
  - Mosaic augmentation (YOLOv8 default)
  - Random crop and rotation

## Training Configuration

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')   # Pretrained COCO weights (transfer learning)

results = model.train(
    data    = 'cigarette_defect.yaml',
    epochs  = 100,
    imgsz   = 640,
    batch   = 16,
    lr0     = 0.01,           # Initial learning rate
    lrf     = 0.01,           # Final LR factor
    momentum= 0.937,
    weight_decay = 0.0005,
    warmup_epochs = 3,
    device  = 'cuda'
)
```

### Models Benchmarked

| Model     | mAP50  | mAP50-95 | Inference (ms) | Params  |
|-----------|--------|----------|----------------|---------|
| YOLOv8n   | 0.882  | 0.673    | 4.2 ms         | 3.2M    |
| YOLOv8s   | 0.901  | 0.701    | 7.8 ms         | 11.2M   |
| YOLOv8m   | 0.913  | 0.718    | 14.1 ms        | 25.9M   |

> YOLOv8n was selected for production — best balance of speed vs accuracy
> for real-time factory line constraints.


## Tech Stack

- **Model:** YOLOv8 (Ultralytics)
- **Language:** Python 3.10
- **Libraries:** PyTorch, OpenCV, Ultralytics, NumPy, Matplotlib
- **Hardware:** NVIDIA GPU (CUDA-enabled)
- **Annotation:** Roboflow / LabelImg
- **Deployment:** On-premise (factory floor camera integration)

## Acknowledgments

Developed during a Data Science Internship at **ITC Limited, Munger, Bihar**
(Jun–Aug 2025). The dataset, production environment, and domain expertise
were provided by ITC's manufacturing QA team.

> ⚠️ **Note:** Proprietary data, internal deployment scripts, and ITC-specific
> configurations are not included in this repository per NDA obligations.

## Author

**Shray Kant** — B.Tech Computer Science (Data Science), Bennett University
- GitHub: [@sky802](https://github.com/sky802)
- LinkedIn: [linkedin.com/in/shraykant17](https://linkedin.com/in/shraykant17)



