# CerberusNet

Multi-modal object detection for adverse weather, built on Faster R-CNN.

CerberusNet fuses **RGB camera**, **LiDAR**, and **radar** through a shared ResNet-18 + FPN encoder with four parallel detection heads. An auxiliary weather/daytime classifier enables condition-aware head selection at inference time.

Trained and evaluated on the [SeeingThroughFog](https://www.uni-ulm.de/en/in/driveu/projects/dense-datasets/) (STF) dataset.

---

## Features

- Shared ResNet-18 + FPN backbone with per-modality input stems
- Four detection heads: `rgb_lidar`, `rgb_radar`, `lidar_radar`, `all` (3-sensor)
- Auxiliary weather (clear / rain / snow / fog) and daytime (day / night) classifier
- Condition-based head routing at inference — selects the optimal head per frame
- Focal loss with class-balanced alpha for weather classification
- K-fold cross-validation with [Weights & Biases](https://wandb.ai) logging
- Standalone aux head fine-tuning with frozen backbone
- Modality-selective inference benchmarking

## Repository Structure

```
CerberusNet/
├── model.py          # Model architecture, losses, checkpointing
├── dataset.py        # STF dataset loading, LiDAR/radar projection, caching
├── train.py          # Multi-head detection training (k-fold CV)
├── aux_train.py      # Standalone auxiliary head training
├── metrics.py        # mAP, precision/recall, confusion matrices, aux metrics
├── inference.py      # Inference benchmarking across modality scenarios
├── viz.py            # Prediction visualization (bounding box grids)
└── tools/            # STF toolkit (calibration, label parsing, projection)
```

## Detected Classes

| Index | Class |
|-------|-------|
| 1 | PassengerCar |
| 2 | LargeVehicle |
| 3 | RidableVehicle |
| 4 | Pedestrian |

---

## Setup

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | ≥ 3.10.12 |
| PyTorch | ≥ 2.1 |
| torchvision | ≥ 0.20.1 (matching PyTorch) |
| CUDA | ≥ 13.2 (recommended) |
| GPU VRAM | ≥ 16 GB |

### 1. Clone the repository

```bash
git clone https://github.com/MichalWeis/CerberusNet.git
cd CerberusNet
```

### 2. Create environment & install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# PyTorch (adjust cu121 to match your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Other dependencies
pip install numpy opencv-python matplotlib seaborn tqdm wandb
```

### 3. Prepare the SeeingThroughFog dataset

Download the STF dataset and ensure the following directory layout:

```
SeeingThroughFog/
├── cam_stereo_left_lut/          # LUT-corrected left camera images (.png)
├── gt_labels/
│   └── cam_left_labels_TMP/      # KITTI-format bounding box labels
├── labeltool_labels/             # Per-frame condition JSONs (weather, daytime, …)
├── lidar_hdl64_last/             # Velodyne HDL-64 point clouds (.bin)
├── radar_targets/                # Radar target detections (.json)
├── calib_cam_stereo_left.json    # Camera intrinsics
└── calib_tf_tree_full.json       # Full extrinsic transform tree
```

### 4. Preprocess & cache the dataset

Raw dataset loading involves LiDAR projection, radar rasterisation, and calibration math. Run this once to cache everything as `.pt` files:

```bash
python -c "
from dataset import preprocess_and_save
preprocess_and_save(
    root_dir='/path/to/SeeingThroughFog',
    cache_dir='/path/to/cache',
    use_camera='left_lut',
)
"
```

---

## Usage

### Train the auxiliary head

Train the weather/daytime classifier with the detection backbone frozen:

```bash
# First, edit the constants at the top of aux_train.py:
#   CACHE_DIR, CHECKPOINT_PATH, AUX_CHECKPOINT_DIR

python aux_train.py
```

### Train the full model (5-fold CV)

```bash
python train.py \
    --dataset /path/to/SeeingThroughFog \
    --checkpoint-dir checkpoints/stf
    --aux-checkpoint /path/to/aux_checkpoint \
    # --no-wandb          # disable W&B logging
    # --resume ckpt.pt    # resume from checkpoint
```

Each fold trains for up to 8 epochs with early stopping (patience 10). Checkpoints are saved every 5 epochs and on best validation loss.

### Benchmark inference

```bash
# With dataset cache
python inference.py \
    --cache-dir /path/to/cache \
    --checkpoint checkpoints/stf/fold_0/epoch_0008.pt \
    --scenarios rgb rgb+lidar rgb+lidar+radar aux

# With synthetic data (no dataset needed)
python inference.py --synthetic --num-samples 500
```

---

## Condition-Aware Head Routing

The auxiliary head predicts weather and daytime from RGB features, then a lookup table selects the optimal detection head per frame:

| Daytime | Weather | Head |
|---------|---------|------|
| Day | Clear | All (3-sensor) |
| Day | Fog | All (3-sensor) |
| Day | Rain | RGB + LiDAR |
| Day | Snow | All (3-sensor) |
| Night | Clear | RGB + LiDAR |
| Night | Fog | All (3-sensor) |
| Night | Rain | RGB + LiDAR |
| Night | Snow | All (3-sensor) |

Set `eval_head="conditional"` when building the model to enable this routing at inference.

---

## License

This project is developed as part of a bachelor thesis.
