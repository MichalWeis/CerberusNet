import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import torch
from tqdm import tqdm
import random
import numpy as np
from dataset import STF_CLASSES


def _batch_modalities(modalities, device):
    """Add batch dim and move every modality tensor to device."""
    return {k: v.unsqueeze(0).to(device) for k, v in modalities.items()}


@torch.no_grad()
def find_frames_with_predictions(
    model, dataset, device, score_threshold=0.3, min_predictions=1
):
    """
    Search a dataset for frames where the model makes detections.

    Returns:
        List of (dataset_index, num_predictions) tuples, up to 16 entries,
        in random order.
    """
    model.eval()
    results = []
    for idx in tqdm(range(len(dataset)), desc="Searching frames"):
        modalities, _ = dataset[idx]
        batch = _batch_modalities(modalities, device)
        outputs = model(batch)[0]
        num_preds = (outputs['scores'] >= score_threshold).sum().item()
        if num_preds >= min_predictions:
            results.append((idx, num_preds))

    random.shuffle(results)
    return results[:16]

def _to_display_image(img):
    # torch -> numpy
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    # CHW -> HWC
    if img.ndim == 3 and img.shape[0] in (1, 3, 4, 5):
        img = np.transpose(img, (1, 2, 0))

    # If fused (e.g., 5 channels), keep RGB only for display
    if img.ndim == 3 and img.shape[2] > 4:
        img = img[:, :, :3]

    # If single-channel with trailing dim
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]

    # Ensure float range is valid for imshow
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    return np.clip(img, 0.0, 1.0)


def _split_modal_views(img):
    arr = img
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()

    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4, 5, 7):
        arr = np.transpose(arr, (1, 2, 0))

    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)

    if arr.ndim == 2:
        rgb = np.stack([arr, arr, arr], axis=-1)
        return rgb, None

    ch = arr.shape[2]
    if ch >= 3:
        rgb = arr[:, :, :3]
    elif ch == 1:
        rgb = np.repeat(arr, 3, axis=2)
    else:
        rgb = _to_display_image(arr)

    lidar = None
    if ch >= 5:
        depth = np.clip(arr[:, :, 3], 0.0, 1.0)
        intensity = np.clip(arr[:, :, 4], 0.0, 1.0)
        hit_mask = depth > 0.0
        lidar = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.float32)
        if np.any(hit_mask):
            cmap = plt.get_cmap("turbo")
            depth_rgb = cmap(depth)[..., :3].astype(np.float32)
            # blend depth color with intensity only on pixels that have a LiDAR return
            lidar[hit_mask] = np.clip(
                0.75 * depth_rgb[hit_mask] + 0.25 * intensity[hit_mask, None],
                0.0, 1.0
            )

    return rgb, lidar


def _draw_boxes(ax, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, score_threshold):
    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            x1, max(0, y1 - 4),
            f"{_label_name(label)} {score:.2f}",
            color='red', fontsize=6, clip_on=True
        )

    for box, label in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor='lime', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            x1, max(0, y1 - 4),
            f"GT:{_label_name(label)}",
            color='lime', fontsize=6, clip_on=True
        )

def _label_name(label_id: int) -> str:
    """Convert 1-based model label id to class name string."""
    idx = int(label_id)
    if 0 <= idx < len(STF_CLASSES):
        return STF_CLASSES[idx]
    return f"cls_{label_id}"


@torch.no_grad()
def visualize_predictions_grid(
    model,
    dataset,
    device,
    frame_indices,          # list of (dataset_idx, num_preds)
    score_threshold=0.3,
    show_gt=True,
    save_path=None,
):
    """
    Create a 4x4 grid of RGB images with predicted and ground-truth boxes.

    Args:
        model: Trained detection model.
        dataset: Dataset instance (not DataLoader).
        device: torch.device.
        frame_indices: List of (idx, num_preds) from find_frames_with_predictions.
        score_threshold: Minimum confidence for showing a prediction.
        show_gt: Whether to overlay ground-truth boxes in green.
        save_path: If given, saves the figure here instead of showing it.
    """
    model.eval()
    indices = [idx for idx, _ in frame_indices[:16]]

    cols = 4
    rows = max(1, (len(indices) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.8))
    axes = np.array(axes).reshape(rows, cols)

    for sample_idx, ds_idx in enumerate(indices):
        row = sample_idx // cols
        col = sample_idx % cols
        ax = axes[row, col]

        modalities, target = dataset[ds_idx]
        batch = _batch_modalities(modalities, device)
        outputs = model(batch)[0]

        pred_boxes  = outputs['boxes'].detach().cpu().numpy()
        pred_scores = outputs['scores'].detach().cpu().numpy()
        pred_labels = outputs['labels'].detach().cpu().numpy()

        gt_boxes  = target['boxes'].cpu().numpy()  if show_gt else []
        gt_labels = target['labels'].cpu().numpy() if show_gt else []

        image_np = _to_display_image(modalities["rgb"])

        ax.imshow(image_np)
        _draw_boxes(ax, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, score_threshold)
        ax.set_title(f"#{ds_idx}", fontsize=8)
        ax.axis('off')

    for sample_idx in range(len(indices), rows * cols):
        row = sample_idx // cols
        col = sample_idx % cols
        axes[row, col].axis('off')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"Saved predictions grid to: {save_path}")
        plt.close(fig)
    else:
        plt.show()