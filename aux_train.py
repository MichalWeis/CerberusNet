import copy
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchvision.transforms.functional as TF

import wandb

from model import (
    build_model,
    CheckpointManager,
    WEATHER_CLASSES,
    WEATHER_TO_IDX,
    DAYTIME_TO_IDX,
    _AUX_IGNORE_INDEX,
    build_aux_targets,
    multiclass_focal_loss,
    WeatherDaytimeHead,
)
from dataset import PreprocessedSTFDataset, collate_fn_stf

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9
NUM_EPOCHS = 60
EARLY_STOP_PATIENCE = 10

MAX_SAMPLES_PER_CLASS = 1000
RAIN_TARGET_COUNT = 1000
# light_fog and dense_fog are merged into a single "fog" class
FOG_SOURCE_KEYS = ("light_fog", "dense_fog")

CACHE_DIR = "/data/SeeingThroughFog/cache"
CHECKPOINT_PATH = (
    "/checkpoints/scheduler_Tmax/epoch_0100.pt"
)
AUX_CHECKPOINT_DIR = "/checkpoints/aux_head_1000_rain"


# ---------------------------------------------------------------------------
# RGB-only wrapper with optional augmentation
# ---------------------------------------------------------------------------

class RGBOnlyDataset(Dataset):
    """
    Wraps a PreprocessedSTFDataset (or Subset thereof).
    Returns only the RGB tensor + conditions; no detection targets needed.
    Optionally applies light spatial augmentations (for oversampled rain).
    """

    def __init__(self, base_dataset, indices: List[int], augment: bool = False):
        self.base = base_dataset
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        modalities, target, conditions = self.base[real_idx]
        rgb = modalities["rgb"]  # (3, H, W)

        if self.augment:
            # Light augmentations: ±10° rotation + random horizontal flip
            angle = random.uniform(-10, 10)
            rgb = TF.rotate(rgb, angle, fill=0.0)
            if random.random() > 0.5:
                rgb = TF.hflip(rgb)

        return rgb, conditions


def collate_aux(batch):
    """Collate for auxiliary training: stack RGB tensors, collect conditions."""
    rgbs, conditions = zip(*batch)
    return torch.stack(rgbs, dim=0), list(conditions)


# ---------------------------------------------------------------------------
# Index scanning: group samples by weather class
# ---------------------------------------------------------------------------

def scan_weather_indices(dataset) -> Dict[str, List[int]]:
    """
    Single pass through the dataset to group indices by weather label.
    Works with PreprocessedSTFDataset (each .pt has a 'conditions' dict).
    """
    groups: Dict[str, List[int]] = defaultdict(list)

    for idx in tqdm(range(len(dataset)), desc="Scanning weather labels"):
        data = torch.load(dataset.files[idx], weights_only=False)
        cond = data.get("conditions", None)
        if not isinstance(cond, dict):
            continue
        weather = cond.get("weather", "unknown")
        # Merge light_fog and dense_fog into a single fog class
        if weather in FOG_SOURCE_KEYS:
            weather = "fog"
        if weather in WEATHER_TO_IDX:
            groups[weather].append(idx)

    summary = "  ".join(f"{k}={len(v)}" for k, v in groups.items())
    print(f"[scan] Weather index counts: {summary}")
    return groups


# ---------------------------------------------------------------------------
# Build balanced index lists
# ---------------------------------------------------------------------------

def build_balanced_indices(
    groups: Dict[str, List[int]],
    max_per_class: int = MAX_SAMPLES_PER_CLASS,
    rain_target: int = RAIN_TARGET_COUNT,
) -> Tuple[List[int], List[int]]:
    """
    Returns two lists of dataset indices:
        normal_indices  – non-augmented samples (cap each class at max_per_class)
        rain_aug_indices – extra rain copies that will get augmented at load time

    The rain class is first capped at max_per_class like everyone else, then
    if the raw count is below rain_target we create additional copies (with
    augmentation flag) to reach rain_target total.
    """
    normal_indices: List[int] = []
    rain_aug_indices: List[int] = []

    for cls_name, idx_list in groups.items():
        random.shuffle(idx_list)
        capped = idx_list[:max_per_class]
        normal_indices.extend(capped)

        if cls_name == "rain":
            # How many more do we need to reach rain_target?
            deficit = rain_target - len(capped)
            if deficit > 0:
                # Oversample with replacement from the original rain indices
                extra = [random.choice(idx_list) for _ in range(deficit)]
                rain_aug_indices.extend(extra)
                print(
                    f"[balance] rain: {len(capped)} real + {deficit} augmented "
                    f"= {len(capped) + deficit} total"
                )
            else:
                print(f"[balance] rain: {len(capped)} (already >= {rain_target})")

    print(
        f"[balance] Total normal={len(normal_indices)}  "
        f"rain_aug={len(rain_aug_indices)}  "
        f"combined={len(normal_indices) + len(rain_aug_indices)}"
    )
    return normal_indices, rain_aug_indices


# ---------------------------------------------------------------------------
# Combined dataset: normal + augmented
# ---------------------------------------------------------------------------

class CombinedAuxDataset(Dataset):
    """Concatenates a non-augmented and an augmented RGBOnlyDataset."""

    def __init__(self, normal: RGBOnlyDataset, augmented: RGBOnlyDataset):
        self.normal = normal
        self.augmented = augmented
        self._len_normal = len(normal)

    def __len__(self):
        return self._len_normal + len(self.augmented)

    def __getitem__(self, idx):
        if idx < self._len_normal:
            return self.normal[idx]
        return self.augmented[idx - self._len_normal]


# ---------------------------------------------------------------------------
# Frozen-backbone feature extractor
# ---------------------------------------------------------------------------

class FrozenBackboneExtractor(nn.Module):
    """
    Wraps the RGB stem + shared encoder from a full MultiModalFasterRCNN,
    freezes all parameters, and exposes a forward that returns `c2`
    (the layer2 midpoint features the aux head consumes).
    """

    def __init__(self, full_model):
        super().__init__()
        self.rgb_stem = copy.deepcopy(full_model.stems["rgb"])
        self.encoder = copy.deepcopy(full_model.encoder)
        # Copy RGB normalisation buffers
        self.register_buffer("rgb_mean", full_model.rgb_mean.clone())
        self.register_buffer("rgb_std", full_model.rgb_std.clone())

        # Freeze everything
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: (B, 3, H, W) float tensor in [0, 1].
        Returns:
            c2: (B, 128, H/8, W/8) — ResNet-18 layer2 output.
        """
        x = (rgb - self.rgb_mean) / self.rgb_std
        stem_out = self.rgb_stem(x)
        c1 = self.encoder.layer1(stem_out)
        c2 = self.encoder.layer2(c1)
        return c2


# ---------------------------------------------------------------------------
# Training & evaluation loops
# ---------------------------------------------------------------------------

def train_one_epoch(backbone, head, dataloader, optimizer, device, epoch,
                    weather_alpha, weather_focal_gamma):
    head.train()
    backbone.eval()  # always eval (frozen)

    total_w_loss = 0.0
    total_d_loss = 0.0
    count = 0

    for rgb_batch, conditions in tqdm(dataloader, desc=f"Train Epoch {epoch+1}"):
        rgb_batch = rgb_batch.to(device)

        # Extract frozen features
        with torch.no_grad():
            c2 = backbone(rgb_batch)

        # Forward through trainable aux head
        aux_out = head(c2)

        w_tgt, d_tgt = build_aux_targets(conditions, device)

        loss = torch.zeros((), device=device)

        if (w_tgt != _AUX_IGNORE_INDEX).any():
            w_loss = multiclass_focal_loss(
                aux_out["weather_logits"], w_tgt,
                alpha=weather_alpha,
                gamma=weather_focal_gamma,
                ignore_index=_AUX_IGNORE_INDEX,
            )
            loss = loss + w_loss
            total_w_loss += w_loss.item()

        if (d_tgt != _AUX_IGNORE_INDEX).any():
            d_loss = F.cross_entropy(
                aux_out["daytime_logits"], d_tgt,
                ignore_index=_AUX_IGNORE_INDEX,
            )
            loss = loss + d_loss
            total_d_loss += d_loss.item()

        if loss.requires_grad:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(head.parameters(), clip_value=5.0)
            optimizer.step()

        count += 1

    return {
        "weather": total_w_loss / max(count, 1),
        "daytime": total_d_loss / max(count, 1),
        "total": (total_w_loss + total_d_loss) / max(count, 1),
    }


@torch.no_grad()
def evaluate(backbone, head, dataloader, device, epoch,
             weather_alpha, weather_focal_gamma):
    head.eval()
    backbone.eval()

    total_w_loss = 0.0
    total_d_loss = 0.0
    correct_w = 0
    correct_d = 0
    total_w = 0
    total_d = 0
    count = 0

    # Per-class weather accuracy tracking
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    # Per-class daytime accuracy tracking (e.g., day/night)
    per_daytime_correct = defaultdict(int)
    per_daytime_total = defaultdict(int)

    for rgb_batch, conditions in tqdm(dataloader, desc=f"Val Epoch {epoch+1}"):
        rgb_batch = rgb_batch.to(device)
        c2 = backbone(rgb_batch)
        aux_out = head(c2)

        w_tgt, d_tgt = build_aux_targets(conditions, device)

        if (w_tgt != _AUX_IGNORE_INDEX).any():
            w_loss = multiclass_focal_loss(
                aux_out["weather_logits"], w_tgt,
                alpha=weather_alpha,
                gamma=weather_focal_gamma,
                ignore_index=_AUX_IGNORE_INDEX,
            )
            total_w_loss += w_loss.item()

            valid_w = w_tgt != _AUX_IGNORE_INDEX
            preds_w = aux_out["weather_logits"][valid_w].argmax(dim=1)
            targets_w = w_tgt[valid_w]
            correct_w += (preds_w == targets_w).sum().item()
            total_w += valid_w.sum().item()

            # Per-class accuracy
            for pred, tgt in zip(preds_w.cpu().tolist(), targets_w.cpu().tolist()):
                per_class_total[tgt] += 1
                if pred == tgt:
                    per_class_correct[tgt] += 1

        if (d_tgt != _AUX_IGNORE_INDEX).any():
            d_loss = F.cross_entropy(
                aux_out["daytime_logits"], d_tgt,
                ignore_index=_AUX_IGNORE_INDEX,
            )
            total_d_loss += d_loss.item()

            valid_d = d_tgt != _AUX_IGNORE_INDEX
            preds_d = aux_out["daytime_logits"][valid_d].argmax(dim=1)
            targets_d = d_tgt[valid_d]
            correct_d += (preds_d == targets_d).sum().item()
            total_d += valid_d.sum().item()

            # Per-class daytime accuracy
            for pred, tgt in zip(preds_d.cpu().tolist(), targets_d.cpu().tolist()):
                per_daytime_total[tgt] += 1
                if pred == tgt:
                    per_daytime_correct[tgt] += 1

        count += 1

    w_acc = correct_w / max(total_w, 1)
    d_acc = correct_d / max(total_d, 1)

    # Per-class weather accuracy summary
    per_class_acc = {}
    for cls_idx in range(len(WEATHER_CLASSES)):
        n = per_class_total.get(cls_idx, 0)
        c = per_class_correct.get(cls_idx, 0)
        acc = c / max(n, 1)
        per_class_acc[WEATHER_CLASSES[cls_idx]] = {"acc": acc, "n": n, "correct": c}

    # Per-class daytime accuracy summary
    idx_to_daytime = {idx: name for name, idx in DAYTIME_TO_IDX.items()}
    per_daytime_acc = {}
    for cls_idx, cls_name in sorted(idx_to_daytime.items()):
        n = per_daytime_total.get(cls_idx, 0)
        c = per_daytime_correct.get(cls_idx, 0)
        acc = c / max(n, 1)
        per_daytime_acc[cls_name] = {"acc": acc, "n": n, "correct": c}

    return {
        "weather": total_w_loss / max(count, 1),
        "daytime": total_d_loss / max(count, 1),
        "total": (total_w_loss + total_d_loss) / max(count, 1),
        "weather_acc": w_acc,
        "daytime_acc": d_acc,
        "per_class": per_class_acc,
        "per_daytime": per_daytime_acc,
    }


# ---------------------------------------------------------------------------
# Early stopping (tracks accuracy — higher is better)
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.002):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("-inf")
        self.counter = 0

    def step(self, metric: float) -> bool:
        if metric > self.best + self.min_delta:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_aux(
    cache_dir: str = CACHE_DIR,
    checkpoint_path: str = CHECKPOINT_PATH,
    aux_checkpoint_dir: str = AUX_CHECKPOINT_DIR,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    use_wandb: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    aux_ckpt_dir = Path(aux_checkpoint_dir)
    aux_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load full dataset & scan weather labels
    # ------------------------------------------------------------------
    full_dataset = PreprocessedSTFDataset(cache_dir=cache_dir)
    weather_groups = scan_weather_indices(full_dataset)

    # ------------------------------------------------------------------
    # 2. Build balanced train / val split
    # ------------------------------------------------------------------
    # First, split each class 80/20 into train/val, then apply balancing
    # only to the train portion (val stays natural distribution for honest eval).
    train_indices_all: List[int] = []
    val_indices_all: List[int] = []
    train_groups: Dict[str, List[int]] = {}

    for cls_name, idx_list in weather_groups.items():
        random.shuffle(idx_list)
        split_pt = int(0.8 * len(idx_list))
        train_groups[cls_name] = idx_list[:split_pt]
        val_indices_all.extend(idx_list[split_pt:])

    # Apply capping + rain oversampling to the train groups
    normal_indices, rain_aug_indices = build_balanced_indices(
        train_groups,
        max_per_class=MAX_SAMPLES_PER_CLASS,
        rain_target=RAIN_TARGET_COUNT,
    )

    # Print per-class effective counts after balancing
    print("\n[balance] Effective train counts after balancing:")
    for cls_name in WEATHER_CLASSES:
        raw = len(train_groups.get(cls_name, []))
        capped = min(raw, MAX_SAMPLES_PER_CLASS)
        extra = 0
        if cls_name == "rain":
            extra = max(0, RAIN_TARGET_COUNT - capped)
        print(f"  {cls_name}: {raw} raw → {capped + extra} effective "
              f"({'capped' if raw > MAX_SAMPLES_PER_CLASS else 'all'}"
              f"{f' + {extra} augmented' if extra else ''})")

    # Build datasets
    train_normal = RGBOnlyDataset(full_dataset, normal_indices, augment=False)
    train_aug = RGBOnlyDataset(full_dataset, rain_aug_indices, augment=True)
    train_dataset = CombinedAuxDataset(train_normal, train_aug)
    val_dataset = RGBOnlyDataset(full_dataset, val_indices_all, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_aux, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_aux, num_workers=4, pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}  Val samples: {len(val_dataset)}")

    # ------------------------------------------------------------------
    # 3. Load backbone from checkpoint & freeze
    # ------------------------------------------------------------------
    from dataset import STF_CLASSES
    full_model = build_model(num_classes=len(STF_CLASSES) + 1)

    print(f"Loading backbone weights from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # The checkpoint stores the full model state_dict. Some aux-head tensors
    # can change shape when label space changes (e.g., 5 weather classes -> 4).
    # Load only matching keys and skip incompatible ones.
    ckpt_state = ckpt["model_state_dict"]
    model_state = full_model.state_dict()

    compatible_state = {}
    skipped_keys = []
    for key, value in ckpt_state.items():
        if key not in model_state:
            continue
        if model_state[key].shape == value.shape:
            compatible_state[key] = value
        else:
            skipped_keys.append((key, tuple(value.shape), tuple(model_state[key].shape)))

    load_info = full_model.load_state_dict(compatible_state, strict=False)
    print(
        f"Loaded {len(compatible_state)} checkpoint tensors; "
        f"skipped {len(skipped_keys)} shape-mismatched tensors."
    )
    if skipped_keys:
        print("Skipped keys due to shape mismatch:")
        for key, ckpt_shape, model_shape in skipped_keys:
            print(f"  - {key}: ckpt {ckpt_shape} vs model {model_shape}")
    if load_info.missing_keys:
        print(f"Missing keys after partial load: {len(load_info.missing_keys)}")

    backbone = FrozenBackboneExtractor(full_model).to(device)
    print(f"Backbone frozen — {sum(p.numel() for p in backbone.parameters()):,} params (all frozen)")

    # ------------------------------------------------------------------
    # 4. Fresh auxiliary head (trainable)
    # ------------------------------------------------------------------
    head = WeatherDaytimeHead(in_channels=128).to(device)
    trainable_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"Auxiliary head — {trainable_params:,} trainable params")

    # ------------------------------------------------------------------
    # 5. Focal-loss alpha from the *balanced* train counts
    # ------------------------------------------------------------------
    # Count effective samples per class in the balanced training set
    balanced_counts = []
    for cls_name in WEATHER_CLASSES:
        raw = len(train_groups.get(cls_name, []))
        capped = min(raw, MAX_SAMPLES_PER_CLASS)
        extra = max(0, RAIN_TARGET_COUNT - capped) if cls_name == "rain" else 0
        balanced_counts.append(capped + extra)

    counts_t = torch.tensor(balanced_counts, dtype=torch.float32)
    alpha = 1.0 / counts_t.clamp(min=1.0).sqrt()
    alpha = alpha / alpha.mean()
    weather_alpha = alpha.to(device)
    weather_focal_gamma = 2.0

    print(f"Focal-loss alpha (from balanced counts): "
          f"{dict(zip(WEATHER_CLASSES, alpha.tolist()))}")

    # ------------------------------------------------------------------
    # 6. Optimizer & scheduler
    # ------------------------------------------------------------------
    optimizer = AdamW(head.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    #scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=5e-5)
    early_stopper = EarlyStopping(patience=EARLY_STOP_PATIENCE)

    # ------------------------------------------------------------------
    # 7. W&B
    # ------------------------------------------------------------------
    if use_wandb:
        wandb.init(
            project="radiate-detection",
            name="aux-head-training",
            config={
                "task": "auxiliary_head_only",
                "backbone_checkpoint": checkpoint_path,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": WEIGHT_DECAY,
                "num_epochs": num_epochs,
                "max_per_class": MAX_SAMPLES_PER_CLASS,
                "rain_target": RAIN_TARGET_COUNT,
                "weather_focal_gamma": weather_focal_gamma,
                "balanced_counts": dict(zip(WEATHER_CLASSES, balanced_counts)),
                "focal_alpha": dict(zip(WEATHER_CLASSES, alpha.tolist())),
            },
        )

    # ------------------------------------------------------------------
    # 8. Training loop
    # ------------------------------------------------------------------
    best_acc = float("-inf")

    for epoch in range(num_epochs):
        train_losses = train_one_epoch(
            backbone, head, train_loader, optimizer, device, epoch,
            weather_alpha, weather_focal_gamma,
        )
        val_results = evaluate(
            backbone, head, val_loader, device, epoch,
            weather_alpha, weather_focal_gamma,
        )

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch+1}/{num_epochs}  (lr={lr:.6f})")
        print(f"  Train — weather: {train_losses['weather']:.4f}  "
              f"daytime: {train_losses['daytime']:.4f}")
        print(f"  Val   — weather: {val_results['weather']:.4f}  "
              f"daytime: {val_results['daytime']:.4f}")
        print(f"  Val accuracy — weather: {val_results['weather_acc']:.4f}  "
              f"daytime: {val_results['daytime_acc']:.4f}")

        # Per-class breakdown
        for cls_name, cls_info in val_results["per_class"].items():
            print(f"    {cls_name:12s}: {cls_info['correct']}/{cls_info['n']} "
                  f"= {cls_info['acc']:.3f}")
        print("  Daytime class accuracy breakdown:")
        for cls_name, cls_info in val_results["per_daytime"].items():
            print(f"    {cls_name:12s}: {cls_info['correct']}/{cls_info['n']} "
                  f"= {cls_info['acc']:.3f}")

        # W&B logging
        if use_wandb:
            log_data = {
                "epoch": epoch + 1,
                "lr": lr,
                "train/weather_loss": train_losses["weather"],
                "train/daytime_loss": train_losses["daytime"],
                "train/total_loss": train_losses["total"],
                "val/weather_loss": val_results["weather"],
                "val/daytime_loss": val_results["daytime"],
                "val/total_loss": val_results["total"],
                "val/weather_acc": val_results["weather_acc"],
                "val/daytime_acc": val_results["daytime_acc"],
            }
            for cls_name, cls_info in val_results["per_class"].items():
                log_data[f"val/weather_{cls_name}_acc"] = cls_info["acc"]
                log_data[f"val/weather_{cls_name}_n"] = cls_info["n"]
            for cls_name, cls_info in val_results["per_daytime"].items():
                log_data[f"val/daytime_{cls_name}_acc"] = cls_info["acc"]
                log_data[f"val/daytime_{cls_name}_n"] = cls_info["n"]
            wandb.log(log_data)

        # Save best model (by weather accuracy)
        metric = val_results["weather_acc"]
        if metric > best_acc:
            best_acc = metric
            save_path = aux_ckpt_dir / "best_aux_head.pt"
            torch.save({
                "epoch": epoch + 1,
                "head_state_dict": head.state_dict(),
                "weather_acc": metric,
                "daytime_acc": val_results["daytime_acc"],
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "weather_alpha": weather_alpha.cpu(),
                "balanced_counts": balanced_counts,
            }, save_path)
            print(f"  ★ Saved best aux head (weather_acc={metric:.4f})")

        # Periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch + 1,
                "head_state_dict": head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }, aux_ckpt_dir / f"aux_epoch_{epoch+1:04d}.pt")

        # Early stopping
        if early_stopper.step(metric):
            print(f"\nEarly stopping at epoch {epoch+1} "
                  f"(no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    # ------------------------------------------------------------------
    # 9. Done
    # ------------------------------------------------------------------
    print(f"\nTraining complete. Best weather accuracy: {best_acc:.4f}")
    print(f"Best checkpoint: {aux_ckpt_dir / 'best_aux_head.pt'}")

    if use_wandb:
        wandb.finish()


# ---------------------------------------------------------------------------
# Loading the trained aux head back into a full model
# ---------------------------------------------------------------------------

def load_aux_head_into_model(
    full_model: nn.Module,
    aux_checkpoint: str,
) -> nn.Module:
    """
    Load a trained auxiliary head checkpoint into an existing
    MultiModalFasterRCNN model, replacing its weather_daytime_head.

    Usage:
        model = build_model(num_classes=5)
        ckpt_mgr.load(...)  # load full model weights
        model = load_aux_head_into_model(model, "checkpoints/aux_head/best_aux_head.pt")
    """
    ckpt = torch.load(aux_checkpoint, map_location="cpu", weights_only=False)
    full_model.weather_daytime_head.load_state_dict(ckpt["head_state_dict"])
    print(f"Loaded aux head from {aux_checkpoint} "
          f"(epoch {ckpt['epoch']}, weather_acc={ckpt.get('weather_acc', '?')})")
    return full_model


if __name__ == "__main__":
    train_aux(
        cache_dir=CACHE_DIR,
        checkpoint_path=CHECKPOINT_PATH,
        aux_checkpoint_dir=AUX_CHECKPOINT_DIR,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        use_wandb=True,
    )