import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from model import build_model, CheckpointManager
from pathlib import Path
import wandb
from typing import Optional, Dict
from metrics import Metrics
from dataset import (
    SeeingThroughFogDataset,
    STF_CLASSES, 
    create_dataloaders,
    preprocess_and_save
)
 
# Training hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
ADAMW_BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.0005
NUM_EPOCHS = 8
EARLY_STOP_PATIENCE = 10
 
# Dataset configuration
DATASET_ROOT = "/data/SeeingThroughFog/SeeingThroughFog"
 

class EarlyStopping:
    """Stop training when validation loss hasn't improved for `patience` epochs."""
 
    def __init__(self, patience: int = 10, min_delta: float = 0.003):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_metric = float('-inf')
        self.counter = 0
 
    def step(self, val_loss: float) -> bool:
        """Returns True when training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
    
    def step_reversed(self, metric: float) -> bool:
        """Returns True when training should stop."""
        if metric > self.best_metric + self.min_delta:
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def freeze_batchnorm_stats(module: torch.nn.Module) -> None:
    """Set BatchNorm layers to eval mode so running stats do not update."""
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()


def _normalize_condition_value(value: str) -> str:
    """Normalize condition tags to stable snake_case tokens for logging keys."""
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")
    
def train_one_epoch(model, dataloader, optimizer, device, epoch=0):
    """
    Performs one complete pass over training data.
    
    The model returns a dict of losses keyed by "{head_name}/{loss_type}".
    All losses are summed into a single scalar for backpropagation.
    """
    model.train()
    total_loss = 0.0
    loss_count = 0
 
    for modalities, targets, _conditions in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
        modalities = list(modalities)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in t.items()} for t in targets]
 
        # Skip batch if any target has no boxes
        if any(t['boxes'].shape[0] == 0 for t in targets):
            continue
 
        loss_dict = model(modalities, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        if torch.isnan(losses) or torch.isinf(losses):
            print(f"Warning: Invalid loss detected: {losses}")
            continue
        
        total_loss += losses.item()
        loss_count += 1
 
        optimizer.zero_grad()
        losses.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=5.0)
        optimizer.step()
 
    avg_loss = total_loss / max(1, loss_count)
    return avg_loss
 
@torch.no_grad()
def evaluate(model, dataloader, device, epoch=0):
    """
    Evaluates the model on validation data without computing gradients.
    Keeps model in train mode to obtain losses from all heads.
    """
    model.train()  # Keep in train mode to get losses
    model.apply(freeze_batchnorm_stats)  # Freeze BN running mean/var updates
    total_loss = 0.0
    loss_count = 0
    
    for modalities, targets, _conditions in tqdm(dataloader, desc=f"Validating Epoch {epoch+1}"):
        modalities = list(modalities)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in t.items()} for t in targets]
 
        loss_dict = model(modalities, targets)
        losses = sum(loss_dict.values())
        
        if not (torch.isnan(losses) or torch.isinf(losses)):
            total_loss += losses.item()
            loss_count += 1
    
    avg_loss = total_loss / max(1, loss_count)
    return avg_loss
 
def evaluate_metrics(model, dataloader, device, score_threshold=0.05):
    """
    Evaluates every detection head and returns a separate Metrics
    instance for each one.

    Returns:
        dict[str, Metrics]: head_name -> Metrics calculator
    """

    model.eval()

    # One Metrics instance per head (created lazily on first batch)
    head_metrics: Dict[str, Metrics] = {}

    with torch.no_grad():
        for modalities, targets, batch_conditions in tqdm(dataloader, desc="Computing Metrics"):
            modalities = [
                {k: v.to(device) for k, v in m.items()}
                for m in modalities
            ]

            # all_detections: dict[head_name -> List[Dict per image]]
            all_detections = model(modalities)

            for head_name, head_outputs in all_detections.items():
                if head_name not in head_metrics:
                    head_metrics[head_name] = Metrics(
                        num_classes=len(STF_CLASSES) + 1,
                        class_names=STF_CLASSES + ["background"],
                        iou_thresholds=[0.5, 0.75],
                    )

                for output, target, conditions in zip(head_outputs, targets, batch_conditions):
                    if len(output['scores']) > 0:
                        mask = output['scores'] >= score_threshold
                        pred_boxes = output['boxes'][mask]
                        pred_scores = output['scores'][mask]
                        pred_labels = output['labels'][mask]
                    else:
                        pred_boxes = torch.zeros((0, 4))
                        pred_scores = torch.zeros((0,))
                        pred_labels = torch.zeros((0,), dtype=torch.int64)

                    gt_boxes = target['boxes']
                    gt_labels = target['labels']

                    head_metrics[head_name].update(
                        pred_boxes.cpu().detach().numpy(),
                        pred_scores.cpu().detach().numpy(),
                        pred_labels.cpu().detach().numpy(),
                        gt_boxes.cpu().detach().numpy(),
                        gt_labels.cpu().detach().numpy(),
                        conditions=conditions,
                    )

    return head_metrics
 
def train(
    dataset_root: str = DATASET_ROOT,
    use_camera: str = "left_lut",
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    checkpoint_dir: Optional[str] = None,
    use_wandb: bool = True,
    resume_from: Optional[str] = None,
    num_folds: int = 5,
    fold_index: int = 0,
    split_seed: int = 42,
):
    """
    Main training loop for SeeingThroughFog dataset.
    
    Args:
        dataset_root: Root directory of SeeingThroughFog dataset
        use_camera: Which camera to use ("left", "right", "left_lut", "right_lut")
        num_epochs: Number of epochs to train
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        checkpoint_dir: Directory to save checkpoints
        use_wandb: Whether to log to Weights & Biases
        resume_from: Path to checkpoint to resume from
        num_folds: Number of folds for k-fold split
        fold_index: Fold index used as validation split in k-fold mode
        split_seed: Seed for deterministic fold partitioning
    """
 
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create checkpoint directory
    if checkpoint_dir is None:
        checkpoint_dir = Path(__file__).resolve().parent / "checkpoints" / "stf"
    else:
        checkpoint_dir = Path(checkpoint_dir)
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
 
    # Initialize wandb
    if use_wandb:
        wandb.init(
            project="radiate-detection",
            name=f"{fold_index + 1}-fold-training",
            config={
                "dataset": "SeeingThroughFog",
                "camera": use_camera,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "adamw_betas": ADAMW_BETAS,
                "weight_decay": WEIGHT_DECAY,
                "num_epochs": num_epochs,
                "num_classes": len(STF_CLASSES),
                "classes": STF_CLASSES,
                "split_mode": "kfold",
                "num_folds": num_folds,
                "fold_index": fold_index,
                "split_seed": split_seed,
            },
        )
 
    # Create datasets
    preprocess_and_save(root_dir="/data/SeeingThroughFog/SeeingThroughFog", cache_dir="/data/SeeingThroughFog/cache")
    
    
    print(f"Loading SeeingThroughFog dataset from {dataset_root}")
    train_loader, val_loader = create_dataloaders(
        root_dir=dataset_root,
        cache_dir="/data/SeeingThroughFog/cache",
        batch_size=batch_size,
        use_camera=use_camera,
        split_mode="kfold",
        num_folds=num_folds,
        fold_index=fold_index,
        seed=split_seed,
    )
    
    print(
        f"Fold {fold_index + 1}/{num_folds} | "
        f"Train batches: {len(train_loader)}, "
        f"Val batches: {len(val_loader)}"
    )
    
    # Build model — no more in_channels; the model handles modalities internally
    model = build_model(
        num_classes=len(STF_CLASSES) + 1,  # +1 for background
        eval_head="all",
    ).to(device)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    
    # Set up optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=ADAMW_BETAS,
        weight_decay=WEIGHT_DECAY
    )
    
    # Cosine annealing with warm restarts: decays LR to near-zero over T_0
    # epochs, then resets. T_mult=2 doubles the period after each restart.
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=5e-5)

    # Early stopping
    early_stopper = EarlyStopping(patience=EARLY_STOP_PATIENCE)

    # Set up checkpoint manager (saves every 5 epochs)
    ckpt_mgr = CheckpointManager(
        model, optimizer, scheduler,
        save_dir=str(checkpoint_dir),
        save_every=5,
    )
 
    # Load checkpoint if resume_from is provided
    if resume_from is not None:
        start_epoch = ckpt_mgr.load(resume_from)
        print(f"Resumed from checkpoint: {resume_from} (epoch {start_epoch})")
 
    best_val_loss = float("inf")
    best_epoch = start_epoch
    best_state_dict = None
    last_epoch = start_epoch
    all_head_metrics = {}
 
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        last_epoch = epoch + 1
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = evaluate(model, val_loader, device, epoch)
        
        # Compute metrics for all heads
        all_head_metrics = evaluate_metrics(model, val_loader, device, score_threshold=0.3)

        # Print summary for each head
        all_head_results = {}
        for head_name, metrics_calc in all_head_metrics.items():
            print(f"\n--- Head: {head_name} ---")
            all_head_results[head_name] = metrics_calc.print_summary(head_name=head_name)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        for head_name, head_res in all_head_results.items():
            print(f"  [{head_name}] mAP@0.50: {head_res.get('mAP@0.50', 0):.4f}  "
                  f"mAP@0.75: {head_res.get('mAP@0.75', 0):.4f}  "
                  f"mAP@[0.5:0.95]: {head_res.get('mAP@[0.5:0.95]', 0):.4f}")
        
        # Log train/val metrics to wandb every epoch for monitoring
        if use_wandb:
            log_data = {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "fold_index": fold_index,
            }
            for head_name, head_res in all_head_results.items():
                log_data[f"{fold_index}/{head_name}/mAP@0.50"] = head_res.get('mAP@0.50', 0)
                log_data[f"{fold_index}/{head_name}/mAP@0.75"] = head_res.get('mAP@0.75', 0)
                log_data[f"{fold_index}/{head_name}/mAP@[0.5:0.95]"] = head_res.get('mAP@[0.5:0.95]', 0)
                log_data[f"{fold_index}/{head_name}/precision"] = head_res.get('precision', 0)
                log_data[f"{fold_index}/{head_name}/recall"] = head_res.get('recall', 0)
                log_data[f"{fold_index}/{head_name}/f1"] = head_res.get('f1', 0)


            # --- Per weather+daytime breakdowns for every head ---
            WEATHER_CONDITIONS = ["clear", "rain", "snow", "light_fog", "dense_fog"]
            DAYTIME_CONDITIONS = ["day", "night"]

            for head_name, metrics_calc in all_head_metrics.items():
                combo_indices = {
                    (weather, daytime): []
                    for weather in WEATHER_CONDITIONS
                    for daytime in DAYTIME_CONDITIONS
                }

                for idx, cond in enumerate(metrics_calc.sample_conditions):
                    if not isinstance(cond, dict):
                        continue

                    weather = _normalize_condition_value(cond.get("weather", "unknown"))
                    daytime = _normalize_condition_value(cond.get("daytime", "unknown"))

                    if (weather, daytime) in combo_indices:
                        combo_indices[(weather, daytime)].append(idx)

                for weather in WEATHER_CONDITIONS:
                    for daytime in DAYTIME_CONDITIONS:
                        indices = combo_indices[(weather, daytime)]
                        prefix = f"{fold_index}_{head_name}/{weather}/{daytime}"

                        if indices:
                            sub_metrics = Metrics(
                                num_classes=metrics_calc.num_classes,
                                class_names=metrics_calc.class_names,
                                iou_thresholds=metrics_calc.iou_thresholds,
                            )

                            for idx in indices:
                                pred_boxes, pred_scores, pred_labels = metrics_calc.predictions[idx]
                                gt_boxes, gt_labels = metrics_calc.ground_truths[idx]
                                sub_metrics.predictions.append((pred_boxes, pred_scores, pred_labels))
                                sub_metrics.ground_truths.append((gt_boxes, gt_labels))

                            combo_metrics = sub_metrics.compute()

                            log_data[f"{prefix}/mAP@0.50"] = combo_metrics.get("mAP@0.50", 0.0)
                            log_data[f"{prefix}/mAP@[0.5:0.95]"] = combo_metrics.get("mAP@[0.5:0.95]", 0.0)
                            log_data[f"{prefix}/precision"] = combo_metrics.get("precision", 0.0)
                            log_data[f"{prefix}/recall"] = combo_metrics.get("recall", 0.0)
                            log_data[f"{prefix}/f1"] = combo_metrics.get("f1", 0.0)
                            log_data[f"{prefix}/num_samples"] = len(indices)
                        else:
                            log_data[f"{prefix}/mAP@0.50"] = 0.0
                            log_data[f"{prefix}/mAP@[0.5:0.95]"] = 0.0
                            log_data[f"{prefix}/precision"] = 0.0
                            log_data[f"{prefix}/recall"] = 0.0
                            log_data[f"{prefix}/f1"] = 0.0
                            log_data[f"{prefix}/num_samples"] = 0

            wandb.log(log_data)
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            ckpt_mgr.save(epoch + 1, best_val_loss=best_val_loss, fold_index=fold_index)
            print(f"  Saved best model with val_loss: {best_val_loss:.4f}")
        
        # Save periodic checkpoint (every 5 epochs)
        ckpt_mgr.step(epoch)
        
        # Step scheduler
        scheduler.step()

        # Early stopping
        if early_stopper.step(val_loss):
            print(f"\nEarly stopping triggered after {epoch+1} epochs "
                  f"(no improvement for {EARLY_STOP_PATIENCE} epochs).")
            break

    # Restore best validation model before final validation summary
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"\nLoaded best model state from epoch {best_epoch} for final validation summary.")
    else:
        print("\nNo validation improvement tracked; evaluating current model on validation split.")

    # Compute final metrics on validation split
    final_val_loss = evaluate(model, val_loader, device, epoch=best_epoch)
    final_val_head_metrics = evaluate_metrics(model, val_loader, device, score_threshold=0.3)
    final_val_head_results = {}

    print("\n=== Final Validation Results ===")
    print(f"  Val Loss (best model): {final_val_loss:.4f}")
    for head_name, metrics_calc in final_val_head_metrics.items():
        print(f"\n--- Val Head: {head_name} ---")
        final_val_head_results[head_name] = metrics_calc.print_summary(head_name=head_name)

    # Log final validation summary once per fold
    if use_wandb:
        final_val_log_data = {
            "fold_index": fold_index,
            "num_folds": num_folds,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "val/final_loss": final_val_loss,
        }

        for head_name, head_res in final_val_head_results.items():
            final_val_log_data[f"val/final/{head_name}/mAP@0.50"] = head_res.get("mAP@0.50", 0.0)
            final_val_log_data[f"val/final/{head_name}/mAP@0.75"] = head_res.get("mAP@0.75", 0.0)
            final_val_log_data[f"val/final/{head_name}/mAP@[0.5:0.95]"] = head_res.get("mAP@[0.5:0.95]", 0.0)
            final_val_log_data[f"val/final/{head_name}/precision"] = head_res.get("precision", 0.0)
            final_val_log_data[f"val/final/{head_name}/recall"] = head_res.get("recall", 0.0)
            final_val_log_data[f"val/final/{head_name}/f1"] = head_res.get("f1", 0.0)

        WEATHER_CONDITIONS = ["clear", "rain", "snow", "light_fog", "dense_fog"]
        DAYTIME_CONDITIONS = ["day", "night"]

        for head_name, metrics_calc in final_val_head_metrics.items():
            combo_indices = {
                (weather, daytime): []
                for weather in WEATHER_CONDITIONS
                for daytime in DAYTIME_CONDITIONS
            }

            for idx, cond in enumerate(metrics_calc.sample_conditions):
                if not isinstance(cond, dict):
                    continue

                weather = _normalize_condition_value(cond.get("weather", "unknown"))
                daytime = _normalize_condition_value(cond.get("daytime", "unknown"))

                if (weather, daytime) in combo_indices:
                    combo_indices[(weather, daytime)].append(idx)

            for weather in WEATHER_CONDITIONS:
                for daytime in DAYTIME_CONDITIONS:
                    indices = combo_indices[(weather, daytime)]
                    prefix = f"final_{head_name}/{weather}/{daytime}"

                    if indices:
                        sub_metrics = Metrics(
                            num_classes=metrics_calc.num_classes,
                            class_names=metrics_calc.class_names,
                            iou_thresholds=metrics_calc.iou_thresholds,
                        )

                        for idx in indices:
                            pred_boxes, pred_scores, pred_labels = metrics_calc.predictions[idx]
                            gt_boxes, gt_labels = metrics_calc.ground_truths[idx]
                            sub_metrics.predictions.append((pred_boxes, pred_scores, pred_labels))
                            sub_metrics.ground_truths.append((gt_boxes, gt_labels))

                        combo_metrics = sub_metrics.compute()

                        final_val_log_data[f"{prefix}/mAP@0.50"] = combo_metrics.get("mAP@0.50", 0.0)
                        final_val_log_data[f"{prefix}/mAP@[0.5:0.95]"] = combo_metrics.get("mAP@[0.5:0.95]", 0.0)
                        final_val_log_data[f"{prefix}/precision"] = combo_metrics.get("precision", 0.0)
                        final_val_log_data[f"{prefix}/recall"] = combo_metrics.get("recall", 0.0)
                        final_val_log_data[f"{prefix}/f1"] = combo_metrics.get("f1", 0.0)
                        final_val_log_data[f"{prefix}/num_samples"] = len(indices)
                    else:
                        final_val_log_data[f"{prefix}/mAP@0.50"] = 0.0
                        final_val_log_data[f"{prefix}/mAP@[0.5:0.95]"] = 0.0
                        final_val_log_data[f"{prefix}/precision"] = 0.0
                        final_val_log_data[f"{prefix}/recall"] = 0.0
                        final_val_log_data[f"{prefix}/f1"] = 0.0
                        final_val_log_data[f"{prefix}/num_samples"] = 0

        wandb.log(final_val_log_data)
    
    # Final checkpoint
    ckpt_mgr.save(last_epoch)
    print(f"\nTraining complete. Final model saved to {checkpoint_dir}")
    
    if use_wandb:
        wandb.finish()
 
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train on SeeingThroughFog dataset')
    parser.add_argument('--dataset', '-d', default=DATASET_ROOT, help='Dataset root directory')
    parser.add_argument('--checkpoint-dir', default="/CerberusNet/checkpoints", help='Checkpoint directory')
    parser.add_argument('--no-wandb', action='store_true', help='Disable W&B logging')
    parser.add_argument('--resume', default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
 
    num_folds = 5
    base_checkpoint_dir = Path(args.checkpoint_dir)

    for fold_idx in range(num_folds):
        print(f"\n================ Fold {fold_idx + 1}/{num_folds} ================")
        fold_checkpoint_dir = base_checkpoint_dir / f"fold_{fold_idx}"

        train(
            dataset_root=args.dataset,
            use_camera='left_lut',
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            checkpoint_dir=str(fold_checkpoint_dir),
            use_wandb=not args.no_wandb,
            resume_from=args.resume,  
            num_folds=num_folds,
            fold_index=fold_idx,
            split_seed=42,
        )