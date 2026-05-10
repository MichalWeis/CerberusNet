import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from tqdm import tqdm
from model import build_model
from pathlib import Path
import wandb
from typing import Optional, Dict, Tuple
import numpy as np
from tools.DatasetViewer.lib.read import load_calib_data, read_label, load_radar_points
from metrics import Metrics
from dataset import (
    SeeingThroughFogDataset, 
    STF_CLASSES, 
    collate_fn_stf,
    create_dataloaders,
)
from viz import visualize_predictions_grid, find_frames_with_predictions

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
NUM_EPOCHS = 1

# Dataset configuration
DATASET_ROOT = "/home/misko/projects/BP/data/SeeingThroughFog/SeeingThroughFog"

def train_one_epoch(model, dataloader, optimizer, device, epoch=0):
    """
    Performs one complete pass over training data.
    
    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): The DataLoader providing training data.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        device (torch.device): The device to run the training on.
        epoch (int): Current epoch number for logging.
    
    Returns:
        float: The average loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    loss_count = 0

    for images, targets in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in t.items()} for t in targets]

        # Skip batch if any target has no boxes
        if any(t['boxes'].shape[0] == 0 for t in targets):
            continue

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        if torch.isnan(losses) or torch.isinf(losses):
            print(f"Warning: Invalid loss detected: {losses}")
            continue
        
        total_loss += losses.item()
        loss_count += 1

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # add this
        optimizer.step()

    avg_loss = total_loss / max(1, loss_count)
    return avg_loss

@torch.no_grad()
def evaluate(model, dataloader, device, epoch=0):
    """
    Evaluates the model on validation data without computing gradients.
    
    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The DataLoader providing validation data.
        device (torch.device): The device to run evaluation on.
        epoch (int): Current epoch number for logging.
    
    Returns:
        float: The average loss over the validation set.
    """
    model.train()  # Keep in train mode to get losses
    total_loss = 0.0
    loss_count = 0
    
    for images, targets in tqdm(dataloader, desc=f"Validating Epoch {epoch+1}"):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        if not (torch.isnan(losses) or torch.isinf(losses)):
            total_loss += losses.item()
            loss_count += 1
    
    avg_loss = total_loss / max(1, loss_count)
    return avg_loss

def evaluate_metrics(model, dataloader, device, score_threshold=0.05):
    """
    Evaluates the model and computes detection metrics (mAP, etc.).
    
    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The DataLoader providing validation data.
        device (torch.device): The device to run evaluation on.
        score_threshold (float): Minimum confidence score for predictions.
    
    Returns:
        Metrics: Metrics calculator object with computed statistics.
    """
    model.eval()
    metrics_calculator = Metrics(
        num_classes=len(STF_CLASSES),
        class_names=STF_CLASSES,
        iou_thresholds=[0.5, 0.75]
    )

    for images, targets in tqdm(dataloader, desc="Computing Metrics"):
        images = list(image.to(device) for image in images)

        outputs = model(images)
        for output, target in zip(outputs, targets):
            # Filter predictions by score threshold
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
            
            metrics_calculator.update(
                pred_boxes.cpu().detach().numpy(),
                pred_scores.cpu().detach().numpy(),
                pred_labels.cpu().detach().numpy(),
                gt_boxes.cpu().detach().numpy(),
                gt_labels.cpu().detach().numpy()
            )
    
    return metrics_calculator

def train(
    dataset_root: str = DATASET_ROOT,
    use_camera: str = "left_lut",
    use_radar: bool = False,
    project_radar: bool = False,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    checkpoint_dir: Optional[str] = None,
    use_wandb: bool = True,
    resume_from: Optional[str] = None,
):
    """
    Main training loop for SeeingThroughFog dataset.
    
    Args:
        dataset_root: Root directory of SeeingThroughFog dataset
        use_camera: Which camera to use ("left", "right", "left_lut", "right_lut")
        use_radar: Whether to include radar data
        project_radar: Project radar detections to camera coordinates
        num_epochs: Number of epochs to train
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        checkpoint_dir: Directory to save checkpoints
        use_wandb: Whether to log to Weights & Biases
        resume_from: Path to checkpoint to resume from
    """

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create checkpoint directory
    # if checkpoint_dir is None:
    #     checkpoint_dir = Path(__file__).resolve().parent / "checkpoints" / "stf"
    # else:
    #     checkpoint_dir = Path(checkpoint_dir)
    
    # checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    if use_wandb:
        wandb.init(
            project="radiate-detection",
            config={
                "dataset": "SeeingThroughFog",
                "camera": use_camera,
                "use_radar": use_radar,
                "project_radar": project_radar,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "momentum": MOMENTUM,
                "weight_decay": WEIGHT_DECAY,
                "num_epochs": num_epochs,
                "num_classes": len(STF_CLASSES),
                "classes": STF_CLASSES,
            },
            tags=["SeeingThroughFog", use_camera, "radar" if use_radar else "camera_only"]
        )

    # Create datasets
    print(f"Loading SeeingThroughFog dataset from {dataset_root}")
    train_loader, val_loader = create_dataloaders(
        root_dir=dataset_root,
        batch_size=batch_size,
        use_camera=use_camera,
        use_radar=use_radar,
        project_radar=project_radar,
        split_ratio=0.8,
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Build model
    in_channels = 7 if (use_radar and project_radar) else 5
    model = build_model(num_classes=len(STF_CLASSES), in_channels=in_channels).to(device)

    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_from and Path(resume_from).exists():
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resumed from checkpoint: {resume_from} (epoch {start_epoch})")
    
    # Set up optimizer
    optimizer = SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    # best_map = float('-inf')
    # best_checkpoint = checkpoint_dir / "best_model.pth"

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = evaluate(model, val_loader, device, epoch)
        
        # Compute metrics
        metrics_calc = evaluate_metrics(model, val_loader, device, score_threshold=0.3)
        metrics = metrics_calc.print_summary()
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  mAP@0.50: {metrics.get('mAP@0.50', 0):.4f}")
        print(f"  mAP@0.75: {metrics.get('mAP@0.75', 0):.4f}")
        print(f"  mAP@[0.5:0.95]: {metrics.get('mAP@[0.5:0.95]', 0):.4f}")
        
        # Log to wandb
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "mAP@0.50": metrics.get('mAP@0.50', 0),
                "mAP@0.75": metrics.get('mAP@0.75', 0),
                "mAP@[0.5:0.95]": metrics.get('mAP@[0.5:0.95]', 0),
                "precision": metrics.get('precision', 0),
                "recall": metrics.get('recall', 0),
                "f1": metrics.get('f1', 0),
                "total_tp": metrics.get('total_tp', 0),
                "total_fp": metrics.get('total_fp', 0),
                "total_fn": metrics.get('total_fn', 0),
            })
        
        # Save best model
        # current_map = metrics.get('mAP@0.50', 0)
        # if current_map > best_map:
        #     best_map = current_map
        #     checkpoint = {
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'best_map': best_map,
        #     }
        #     torch.save(checkpoint, best_checkpoint)
        #     print(f"  Saved best model with mAP@0.50: {best_map:.4f}")
        
        # Save periodic checkpoint
        # if (epoch + 1) % 5 == 0:
        #     periodic_checkpoint = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
        #     torch.save(checkpoint, periodic_checkpoint)
        
        # Step scheduler
        #scheduler.step()
    
    # Final checkpoint
    # final_checkpoint = checkpoint_dir / "final_model.pth"
    # torch.save({
    #     'epoch': num_epochs - 1,
    #     'model_state_dict': model.state_dict(),
    # }, final_checkpoint)
    # print(f"\nTraining complete. Final model saved to {final_checkpoint}")
    
    if use_wandb:
        wandb.finish()
    
    Path("images").mkdir(exist_ok=True)

    # Save confusion matrix
    metrics_calc.plot_confusion_matrix(
        iou_threshold=0.5,
        save_path="images/confusion_matrix.png"
    )

    # Get TP/FP/FN breakdown
    results = metrics_calc.get_tp_fp_fn_tn(iou_threshold=0.5)
    print(f"TP: {results['total_TP']}, FP: {results['total_FP']}, FN: {results['total_FN']}")

    # Get underlying val dataset (works with random_split or plain dataset)
    val_dataset = val_loader.dataset

    # Find up to 16 frames where the model makes detections
    print("Searching for frames with detections...")
    frames_with_preds = find_frames_with_predictions(
        model, val_dataset, device, score_threshold=0.0, min_predictions=1
    )

    if frames_with_preds:
        visualize_predictions_grid(
            model, val_dataset, device, frames_with_preds,
            score_threshold=0.0,
            show_gt=True,
            save_path="images/predictions_grid.png",
        )
    else:
        print("No frames with predictions found at threshold 0.0.")

    #return best_checkpoint


if __name__ == "__main__":
    import argparse
    
    # parser = argparse.ArgumentParser(description='Train on SeeingThroughFog dataset')
    # parser.add_argument('--dataset', '-d', default=DATASET_ROOT, help='Dataset root directory')
    # parser.add_argument('--camera', '-c', default='left', choices=['left', 'right', 'left_lut', 'right_lut'], help='Camera to use')
    # parser.add_argument('--use-radar', action='store_true', help='Include radar data')
    # parser.add_argument('--project-radar', action='store_true', help='Project radar to camera coordinates')
    # parser.add_argument('--epochs', '-e', type=int, default=NUM_EPOCHS, help='Number of epochs')
    # parser.add_argument('--batch-size', '-b', type=int, default=BATCH_SIZE, help='Batch size')
    # parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    # parser.add_argument('--checkpoint-dir', default=None, help='Checkpoint directory')
    # parser.add_argument('--no-wandb', action='store_true', help='Disable W&B logging')
    # parser.add_argument('--resume', default=None, help='Resume from checkpoint')
    
    # args = parser.parse_args()
    
    # best_model = train(
    #     dataset_root=args.dataset,
    #     use_camera=args.camera,
    #     use_radar=args.use_radar,
    #     project_radar=args.project_radar,
    #     num_epochs=args.epochs,
    #     batch_size=args.batch_size,
    #     learning_rate=args.lr,
    #     checkpoint_dir=args.checkpoint_dir,
    #     use_wandb=not args.no_wandb,
    #     resume_from=args.resume,
    # )

    train( #best_model =
        dataset_root=DATASET_ROOT,
        use_camera='left_lut',
        use_radar=True,
        project_radar=True,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        checkpoint_dir=None,
        use_wandb=False,
        resume_from=None,
    )
    
    #print(f"Best model checkpoint: {best_model}")
