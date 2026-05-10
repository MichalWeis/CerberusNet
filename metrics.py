# File made with help from GitHub Copilot
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def _to_numpy(x):
    """Convert tensor or array to numpy array on CPU."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def box_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes using NumPy.
    Args:
        boxes1: array of shape (N, 4)
        boxes2: array of shape (M, 4)
    Returns:
        iou: array of shape (N, M)
    """
    boxes1 = _to_numpy(boxes1)
    boxes2 = _to_numpy(boxes2)
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = np.clip(rb - lt, 0, None)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou

def compute_ap(recalls, precisions):
    """
    Compute Average Precision (AP) from recall and precision curves.
    Args:
        recalls: List of recall values
        precisions: List of precision values
    Returns:
        ap: Average Precision score
    """

    ap = 0.0
    for t in np.linspace(0, 1, 11):
        precisions_above_recall = precisions[recalls >= t]
        if len(precisions_above_recall) == 0:
            p = 0
        else:
            p = precisions_above_recall.max()
        ap += p / 11.0
    return ap

def compute_ap_coco(recalls, precisions):
    """
    Compute Average Precision using all-point interpolation (COCO style).
    """
    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[0], precisions, [0]])

    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Find points where recall changes
    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    return ap

class Metrics:
    """
    Computes object detection metrics including mAP at various IoU thresholds.
    """

    def __init__(self, num_classes, class_names=None, iou_thresholds=None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.iou_thresholds = iou_thresholds
        self.reset()

    def reset(self):
        """
        Reset all accumulated predictions and ground truths.
        """
        self.predictions = [] # List of (boxes, scores, labels) per image
        self.ground_truths = [] # List of (boxes, labels) per image
        self.sample_conditions = []  # List of condition dicts per image

    def update(self, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels,
               conditions=None):
        """
        Add predictions and ground truths for one image.
        Args:
            pred_boxes: Tensor/array of shape (N, 4)
            pred_scores: Tensor/array of shape (N,)
            pred_labels: Tensor/array of shape (N,)
            gt_boxes: Tensor/array of shape (M, 4)
            gt_labels: Tensor/array of shape (M,)
            conditions: Optional dict of condition tags for this sample,
                e.g. {"weather": "clear", "daytime": "night", ...}
        """

        self.predictions.append((
            _to_numpy(pred_boxes),
            _to_numpy(pred_scores),
            _to_numpy(pred_labels)
        ))
        self.ground_truths.append((
            _to_numpy(gt_boxes),
            _to_numpy(gt_labels)
        ))
        self.sample_conditions.append(conditions)

    def _to_class_index(self, label):
        """
        Convert detector/dataset label id to 0-based class index.

        Expected convention:
        - 0: background
        - 1..num_classes: foreground classes
        """
        label = int(label)
        if label <= 0 or label > self.num_classes:
            return None
        return label - 1
    
    def _pr_for_class(self, cls_id, class_preds, class_gts, iou_threshold):
        """
        Compute precision-recall arrays for one class.
        Args:
            cls_id: 0-based class index
            class_preds: dict[class_id] -> list[(img_id, score, box)]
            class_gts: dict[class_id] -> list[(img_id, box)]
            iou_threshold: float
        Returns:
            (recalls, precisions): np.ndarray, np.ndarray
        """
        preds = class_preds.get(cls_id, [])
        gts = class_gts.get(cls_id, [])
        num_gt = len(gts)

        if num_gt == 0:
            return np.array([]), np.array([])
        if len(preds) == 0:
            return np.array([0.0]), np.array([0.0])

        preds = sorted(preds, key=lambda x: -x[1])  # score desc

        # GT boxes by image
        img_gt_boxes = defaultdict(list)
        for img_id, box in gts:
            img_gt_boxes[img_id].append(box)

        # matched flags per image
        img_gt_matched = {
            img_id: np.zeros(len(boxes), dtype=bool)
            for img_id, boxes in img_gt_boxes.items()
        }

        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)

        for i, (img_id, _, pred_box) in enumerate(preds):
            gt_boxes_img = img_gt_boxes.get(img_id, None)
            if gt_boxes_img is None or len(gt_boxes_img) == 0:
                fp[i] = 1.0
                continue

            pred_arr = np.asarray(pred_box, dtype=np.float32).reshape(1, 4)
            gt_arr = np.asarray(gt_boxes_img, dtype=np.float32)
            ious = box_iou(pred_arr, gt_arr)[0]

            best_idx = int(np.argmax(ious))
            if ious[best_idx] >= iou_threshold and not img_gt_matched[img_id][best_idx]:
                tp[i] = 1.0
                img_gt_matched[img_id][best_idx] = True
            else:
                fp[i] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        recalls = tp_cum / max(num_gt, 1)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        return recalls, precisions

    def compute_ap_per_class(self, iou_thresholds=0.5):
        """
        Compute AP for each class at a given IoU threshold.
        Returns:
            dict: {class_id: AP}
        """
        
        # Collect all predictions and ground truths per class
        class_predictions = defaultdict(list) # class_id -> [(img_id, score, box)]
        class_gt = defaultdict(list) # class_id -> [(img_id, box)]

        for img_id, ((pred_boxes, pred_scores, pred_labels), (gt_boxes, gt_labels)) in enumerate(
            zip(self.predictions, self.ground_truths)):

            # Add predictions
            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                class_idx = self._to_class_index(label)
                if class_idx is None:
                    continue
                class_predictions[class_idx].append((img_id, float(score), box))
            
            # Add ground truths
            for box, label in zip(gt_boxes, gt_labels):
                class_idx = self._to_class_index(label)
                if class_idx is None:
                    continue
                class_gt[class_idx].append((img_id, box))
        
        ap_per_class = {}

        for class_id in range(self.num_classes):
            preds = class_predictions[class_id]
            gts = class_gt[class_id]

            if len(gts) == 0:
                ap_per_class[class_id] = None # No ground truth for this class
                continue

            if len(preds) == 0:
                ap_per_class[class_id] = 0.0 # No predictions for this class
                continue

            # Sort predictions by score (descending)
            preds.sort(key=lambda x: x[1], reverse=True)

            # Track which GT boxes have been matched
            gt_matched = defaultdict(set)  # img_id -> set of matched gt indices
            
            tp = np.zeros(len(preds))
            fp = np.zeros(len(preds))

            for pred_idx, (img_id, score, pred_box) in enumerate(preds):
                # Get GT boxes for this image and class
                img_gts = [(i, box) for i, (gid, box) in enumerate(gts) if gid == img_id]
                
                if len(img_gts) == 0:
                    fp[pred_idx] = 1
                    continue

                # Find best matching GT box
                best_iou = 0
                best_gt_idx = -1

                pred_box_arr = np.asarray(pred_box).reshape(1, -1)

                for gt_local_idx, (gt_global_idx, gt_box) in enumerate(img_gts):
                    gt_box_arr = np.asarray(gt_box).reshape(1, -1)
                    iou = box_iou(pred_box_arr, gt_box_arr)[0, 0]
                    
                    if iou > best_iou and gt_global_idx not in gt_matched[img_id]:
                        best_iou = iou
                        best_gt_idx = gt_global_idx
                
                if best_iou >= iou_thresholds and best_gt_idx != -1:
                    tp[pred_idx] = 1
                    gt_matched[img_id].add(best_gt_idx)
                else:
                    fp[pred_idx] = 1
            
            # Compute precision and recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            recalls = tp_cumsum / len(gts)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
            
            ap_per_class[class_id] = compute_ap_coco(recalls, precisions)
        
        return ap_per_class
    
    def compute_precision_recall(self, iou_threshold=0.5):
        """
        Compute overall and per-class precision and recall at a given IoU threshold.
        Args:
            iou_threshold: IoU threshold for matching predictions to ground truths
        Returns:
            dict: Dictionary containing precision and recall metrics
        """
        results = {}
        
        # Per-class statistics
        class_tp = defaultdict(int)
        class_fp = defaultdict(int)
        class_fn = defaultdict(int)
        
        for (pred_boxes, pred_scores, pred_labels), (gt_boxes, gt_labels) in zip(
            self.predictions, self.ground_truths
        ):
            # Track matched GT boxes per class
            gt_matched = set()

            gt_class_indices = [self._to_class_index(lbl) for lbl in gt_labels]
            
            # Sort predictions by score (descending)
            if len(pred_scores) > 0:
                sorted_indices = np.argsort(pred_scores)[::-1]
                pred_boxes = pred_boxes[sorted_indices]
                pred_labels = pred_labels[sorted_indices]
            
            # Match predictions to ground truths
            for pred_idx, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
                pred_label_idx = self._to_class_index(pred_label)
                if pred_label_idx is None:
                    continue

                best_iou = 0
                best_gt_idx = -1
                
                pred_box_arr = np.asarray(pred_box).reshape(1, -1)
                
                for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                    gt_label_idx = gt_class_indices[gt_idx]
                    if gt_label_idx is None or gt_label_idx != pred_label_idx or gt_idx in gt_matched:
                        continue
                    
                    gt_box_arr = np.asarray(gt_box).reshape(1, -1)
                    iou = box_iou(pred_box_arr, gt_box_arr)[0, 0]
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_threshold and best_gt_idx != -1:
                    class_tp[pred_label_idx] += 1
                    gt_matched.add(best_gt_idx)
                else:
                    class_fp[pred_label_idx] += 1
            
            # Count false negatives (unmatched ground truths)
            for gt_idx, gt_label in enumerate(gt_labels):
                gt_label_idx = gt_class_indices[gt_idx]
                if gt_label_idx is not None and gt_idx not in gt_matched:
                    class_fn[gt_label_idx] += 1
        
        # Compute per-class precision and recall
        for class_id in range(self.num_classes):
            tp = class_tp[class_id]
            fp = class_fp[class_id]
            fn = class_fn[class_id]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            results[f"precision_{class_name}"] = precision
            results[f"recall_{class_name}"] = recall
            results[f"f1_{class_name}"] = f1
            results[f"tp_{class_name}"] = tp
            results[f"fp_{class_name}"] = fp
            results[f"fn_{class_name}"] = fn
        
        # Compute overall (micro-averaged) precision and recall
        total_tp = sum(class_tp.values())
        total_fp = sum(class_fp.values())
        total_fn = sum(class_fn.values())
        
        results["precision"] = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        results["recall"] = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        results["f1"] = (2 * results["precision"] * results["recall"] / 
                        (results["precision"] + results["recall"]) 
                        if (results["precision"] + results["recall"]) > 0 else 0.0)
        results["total_tp"] = total_tp
        results["total_fp"] = total_fp
        results["total_fn"] = total_fn
        
        return results
    
    def _build_full_confusion_matrix(self, iou_threshold=0.5):
        """
        Build full confusion matrix for object detection (includes Background).

        Args:
            iou_threshold: IoU threshold for matching predictions to ground truths
        Returns:
            tuple: (confusion_matrix, labels, fp_per_class, fn_per_class)
        """
        bg_idx = self.num_classes
        confusion_matrix = np.zeros((self.num_classes + 1, self.num_classes + 1), dtype=np.int32)
        fp_per_class = np.zeros(self.num_classes, dtype=np.int32)
        fn_per_class = np.zeros(self.num_classes, dtype=np.int32)
        labels = list(self.class_names) + ["Background"]

        for (pred_boxes, pred_scores, pred_labels), (gt_boxes, gt_labels) in zip(
            self.predictions, self.ground_truths
        ):
            gt_matched = set()
            gt_class_indices = [self._to_class_index(lbl) for lbl in gt_labels]

            if len(pred_scores) > 0:
                sorted_indices = np.argsort(pred_scores)[::-1]
                pred_boxes = pred_boxes[sorted_indices]
                pred_labels = pred_labels[sorted_indices]

            for pred_box, pred_label in zip(pred_boxes, pred_labels):
                pred_class_idx = self._to_class_index(pred_label)
                if pred_class_idx is None:
                    continue

                best_iou = 0.0
                best_gt_idx = -1
                best_gt_class_idx = -1

                pred_box_arr = np.asarray(pred_box).reshape(1, -1)

                for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                    if gt_idx in gt_matched:
                        continue
                    gt_class_idx = gt_class_indices[gt_idx]
                    if gt_class_idx is None:
                        continue

                    gt_box_arr = np.asarray(gt_box).reshape(1, -1)
                    iou = box_iou(pred_box_arr, gt_box_arr)[0, 0]

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                        best_gt_class_idx = gt_class_idx

                if best_iou >= iou_threshold and best_gt_idx != -1:
                    confusion_matrix[best_gt_class_idx, pred_class_idx] += 1
                    gt_matched.add(best_gt_idx)
                else:
                    confusion_matrix[bg_idx, pred_class_idx] += 1
                    fp_per_class[pred_class_idx] += 1

            for gt_idx, gt_label in enumerate(gt_labels):
                if gt_idx in gt_matched:
                    continue
                gt_class_idx = gt_class_indices[gt_idx]
                if gt_class_idx is not None:
                    confusion_matrix[gt_class_idx, bg_idx] += 1
                    fn_per_class[gt_class_idx] += 1

        return confusion_matrix, labels, fp_per_class, fn_per_class

    def _filter_confusion_matrix(
        self,
        confusion_matrix,
        labels,
        fp_per_class,
        fn_per_class,
        exclude_labels=None,
        exclude_background=True,
    ):
        """
        Remove selected rows/columns from the confusion matrix and align FP/FN arrays.
        """
        if exclude_labels is None:
            exclude_labels = {"dont care", "obstacle", "train"}

        excluded_names = {str(name).strip().lower() for name in exclude_labels}
        if exclude_background:
            excluded_names.add("background")

        keep_indices = [
            idx for idx, name in enumerate(labels)
            if str(name).strip().lower() not in excluded_names
        ]

        filtered_matrix = confusion_matrix[np.ix_(keep_indices, keep_indices)]
        filtered_labels = [labels[idx] for idx in keep_indices]

        keep_class_indices = [idx for idx in keep_indices if idx < self.num_classes]
        filtered_fp = fp_per_class[keep_class_indices] if len(keep_class_indices) > 0 else np.array([], dtype=np.int32)
        filtered_fn = fn_per_class[keep_class_indices] if len(keep_class_indices) > 0 else np.array([], dtype=np.int32)

        return filtered_matrix, filtered_labels, filtered_fp, filtered_fn

    def compute_confusion_matrix(self, iou_threshold=0.5, exclude_labels=None, exclude_background=True):
        """
        Compute confusion matrix for object detection.

        By default, removes rows/columns for background, don't-care, obstacle, and train.

        Args:
            iou_threshold: IoU threshold for matching predictions to ground truths
            exclude_labels: Iterable of class names to remove (case-insensitive)
            exclude_background: Whether to remove the Background row/column
        Returns:
            tuple: (confusion_matrix, labels, fp_per_class, fn_per_class)
        """
        full_matrix, full_labels, full_fp, full_fn = self._build_full_confusion_matrix(iou_threshold)
        return self._filter_confusion_matrix(
            full_matrix,
            full_labels,
            full_fp,
            full_fn,
            exclude_labels=exclude_labels,
            exclude_background=exclude_background,
        )

    
    def get_tp_fp_fn_tn(self, iou_threshold=0.5):
        """
        Get TP, FP, FN, TN counts from the confusion matrix.
        
        For object detection:
        - TP: Correct detections (diagonal elements)
        - FP: False detections (predictions with no GT match) + wrong class predictions
        - FN: Missed detections (GTs with no prediction match) + wrong class predictions
        - TN: Not directly applicable in object detection
        
        Args:
            iou_threshold: IoU threshold for matching
        Returns:
            dict: Dictionary with TP, FP, FN counts (overall and per-class)
        """
        confusion_matrix, labels, fp_per_class, fn_per_class = self._build_full_confusion_matrix(iou_threshold)
        bg_idx = self.num_classes

        results = {
            "confusion_matrix": confusion_matrix,
            "labels": labels,
            "fp_per_class": fp_per_class,
            "fn_per_class": fn_per_class,
            "per_class": {}
        }

        total_tp = 0
        total_fp = 0
        total_fn = 0

        for class_id in range(self.num_classes):
            class_name = self.class_names[class_id]
            tp = int(confusion_matrix[class_id, class_id])

            fp = int(confusion_matrix[bg_idx, class_id])
            for other_class in range(self.num_classes):
                if other_class != class_id:
                    fp += int(confusion_matrix[other_class, class_id])

            fn = int(confusion_matrix[class_id, bg_idx])
            for other_class in range(self.num_classes):
                if other_class != class_id:
                    fn += int(confusion_matrix[class_id, other_class])

            results["per_class"][class_name] = {"TP": tp, "FP": fp, "FN": fn}
            total_tp += tp
            total_fp += fp
            total_fn += fn

        class_matrix = confusion_matrix[:self.num_classes, :self.num_classes]
        off_diag = int(class_matrix.sum() - np.trace(class_matrix))
        results["total_TP"] = total_tp
        results["total_FP"] = int(confusion_matrix[bg_idx, :self.num_classes].sum()) + off_diag
        results["total_FN"] = int(confusion_matrix[:self.num_classes, bg_idx].sum()) + off_diag

        return results
    
    def plot_confusion_matrix(
        self,
        iou_threshold=0.5,
        normalize=False,
        save_path=None,
        figsize=(10, 8),
        exclude_labels=None,
        exclude_background=True,
    ):
        """
        Plot the confusion matrix as a heatmap.
        
        Args:
            iou_threshold: IoU threshold for matching predictions to ground truths
            normalize: If True, normalize by ground truth (row-wise)
            save_path: Path to save the figure (optional)
            figsize: Figure size tuple
            exclude_labels: Iterable of class names to remove (case-insensitive)
            exclude_background: If True, remove Background row and column from the plot
        Returns:
            matplotlib.figure.Figure: The confusion matrix figure
        """
        confusion_matrix, labels, fp_per_class, fn_per_class = self.compute_confusion_matrix(
            iou_threshold=iou_threshold,
            exclude_labels=exclude_labels,
            exclude_background=exclude_background,
        )
        
        if normalize:
            # Normalize by row (ground truth)
            row_sums = confusion_matrix.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
            confusion_matrix = confusion_matrix.astype(np.float32) / row_sums
            fmt = '.2f'
            title = f'Normalized Confusion Matrix (IoU={iou_threshold})'
        else:
            fmt = 'd'
            title = f'Confusion Matrix (IoU={iou_threshold})'
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )
        
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('Ground Truth Class')
        ax.set_title(title)
        
        # Add explanatory text
        total_fp = int(fp_per_class.sum())
        total_fn = int(fn_per_class.sum())
        has_background = any(str(label).strip().lower() == "background" for label in labels)
        if has_background:
            fig.text(0.5, -0.02, 
                     f'Background row = FP (no GT match) | Background col = FN (missed GT) | Total FP: {total_fp} | Total FN: {total_fn}',
                     ha='center', fontsize=9, style='italic')
        else:
            fig.text(0.5, -0.02, 
                     f'Rows: GT class | Columns: Predicted class | Classes removed from matrix | Remaining-class FP: {total_fp} | FN: {total_fn}',
                     ha='center', fontsize=9, style='italic')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def compute(self):
        """
        Compute all metrics.
        Returns:
            dict: Dictionary containing all computed metrics
        """
        metrics = {}
        
        # Compute mAP at different IoU thresholds
        for iou_thresh in tqdm(self.iou_thresholds, desc="Computing mAP"):
            ap_per_class = self.compute_ap_per_class(iou_thresholds=iou_thresh)
            
            # Store per-class AP
            for class_id, ap in ap_per_class.items():
                if ap is not None:
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    metrics[f"AP@{iou_thresh:.2f}_{class_name}"] = ap
            
            # Compute mAP (mean over classes with GT)
            valid_aps = [ap for ap in ap_per_class.values() if ap is not None]
            if valid_aps:
                metrics[f"mAP@{iou_thresh:.2f}"] = np.mean(valid_aps)
            else:
                metrics[f"mAP@{iou_thresh:.2f}"] = 0.0
        
        # Compute mAP@[0.5:0.95] (COCO-style)
        coco_thresholds = np.arange(0.5, 1.0, 0.05)
        all_aps = []
        for iou_thresh in tqdm(coco_thresholds, desc="Computing COCO-style mAP"):
            ap_per_class = self.compute_ap_per_class(iou_thresholds=iou_thresh)
            valid_aps = [ap for ap in ap_per_class.values() if ap is not None]
            if valid_aps:
                all_aps.append(np.mean(valid_aps))
        
        if all_aps:
            metrics["mAP@[0.5:0.95]"] = np.mean(all_aps)
        else:
            metrics["mAP@[0.5:0.95]"] = 0.0
        
        # Count statistics
        total_gt = sum(len(gt[0]) for gt in self.ground_truths)
        total_pred = sum(len(pred[0]) for pred in self.predictions)
        metrics["total_ground_truths"] = total_gt
        metrics["total_predictions"] = total_pred

        # Compute precision and recall at IoU=0.5
        precision_recall = self.compute_precision_recall(iou_threshold=0.5)
        metrics.update(precision_recall)
        
        return metrics

    def compute_per_condition(self, condition_key: str, iou_threshold: float = 0.5):
        """
        Split accumulated samples by a condition dimension and compute
        mAP@iou_threshold for each condition value.

        Args:
            condition_key: Which condition to group by, e.g. "weather",
                "daytime", "environment", "illumination", "infrastructure".
            iou_threshold: IoU threshold for AP computation.

        Returns:
            dict[str, dict]: mapping condition_value -> metrics dict.
            Each inner dict contains at least "mAP@<threshold>", "num_samples",
            and per-class AP keys.
        """
        # Group sample indices by condition value
        groups = defaultdict(list)
        for idx, cond in enumerate(self.sample_conditions):
            if cond is None:
                val = "unknown"
            else:
                val = cond.get(condition_key, "unknown")
            groups[val].append(idx)

        results = {}
        for cond_val, indices in sorted(groups.items()):
            sub_metrics = Metrics(
                num_classes=self.num_classes,
                class_names=self.class_names,
                iou_thresholds=self.iou_thresholds,
            )
            for idx in indices:
                pred_boxes, pred_scores, pred_labels = self.predictions[idx]
                gt_boxes, gt_labels = self.ground_truths[idx]
                sub_metrics.predictions.append((pred_boxes, pred_scores, pred_labels))
                sub_metrics.ground_truths.append((gt_boxes, gt_labels))
                sub_metrics.sample_conditions.append(self.sample_conditions[idx])

            ap_per_class = sub_metrics.compute_ap_per_class(iou_thresholds=iou_threshold)
            valid_aps = [ap for ap in ap_per_class.values() if ap is not None]
            mean_ap = float(np.mean(valid_aps)) if valid_aps else 0.0

            entry = {
                f"mAP@{iou_threshold:.2f}": mean_ap,
                "num_samples": len(indices),
            }
            for class_id, ap in ap_per_class.items():
                if ap is not None:
                    cname = (self.class_names[class_id]
                             if class_id < len(self.class_names)
                             else f"class_{class_id}")
                    entry[f"AP@{iou_threshold:.2f}_{cname}"] = ap

            # Also compute precision/recall summary
            pr = sub_metrics.compute_precision_recall(iou_threshold=iou_threshold)
            entry["precision"] = pr.get("precision", 0.0)
            entry["recall"] = pr.get("recall", 0.0)
            entry["f1"] = pr.get("f1", 0.0)

            results[cond_val] = entry

        return results

    def print_summary(self, head_name: str = None):
        """
        Print a formatted summary of all metrics.
        """

        metrics = self.compute()
        
        title = "Object Detection Metrics Summary"
        if head_name:
            title += f"  [{head_name}]"

        print("\n" + "=" * 50)
        print(title)
        print("=" * 50)
        
        print(f"\nTotal Ground Truths: {metrics['total_ground_truths']}")
        print(f"Total Predictions: {metrics['total_predictions']}")
        
        print(f"\n--- mAP Metrics ---")
        print(f"mAP@0.50: {metrics.get('mAP@0.50', 0.0):.4f}")
        print(f"mAP@0.75: {metrics.get('mAP@0.75', 0.0):.4f}")
        print(f"mAP@[0.5:0.95]: {metrics.get('mAP@[0.5:0.95]', 0.0):.4f}")
        
        print(f"\n--- Overall Precision/Recall @IoU=0.50 ---")
        print(f"Precision: {metrics.get('precision', 0.0):.4f}")
        print(f"Recall: {metrics.get('recall', 0.0):.4f}")
        print(f"F1 Score: {metrics.get('f1', 0.0):.4f}")
        print(f"TP: {metrics.get('total_tp', 0)} | FP: {metrics.get('total_fp', 0)} | FN: {metrics.get('total_fn', 0)}")
        
        print("\n--- Per-class Metrics @IoU=0.50 ---")
        for class_id in range(self.num_classes):
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            ap = metrics.get(f"AP@0.50_{class_name}", None)
            precision = metrics.get(f"precision_{class_name}", 0.0)
            recall = metrics.get(f"recall_{class_name}", 0.0)
            f1 = metrics.get(f"f1_{class_name}", 0.0)
            tp = metrics.get(f"tp_{class_name}", 0)
            fp = metrics.get(f"fp_{class_name}", 0)
            fn = metrics.get(f"fn_{class_name}", 0)
            
            if ap is not None or (tp + fp + fn) > 0:
                ap_str = f"{ap:.4f}" if ap is not None else "N/A"
                print(f"  {class_name}:")
                print(f"    AP: {ap_str} | P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}")
                print(f"    TP: {tp} | FP: {fp} | FN: {fn}")

        print("=" * 50 + "\n")
        
        return metrics


# ---------------------------------------------------------------------------
# Auxiliary weather / daytime classification metrics
# ---------------------------------------------------------------------------

class AuxMetrics:
    """
    Accumulates confusion matrices for the auxiliary weather & daytime
    classifiers attached to the RGB midpoint features and computes overall,
    macro, and per-class accuracy.

    Typical usage:
        aux = AuxMetrics()
        for modalities, targets, batch_conditions in loader:
            _ = model(modalities)                      # triggers forward
            aux.update(model.last_aux_predictions,     # logits dict
                       batch_conditions)               # list/tuple of dicts
        results = aux.compute()                        # flat metric dict
        aux.print_summary()
        aux.plot_confusion_matrix("weather", save_path="aux_weather_cm.png")
    """

    # Ignore index used by build_aux_targets for unknown / OOV labels.
    IGNORE_INDEX = -100

    def __init__(self, weather_classes=None, daytime_classes=None):
        """
        Args:
            weather_classes: list of weather class names in label-index order.
                Defaults to model.WEATHER_CLASSES.
            daytime_classes: list of daytime class names in label-index order.
                Defaults to model.DAYTIME_CLASSES.
        """
        # Imported lazily so metrics.py doesn't eagerly depend on model.py.
        if weather_classes is None or daytime_classes is None:
            from model import WEATHER_CLASSES, DAYTIME_CLASSES
            weather_classes = weather_classes or WEATHER_CLASSES
            daytime_classes = daytime_classes or DAYTIME_CLASSES

        self.weather_classes = list(weather_classes)
        self.daytime_classes = list(daytime_classes)
        self.reset()

    def reset(self):
        """Zero out both confusion matrices and the ignored-sample counters."""
        nw = len(self.weather_classes)
        nd = len(self.daytime_classes)
        self.weather_cm = np.zeros((nw, nw), dtype=np.int64)
        self.daytime_cm = np.zeros((nd, nd), dtype=np.int64)
        self.weather_ignored = 0
        self.daytime_ignored = 0

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------

    def update(self, aux_pred, conditions):
        """
        Accumulate one batch's predictions. Safe no-op if ``aux_pred`` is
        ``None`` (e.g. if the model forward didn't run the aux head).

        Args:
            aux_pred: dict with keys 'weather_logits' (B, Cw) and
                'daytime_logits' (B, Cd) — as stashed on
                ``model.last_aux_predictions``.
            conditions: iterable of per-sample condition dicts (one per image)
                with string keys 'weather' and 'daytime'. Items that are
                missing, not dicts, or carry out-of-vocabulary labels are
                counted as *ignored* and excluded from the confusion matrix.
        """
        if aux_pred is None:
            return

        # Lazy import to avoid a hard dependency from metrics → model.
        from model import build_aux_targets

        device = aux_pred["weather_logits"].device
        w_tgt, d_tgt = build_aux_targets(list(conditions), device)

        w_pred = aux_pred["weather_logits"].argmax(dim=1).cpu().numpy()
        d_pred = aux_pred["daytime_logits"].argmax(dim=1).cpu().numpy()
        w_tgt_np = w_tgt.cpu().numpy()
        d_tgt_np = d_tgt.cpu().numpy()

        # Weather: count ignored, increment CM only on valid samples.
        w_valid = w_tgt_np != self.IGNORE_INDEX
        self.weather_ignored += int((~w_valid).sum())
        np.add.at(self.weather_cm, (w_tgt_np[w_valid], w_pred[w_valid]), 1)

        # Daytime
        d_valid = d_tgt_np != self.IGNORE_INDEX
        self.daytime_ignored += int((~d_valid).sum())
        np.add.at(self.daytime_cm, (d_tgt_np[d_valid], d_pred[d_valid]), 1)

    # ------------------------------------------------------------------
    # compute
    # ------------------------------------------------------------------

    @staticmethod
    def _cm_to_metrics(cm: np.ndarray, class_names, prefix: str, ignored: int):
        """
        Turn a confusion matrix into a flat metrics dict.

        Keys produced (for prefix='weather'):
            weather/acc             overall top-1 accuracy
            weather/macro_acc       unweighted mean of per-class recall
            weather/total           number of valid samples
            weather/ignored         samples dropped due to unknown label
            weather/<cls>/acc       per-class recall (diag / row-sum)
            weather/<cls>/support   per-class sample count (row-sum)
        """
        total = int(cm.sum())
        correct = int(np.trace(cm))
        overall = correct / total if total > 0 else 0.0

        out = {
            f"{prefix}/acc": overall,
            f"{prefix}/total": total,
            f"{prefix}/ignored": ignored,
        }

        per_class_accs = []
        for i, name in enumerate(class_names):
            support = int(cm[i].sum())
            acc = (cm[i, i] / support) if support > 0 else 0.0
            out[f"{prefix}/{name}/acc"] = float(acc)
            out[f"{prefix}/{name}/support"] = support
            if support > 0:
                per_class_accs.append(acc)

        out[f"{prefix}/macro_acc"] = (
            float(np.mean(per_class_accs)) if per_class_accs else 0.0
        )
        return out

    def compute(self):
        """
        Return a flat dict of metrics, keyed like ``weather/acc``,
        ``weather/clear/acc``, ``daytime/day/support``, etc.

        W&B-friendly: every key uses ``/`` as a group separator so the UI
        nests the metrics under ``weather`` and ``daytime`` panels.
        """
        out = {}
        out.update(self._cm_to_metrics(
            self.weather_cm, self.weather_classes, "weather", self.weather_ignored
        ))
        out.update(self._cm_to_metrics(
            self.daytime_cm, self.daytime_classes, "daytime", self.daytime_ignored
        ))
        return out

    # ------------------------------------------------------------------
    # presentation
    # ------------------------------------------------------------------

    def print_summary(self, head_name: str = None):
        """Print a formatted console summary of the aux classifier metrics."""
        metrics = self.compute()

        title = "Auxiliary Classifier Metrics (weather / daytime)"
        if head_name:
            title += f"  [{head_name}]"

        print("\n" + "=" * 50)
        print(title)
        print("=" * 50)

        for prefix, class_names in (
            ("weather", self.weather_classes),
            ("daytime", self.daytime_classes),
        ):
            print(
                f"\n--- {prefix} ---  "
                f"acc={metrics[f'{prefix}/acc']:.4f}  "
                f"macro={metrics[f'{prefix}/macro_acc']:.4f}  "
                f"n={metrics[f'{prefix}/total']}  "
                f"ignored={metrics[f'{prefix}/ignored']}"
            )
            for name in class_names:
                acc = metrics[f"{prefix}/{name}/acc"]
                support = metrics[f"{prefix}/{name}/support"]
                print(f"  {name:>12s}: acc={acc:.4f}  support={support}")

        print("=" * 50 + "\n")

        return metrics

    def plot_confusion_matrix(
        self,
        task: str,
        normalize: bool = False,
        save_path=None,
        figsize=(6, 5),
    ):
        """
        Plot the confusion matrix for one aux task ('weather' or 'daytime').

        Args:
            task: which task to plot — 'weather' or 'daytime'.
            normalize: if True, normalize rows (ground truth) to proportions.
            save_path: optional path to save the figure as PNG.
            figsize: figure size tuple.
        Returns:
            matplotlib.figure.Figure
        """
        if task == "weather":
            cm, labels = self.weather_cm, self.weather_classes
        elif task == "daytime":
            cm, labels = self.daytime_cm, self.daytime_classes
        else:
            raise ValueError(f"task must be 'weather' or 'daytime', got {task!r}")

        cm_display = cm.astype(np.float64)
        if normalize:
            row_sums = cm_display.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            cm_display = cm_display / row_sums
            fmt = ".2f"
            title = f"Aux {task} — Normalized Confusion Matrix"
        else:
            fmt = "d"
            cm_display = cm  # show raw ints
            title = f"Aux {task} — Confusion Matrix"

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm_display,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar_kws={"label": "Proportion" if normalize else "Count"},
        )
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Ground Truth Class")
        ax.set_title(title)

        ignored = self.weather_ignored if task == "weather" else self.daytime_ignored
        fig.text(
            0.5, -0.02,
            f"Rows: GT class | Columns: Predicted class | Ignored (unknown label): {ignored}",
            ha="center", fontsize=9, style="italic",
        )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Aux {task} confusion matrix saved to {save_path}")

        return fig