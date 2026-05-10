import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from collections import OrderedDict
from typing import Dict, Optional, Tuple, List

# FPN output level names (with LastLevelMaxPool extra level)
_FPN_KEYS = ['0', '1', '2', '3', 'pool']
# ROI pooler only uses the main FPN levels, not 'pool'
_ROI_KEYS = ['0', '1', '2', '3']

# ---------------------------------------------------------------------------
# Auxiliary weather / daytime classification classes
# ---------------------------------------------------------------------------
WEATHER_CLASSES = ["clear", "rain", "snow", "fog"]
DAYTIME_CLASSES = ["day", "night"]

WEATHER_TO_IDX = {c: i for i, c in enumerate(WEATHER_CLASSES)}
DAYTIME_TO_IDX = {c: i for i, c in enumerate(DAYTIME_CLASSES)}
# Sentinel for "unknown" / missing labels — cross_entropy ignores -100 by default.
_AUX_IGNORE_INDEX = -100

# ---------------------------------------------------------------------------
# Detection-head routing table (paper Table 5)
# ---------------------------------------------------------------------------
CONDITION_TO_HEAD: Dict[Tuple[str, str], str] = {
    # Day
    ("day",   "clear"): "all",
    ("day",   "fog"):   "all",
    ("day",   "rain"):  "rgb_lidar",
    ("day",   "snow"):  "all",
    # Night
    ("night", "clear"): "rgb_lidar",
    ("night", "fog"):   "all",
    ("night", "rain"):  "rgb_lidar",
    ("night", "snow"):  "all",
}
# Fallback head
_DEFAULT_ROUTED_HEAD = "all"

def route_heads_from_aux(aux_out: Dict[str, torch.Tensor]) -> List[str]:
    """
    Per-sample head selection driven by the auxiliary classifier.

    Args:
        aux_out: dict with 'weather_logits' (B, |W|) and 'daytime_logits'
                 (B, |D|), as returned by WeatherDaytimeHead.

    Returns:
        List of length B; entry i is the head name to use for sample i,
        derived by argmax-ing the aux logits and looking up CONDITION_TO_HEAD.
    """
    w_pred = aux_out["weather_logits"].argmax(dim=1).tolist()
    d_pred = aux_out["daytime_logits"].argmax(dim=1).tolist()

    routes: List[str] = []
    for w_idx, d_idx in zip(w_pred, d_pred):
        weather = WEATHER_CLASSES[w_idx]
        daytime = DAYTIME_CLASSES[d_idx]
        routes.append(CONDITION_TO_HEAD.get((daytime, weather), _DEFAULT_ROUTED_HEAD))
    return routes
 

# ---------------------------------------------------------------------------
# Weighted cross-entropy patch for ROI classification
# ---------------------------------------------------------------------------
from torchvision.ops import sigmoid_focal_loss
import torchvision.models.detection.roi_heads as _roi_heads_module

_original_fastrcnn_loss = _roi_heads_module.fastrcnn_loss
 
#                          bg   PassCar  LargeVeh  RidableVeh  Pedestrian
_CE_WEIGHTS = torch.tensor([1.0,   1.0,     1.4,      1.5,        1.0])
 
 
def _weighted_fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    """
    Drop-in replacement for torchvision's fastrcnn_loss.
    Uses weighted cross-entropy to handle class imbalance,
    keeps smooth-L1 for box regression.
    """
    labels = torch.cat(labels, dim=0)
 
    w = _CE_WEIGHTS.to(class_logits.device)
    classification_loss = F.cross_entropy(class_logits, labels, weight=w)
 
    # --- Box regression loss (unchanged) ---
    sampled_pos_inds = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds]
    N, num_classes = class_logits.shape
 
    box_regression = box_regression.reshape(N, num_classes, 4)
    box_regression = box_regression[sampled_pos_inds, labels_pos]
    regression_targets = torch.cat(regression_targets, dim=0)
    regression_targets = regression_targets[sampled_pos_inds]
 
    box_loss = F.smooth_l1_loss(
        box_regression, regression_targets, beta=1.0 / 9, reduction="sum"
    ) / max(labels.numel(), 1)
 
    return classification_loss, box_loss
 
 
# Apply the weighted loss by default
#_roi_heads_module.fastrcnn_loss = _weighted_fastrcnn_loss


# ---------------------------------------------------------------------------
# Focal-loss patch for ROI classification
# ---------------------------------------------------------------------------
 
# Save original so we can restore if needed
#_original_fastrcnn_loss = _roi_heads_module.fastrcnn_loss
 
_FOCAL_ALPHA = 0.25
_FOCAL_GAMMA = 2.0
 
 
def _focal_fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    """
    Drop-in replacement for torchvision's fastrcnn_loss.
    Uses sigmoid focal loss for the classifier to handle class imbalance,
    keeps smooth-L1 for box regression.
    """

    # --- Focal classification loss ---
    labels = torch.cat(labels, dim=0)
    num_classes = class_logits.shape[1]
    targets_one_hot = torch.zeros_like(class_logits)
    targets_one_hot.scatter_(1, labels.unsqueeze(1), 1)
 
    classification_loss = sigmoid_focal_loss(
        class_logits, targets_one_hot,
        alpha=_FOCAL_ALPHA, gamma=_FOCAL_GAMMA, reduction="mean",
    )
 
    # --- Box regression loss (unchanged) ---
    sampled_pos_inds = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds]
    N, num_classes = class_logits.shape
 
    box_regression = box_regression.reshape(N, num_classes, 4)
    box_regression = box_regression[sampled_pos_inds, labels_pos]
    regression_targets = torch.cat(regression_targets, dim=0)
    regression_targets = regression_targets[sampled_pos_inds]
 
    box_loss = F.smooth_l1_loss(
        box_regression, regression_targets, beta=1.0 / 9, reduction="sum"
    ) / max(labels.numel(), 1)
 
    return classification_loss, box_loss
 
 
def enable_focal_loss(alpha=0.25, gamma=2.0):
    """Monkey-patch torchvision's ROI head loss with focal loss."""
    global _FOCAL_ALPHA, _FOCAL_GAMMA
    _FOCAL_ALPHA = alpha
    _FOCAL_GAMMA = gamma
    _roi_heads_module.fastrcnn_loss = _focal_fastrcnn_loss
 
 
def disable_focal_loss():
    """Restore the original cross-entropy ROI head loss."""
    _roi_heads_module.fastrcnn_loss = _original_fastrcnn_loss


def multiclass_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Multi-class focal loss for single-label classification.

    Equivalent to (optionally-weighted) cross-entropy scaled by a modulating
    factor (1 - p_t)**gamma, where p_t is the softmax probability the model
    assigned to the correct class. This down-weights easy examples (high p_t,
    like the majority "clear" frames) so minority classes contribute a larger
    share of the gradient.

    Args:
        logits:        (N, C) raw class scores.
        target:        (N,)   int64 class indices; entries equal to
                              ``ignore_index`` are skipped entirely.
        alpha:         Optional (C,) per-class weighting tensor. When class
                       frequencies are imbalanced this should come from
                       1 / sqrt(count) (smoothed inverse frequency).
                       ``None`` = uniform.
        gamma:         Focusing parameter. 0 → weighted CE. Typical 1.0–5.0,
                       with 2.0 the paper default.
        ignore_index:  Targets equal to this are dropped before loss.
        reduction:     'mean' (default), 'sum', or 'none'.

    Returns:
        Scalar tensor (or (N_valid,) if reduction='none'). Zero on same device
        as logits when no samples are valid.
    """
    valid = target != ignore_index
    if not valid.any():
        return torch.zeros((), device=logits.device, dtype=logits.dtype)

    logits = logits[valid]
    target = target[valid]

    # log-softmax is the numerically stable way to get both log p and p_t.
    log_probs = F.log_softmax(logits, dim=-1)                    # (N, C)
    log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1) # (N,)
    pt = log_pt.exp()                                            # (N,)

    focal_weight = (1.0 - pt).pow(gamma)                         # (N,)
    loss = -focal_weight * log_pt                                # (N,)

    if alpha is not None:
        # Gather per-sample alpha from the per-class alpha vector.
        at = alpha.to(logits.device).gather(0, target)           # (N,)
        loss = loss * at

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
 
 
# Enable focal loss by default
#enable_focal_loss()
#disable_focal_loss()


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ModalityStem(nn.Module):
    """
    Per-modality input layer: conv7x7 → BatchNorm → ReLU → MaxPool.
    Maps any number of input channels to 64, matching ResNet layer1 input.
    Optionally initialized from pretrained ResNet conv1 weights.
    """

    def __init__(self, in_channels: int, pretrained_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if pretrained_weight is not None:
            with torch.no_grad():
                ch = min(in_channels, pretrained_weight.shape[1])
                self.conv.weight[:, :ch] = pretrained_weight[:, :ch]
                if in_channels > pretrained_weight.shape[1]:
                    mean_w = pretrained_weight.mean(dim=1, keepdim=True)
                    for c in range(pretrained_weight.shape[1], in_channels):
                        self.conv.weight[:, c : c + 1] = mean_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool(self.relu(self.norm(self.conv(x))))


class SharedEncoder(nn.Module):
    """
    Shared backbone: ResNet-18 layer1-4 + FPN (256ch output per level).
    Called once per modality with the same weights — gradients accumulate.
    """

    def __init__(self):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.layer1 = resnet.layer1  # 64 → 64
        self.layer2 = resnet.layer2  # 64 → 128
        self.layer3 = resnet.layer3  # 128 → 256
        self.layer4 = resnet.layer4  # 256 → 512

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[64, 128, 256, 512],
            out_channels=256,
            extra_blocks=LastLevelMaxPool(),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 64, H/4, W/4) — output of a ModalityStem.
        Returns:
            OrderedDict with keys '0','1','2','3','pool', each (B, 256, …).
        """
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return self.fpn(OrderedDict([("0", c1), ("1", c2), ("2", c3), ("3", c4)]))

    def forward_with_mid(
        self, x: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Same as ``forward`` but also returns the ResNet-18 midpoint features
        (``c2``, the output of layer2) which are fed to auxiliary heads.

        Args:
            x: (B, 64, H/4, W/4) — output of a ModalityStem.
        Returns:
            fpn_features: OrderedDict keyed '0','1','2','3','pool' (each 256 ch).
            c2:           (B, 128, H/8, W/8) — ResNet-18 midpoint (layer2 output).
        """
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        #print(f"[DEBUG backbone] layer3 output: {c3.shape}, layer4 output: {c4.shape}")
        fpn = self.fpn(OrderedDict([("0", c1), ("1", c2), ("2", c3), ("3", c4)]))
        return fpn, c2


class WeatherDaytimeHead(nn.Module):
    """
    Auxiliary classifier over RGB mid-level features.

    Consumes the output of ResNet-18 ``layer2`` (the midpoint of the backbone —
    128 channels at H/8 × W/8) and runs it through a dedicated convolutional
    trunk that is NOT shared with the main detection encoder. Features are then
    globally pooled and fed to two independent linear classifiers:

        * ``weather_logits`` — (B, 4): clear, rain, snow, fog
        * ``daytime_logits`` — (B, 2): day, night

    The conv trunk mirrors the "halving spatial / doubling channels" pattern
    that ResNet layer3/layer4 would have applied, so the final feature map
    matches the scale of the main backbone output (~H/32 × W/32) before pooling.
    """

    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 256,
        num_weather_classes: int = len(WEATHER_CLASSES),
        num_daytime_classes: int = len(DAYTIME_CLASSES),
        dropout: float = 0.2,
    ):
        super().__init__()

        # Dedicated conv trunk: 128 → 256 → 256 → 256, with two stride-2 blocks
        # to bring H/8 down to H/32 before global pooling.
        self.conv_trunk = nn.Sequential(
            # block 1: downsample H/8 → H/16, 128 → 256
            nn.Conv2d(in_channels, hidden_channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, hidden_channels),
            nn.ReLU(inplace=True),
            # block 2: refine, no downsample
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, hidden_channels),
            nn.ReLU(inplace=True),
            # block 3: downsample H/16 → H/32
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)  # (B, C, 1, 1)

        # Two independent FC classifiers, each with a small bottleneck.
        self.weather_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_weather_classes),
        )
        self.daytime_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_daytime_classes),
        )

    def forward(self, rgb_mid: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            rgb_mid: (B, 128, H/8, W/8) — RGB features at ResNet-18 midpoint.
        Returns:
            dict with 'weather_logits' (B, 5) and 'daytime_logits' (B, 2).
        """
        trunk_out = self.conv_trunk(rgb_mid)
        #print(f"[DEBUG head] conv_trunk output: {trunk_out.shape}")
        h = self.pool(trunk_out)  # (B, C, 1, 1)
        return {
            "weather_logits": self.weather_classifier(h),
            "daytime_logits": self.daytime_classifier(h),
        }


def build_aux_targets(
    conditions: Optional[List[Optional[dict]]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a list of per-sample condition dicts (as produced by the dataset's
    ``load_labeltool_label``) into tensors of class indices suitable for
    ``F.cross_entropy(..., ignore_index=_AUX_IGNORE_INDEX)``.

    Unknown / missing / out-of-vocabulary labels are mapped to the ignore index
    so they contribute no gradient.
    """
    weather_idx: List[int] = []
    daytime_idx: List[int] = []
    for c in conditions:
        w = c.get("weather", "unknown") if isinstance(c, dict) else "unknown"
        d = c.get("daytime", "unknown") if isinstance(c, dict) else "unknown"
        if w in ("light_fog", "dense_fog"):
            w = "fog"
        weather_idx.append(WEATHER_TO_IDX.get(w, _AUX_IGNORE_INDEX))
        daytime_idx.append(DAYTIME_TO_IDX.get(d, _AUX_IGNORE_INDEX))
    return (
        torch.tensor(weather_idx, dtype=torch.long, device=device),
        torch.tensor(daytime_idx, dtype=torch.long, device=device),
    )


class DetectionHead(nn.Module):
    """
    One detection head: channel reduction (1x1 conv) + RPN + ROI heads.
    Each fusion combination gets its own instance.
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()

        out_ch = 256

        # 1x1 conv to reduce concatenated channels back to 256 per FPN level
        self.reducers = nn.ModuleDict(
            {
                k: nn.Sequential(
                    nn.Conv2d(in_channels, out_ch, 1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
                for k in _FPN_KEYS
            }
        )

        # --- RPN ---
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_gen = AnchorGenerator(anchor_sizes, aspect_ratios)

        rpn_head = RPNHead(
            out_ch, anchor_gen.num_anchors_per_location()[0]
        )
        self.rpn = RegionProposalNetwork(
            anchor_gen,
            rpn_head,
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n={"training": 2000, "testing": 1000},
            post_nms_top_n={"training": 2000, "testing": 1000},
            nms_thresh=0.7,
        )

        # -- ROI heads ---
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=_ROI_KEYS, output_size=7, sampling_ratio=2
        )
        box_head = TwoMLPHead(out_ch * 7 * 7, 1024)
        box_predictor = FastRCNNPredictor(1024, num_classes)

        self.roi_heads = RoIHeads(
            box_roi_pool=roi_pooler,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=0.05,
            nms_thresh=0.5,
            detections_per_img=100,
        )

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        image_list: ImageList,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        """
        Returns:
            detections: List[Dict] (empty during training)
            losses:     Dict[str, Tensor] (empty during eval)
        """

        reduced = OrderedDict(
            {k: self.reducers[k](v) for k, v in features.items()}
        )

        proposals, rpn_losses = self.rpn(image_list, reduced, targets)
        detections, roi_losses = self.roi_heads(
            reduced, proposals, image_list.image_sizes, targets
        )

        losses = {}
        losses.update(rpn_losses)
        losses.update(roi_losses)
        return detections, losses
    

# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class MultiModalFasterRCNN(nn.Module):
    """
    Multi-modal Faster R-CNN.

    - Per-modality stem (conv1) maps raw channels → 64.
    - Shared ResNet-18 + FPN encodes each modality → 256-ch feature maps.
    - All C(n,2) pairwise fusions + one n-way fusion, each with its own
      detection head (RPN + ROI).
    - Training returns the sum-ready loss dict from all heads.
    - Inference returns detections from `eval_head`.
    """

    def __init__(
        self,
        num_classes: int,
        min_size: int = 378,#450,
        max_size: int = 672,#800,
        eval_head: str = "all",
        aux_weather_weight: float = 0.2,
        aux_daytime_weight: float = 0.2,
        weather_focal_gamma: float = 2.0,
        weather_class_counts: Optional[List[int]] = None,
    ):
        """
        Args:
            weather_focal_gamma: Focusing exponent for weather focal loss.
                ``0.0`` disables focal behaviour and collapses to plain
                (optionally alpha-weighted) cross-entropy.
            weather_class_counts: Per-class sample counts from the *training*
                split, in the same order as ``WEATHER_CLASSES``. Used to
                derive smoothed inverse-frequency alpha weights. Pass ``None``
                to disable alpha weighting (uniform alpha).
        """
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.eval_head = eval_head
        self.aux_weather_weight = aux_weather_weight
        self.aux_daytime_weight = aux_daytime_weight
        self.weather_focal_gamma = float(weather_focal_gamma)

        # --- Per-class alpha weights for weather focal loss ---
        # Inverse-sqrt smoothing: 1/sqrt(count), normalised to mean 1.0 so the
        # overall loss magnitude isn't inflated relative to the daytime head.
        # Stored as a buffer so it moves with .to(device) and survives
        # state_dict save/load.
        if weather_class_counts is not None:
            counts = torch.tensor(weather_class_counts, dtype=torch.float32)
            if counts.numel() != len(WEATHER_CLASSES):
                raise ValueError(
                    f"weather_class_counts must have {len(WEATHER_CLASSES)} entries, "
                    f"got {counts.numel()}"
                )
            # Clamp to avoid division by zero for absent classes.
            alpha = 1.0 / counts.clamp(min=1.0).sqrt()
            alpha = alpha / alpha.mean()
            self.register_buffer("weather_alpha", alpha)
        else:
            self.register_buffer(
                "weather_alpha",
                torch.ones(len(WEATHER_CLASSES), dtype=torch.float32),
            )

        # --- pretrained conv1 for stem initialization ---
        _resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        _pretrained_w = _resnet.conv1.weight.clone()
        del _resnet

        # --- per-modality stems ---
        self.stems = nn.ModuleDict(
            {
                "rgb": ModalityStem(3, _pretrained_w),
                "lidar": ModalityStem(2, _pretrained_w),
                "radar": ModalityStem(2, _pretrained_w)
            }
        )

        # --- shared encoder ---
        self.encoder = SharedEncoder()

        # --- per-modality normalization (buffers move with .to(device)) ---
        self.register_buffer(
            "rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "lidar_mean", torch.tensor([0.0, 0.0]).view(1, 2, 1, 1)
        )
        self.register_buffer(
            "lidar_std", torch.tensor([1.0, 1.0]).view(1, 2, 1, 1)
        )
        self.register_buffer(
            "radar_mean", torch.tensor([0.0, 0.0]).view(1, 2, 1, 1)
        )
        self.register_buffer(
            "radar_std", torch.tensor([1.0, 1.0]).view(1, 2, 1, 1)
        )

        # --- fusion combinations & detection heads ---
        modality_names = ["rgb", "lidar", "radar"]

        self.fusion_combos: Dict[str, Tuple[str, ...]] = {}
        self.heads = nn.ModuleDict()

        head_modalities = {
            "rgb_lidar": ("rgb", "lidar"),
            "rgb_radar": ("rgb", "radar"),
            "rgb": ("rgb",),
        }
        for name, combo in head_modalities.items():
            self.fusion_combos[name] = combo
            self.heads[name] = DetectionHead(256 * len(combo), num_classes)
        # one all-sensor (3-sensor) head
        self.fusion_combos["all"] = tuple(modality_names)
        self.heads["all"] = DetectionHead(256 * len(modality_names), num_classes)

        # --- auxiliary RGB-only weather / daytime classifier ---
        # Fed from the midpoint of ResNet-18 (layer2 output, 128 channels).
        self.weather_daytime_head = WeatherDaytimeHead(in_channels=128)
        # Populated every forward pass; handy for inference-time readout.
        self.last_aux_predictions: Optional[Dict[str, torch.Tensor]] = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _target_size(self, h: int, w: int) -> Tuple[int, int]:
        """Compute resize dimensions respecting min_size / max_size."""
        scale = self.min_size / min(h, w)
        if max(h, w) * scale > self.max_size:
            scale = self.max_size / max(h, w)
        return int(h * scale), int(w * scale)

    def _normalize(self, name: str, x: torch.Tensor) -> torch.Tensor:
        mean = getattr(self, f"{name}_mean")
        std = getattr(self, f"{name}_std")
        return (x - mean) / std

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        modalities_list: List[Dict[str, torch.Tensor]],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        conditions: Optional[List[Optional[dict]]] = None,
    ):
        """
        Args:
            modalities_list: one dict per image, each containing:
                'rgb'   → (3, H, W) float tensor
                'lidar' → (2, H, W) float tensor
                'radar' → (2, H, W) float tensor  (if use_radar)
            targets: standard Faster-RCNN target dicts (boxes, labels, …).
            conditions: optional list of per-sample condition dicts
                (one per image, as produced by the dataset). When provided in
                training mode, enables the auxiliary weather/daytime loss.
                Expected keys: 'weather' ∈ {clear, rain, snow, fog} and
                'daytime' ∈ {day, night}. Anything else
                (including 'unknown' or a missing key) is silently ignored
                via ``ignore_index`` so samples without labels contribute no
                gradient.

        Returns:
            Training → Dict[str, Tensor]: losses keyed as "{head}/{loss_name}".
                Additionally includes ``aux/weather`` and ``aux/daytime``
                when ``conditions`` is provided and at least one sample has a
                known label for the given task.
            Eval     → Dict[str, List[Dict[str, Tensor]]]: detections keyed by
                fusion-head name. Auxiliary predictions for the batch are
                stashed on ``self.last_aux_predictions``.
        """

        
        device = next(self.parameters()).device
        batch_size = len(modalities_list)
        
        # --- original spatial size (assumed identical across batch) ---
        orig_h, orig_w = modalities_list[0]["rgb"].shape[-2:]
        new_h, new_w = self._target_size(orig_h, orig_w)

        # # --- stack, normalize, resize each modality ---
        batches: Dict[str, torch.Tensor] = {}
        for name in self.stems:
            t = torch.stack([m[name] for m in modalities_list]).to(device)
            t = self._normalize(name, t)
            t = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
            batches[name] = t

        # # --- rescale target boxes to match resized images ---
        if targets is not None:
            sx = new_w / orig_w
            sy = new_h / orig_h
            scaled_targets = []
            for t in targets:
                st = {
                    k: v.clone().to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                }
                st["boxes"][:, [0, 2]] *= sx
                st["boxes"][:, [1, 3]] *= sy
                scaled_targets.append(st)
        else:
            scaled_targets = None

        # --- ImageList (same size → no padding, reuse for all heads) ---
        image_sizes = [(new_h, new_w)] * batch_size
        image_list = ImageList(batches["rgb"], image_sizes)

        # --- encode each modality through its stem + shared encoder ---
        # For RGB, we additionally capture the ResNet-18 midpoint (layer2
        # output, 128ch) to feed the auxiliary weather/daytime head.
        features: Dict[str, Dict[str, torch.Tensor]] = {}
        rgb_mid: Optional[torch.Tensor] = None
        for name in self.stems:
            stem_out = self.stems[name](batches[name])
            if name == "rgb":
                features[name], rgb_mid = self.encoder.forward_with_mid(stem_out)
            else:
                features[name] = self.encoder(stem_out)

        # --- auxiliary RGB-only weather / daytime predictions ---
        aux_out = self.weather_daytime_head(rgb_mid)
        # Detach a copy for downstream inspection without holding the graph.
        self.last_aux_predictions = {k: v.detach() for k, v in aux_out.items()}

        # --- choose which heads to run ---
        if self.training:
            heads_to_run = list(self.heads.keys())
        elif self.eval_head == "conditional":
            # Per-sample routing from the auxiliary classifier (Table 5).
            sample_routes = route_heads_from_aux(aux_out)
            # Run only the unique heads the batch actually needs.
            heads_to_run = list(dict.fromkeys(sample_routes))
        elif self.eval_head == "all":
            heads_to_run = list(self.heads.keys())
        else:
            # Single explicitly-named head (e.g. "rgb_lidar").
            heads_to_run = [self.eval_head]

        # --- fuse & detect per head ---
        all_losses: Dict[str, torch.Tensor] = {}
        all_detections: Dict[str, List[Dict[str, torch.Tensor]]] = {}

        for head_name in heads_to_run:
            modality_keys = self.fusion_combos[head_name]

            # concatenate FPN features at each level
            fused_features = OrderedDict()
            for fpn_key in features[modality_keys[0]]:
                fused_features[fpn_key] = torch.cat(
                    [features[m][fpn_key] for m in modality_keys], dim=1
                )

            detections, losses = self.heads[head_name](
                fused_features, image_list, scaled_targets
            )

            for k, v in losses.items():
                weight = 1.3 if head_name == "lidar_radar" else 1.0
                all_losses[f"{head_name}/{k}"] = v * weight

            all_detections[head_name] = detections

        # --- auxiliary weather / daytime loss (training only) -------------
        if self.training and conditions is not None:
            w_tgt, d_tgt = build_aux_targets(conditions, device)

            # Weather: focal loss with per-class alpha. When
            # weather_focal_gamma=0 this collapses to alpha-weighted CE.
            # Skip entirely if no sample in the batch has a valid label,
            # otherwise the loss would be NaN.
            if (w_tgt != _AUX_IGNORE_INDEX).any():
                all_losses["aux/weather"] = self.aux_weather_weight * multiclass_focal_loss(
                    aux_out["weather_logits"], w_tgt,
                    alpha=self.weather_alpha,
                    gamma=self.weather_focal_gamma,
                    ignore_index=_AUX_IGNORE_INDEX,
                )
            # Daytime: plain cross-entropy — the classes are roughly balanced
            # and the head is already converging fine.
            if (d_tgt != _AUX_IGNORE_INDEX).any():
                all_losses["aux/daytime"] = self.aux_daytime_weight * F.cross_entropy(
                    aux_out["daytime_logits"], d_tgt,
                    ignore_index=_AUX_IGNORE_INDEX,
                )

        # --- return ---
        if self.training:
            return all_losses

        # map predictions back to original image size for every head we ran
        sx = orig_w / new_w
        sy = orig_h / new_h
        for head_name, dets in all_detections.items():
            for det in dets:
                det["boxes"][:, [0, 2]] *= sx
                det["boxes"][:, [1, 3]] *= sy

        # Conditional routing: pick each sample's detections from the head
        # the auxiliary classifier assigned it.
        if self.eval_head == "conditional" and sample_routes is not None:
            return [
                all_detections[route][i] for i, route in enumerate(sample_routes)
            ]

        return all_detections
    
# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Saves model + optimizer + scheduler state every *save_every* epochs and
    optionally resumes from a prior checkpoint.

    Usage:
        ckpt_mgr = CheckpointManager(model, optimizer, scheduler,
                                      save_dir="checkpoints", save_every=5)

        # (optional) resume from latest or a specific file
        start_epoch = ckpt_mgr.load()            # latest in save_dir
        # start_epoch = ckpt_mgr.load("checkpoints/epoch_20.pt")  # specific

        for epoch in range(start_epoch, max_epochs):
            train_one_epoch(...)
            ckpt_mgr.step(epoch)                  # saves if epoch % 5 == 0
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        save_dir: str = "checkpoints",
        save_every: int = 5,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = Path(save_dir)
        self.save_every = save_every
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ---- save ---------------------------------------------------------------

    def _build_state(self, epoch: int, **extra) -> dict:
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        state.update(extra)
        return state

    def save(self, epoch: int, **extra) -> Path:
        """Force-save a checkpoint for the given epoch."""
        path = self.save_dir / f"epoch_{epoch:04d}.pt"
        torch.save(self._build_state(epoch, **extra), path)
        logger.info("Checkpoint saved → %s", path)
        return path

    def step(self, epoch: int, **extra) -> Optional[Path]:
        """Call at the end of every epoch; saves only when due."""
        if (epoch + 1) % self.save_every == 0:
            return self.save(epoch + 1, **extra)
        return None

    # ---- load ---------------------------------------------------------------

    @staticmethod
    def _latest_in(directory: Path) -> Optional[Path]:
        pts = sorted(directory.glob("epoch_*.pt"))
        return pts[-1] if pts else None

    def load(self, path: Optional[str] = None) -> int:
        """
        Restore state from a checkpoint file.

        Args:
            path: explicit .pt file to load. If *None*, loads the latest
                  ``epoch_*.pt`` in ``self.save_dir`` (if any exist).

        Returns:
            The epoch to resume from (0 when no checkpoint is found).
        """
        if path is not None:
            ckpt_path = Path(path)
        else:
            ckpt_path = self._latest_in(self.save_dir)

        if ckpt_path is None or not ckpt_path.exists():
            logger.info("No checkpoint found — starting from scratch.")
            return 0

        device = next(self.model.parameters()).device
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        epoch = ckpt["epoch"]
        logger.info("Resumed from %s  (epoch %d)", ckpt_path, epoch)
        return epoch


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_model(num_classes: int, **kwargs):
    """
    Build multi-modal Faster R-CNN.

    Args:
        num_classes: number of object classes (excluding background —
                     Faster R-CNN adds +1 internally).
        **kwargs:    forwarded to MultiModalFasterRCNN (min_size, max_size,
                     eval_head, …).
    """
    return MultiModalFasterRCNN(
        num_classes=num_classes, **kwargs
    )


def build_model_with_checkpointing(
    num_classes: int,
    optimizer: torch.optim.Optimizer = None,
    scheduler=None,
    save_dir: str = "checkpoints",
    save_every: int = 5,
    resume: bool = False,
    resume_path: Optional[str] = None,
    **kwargs,
):
    """
    Convenience wrapper: builds model, creates a CheckpointManager,
    and optionally resumes.

    Args:
        num_classes:  passed to build_model.
        optimizer:    required for checkpointing; if None only the model is
                      returned and no CheckpointManager is created.
        scheduler:    optional LR scheduler.
        save_dir:     directory for checkpoint files.
        save_every:   save interval in epochs (default 5).
        resume:       set True to load the latest checkpoint.
        resume_path:  explicit .pt path to resume from (overrides auto-detect).
        **kwargs:     forwarded to MultiModalFasterRCNN.

    Returns:
        (model, ckpt_manager, start_epoch)
    """
    model = build_model(num_classes, **kwargs)

    if optimizer is None:
        raise ValueError("An optimizer is required to create a CheckpointManager.")

    ckpt_mgr = CheckpointManager(
        model, optimizer, scheduler,
        save_dir=save_dir, save_every=save_every,
    )

    start_epoch = 0
    if resume or resume_path is not None:
        start_epoch = ckpt_mgr.load(resume_path)

    return model, ckpt_mgr, start_epoch