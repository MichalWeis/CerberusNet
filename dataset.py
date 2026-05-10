import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
from pathlib import Path
from typing import Optional
import cv2
from tqdm import tqdm
from tools.DatasetViewer.lib.read import load_calib_data, load_velodyne_scan, read_label, load_radar_points
from tools.ProjectionTools.Lidar2RGB.lib.utils import filter
from tools.CreateTFRecords.generic_tf_tools.resize import resize
import matplotlib.pyplot as plt



# SeeingThroughFog classes
STF_CLASSES = [
    "PassengerCar",
    "LargeVehicle",
    "RidableVehicle",
    "Pedestrian",
    #"person",
    #"Vehicle",
    #"train",
    #"Obstacle",
    #"DontCare"
]

# Classes to ignore entirely (not fed to the model)
IGNORED_CLASSES = {"DontCare", "Obstacle", "train", "Vehicle"}

# Map group variants to their base class
STF_CLASS_ALIASES = {
    "PassengerCar_is_group": "PassengerCar",
    "LargeVehicle_is_group": "LargeVehicle",
    "RidableVehicle_is_group": "RidableVehicle",
    "person": "Pedestrian",
    "Pedestrian_is_group": "Pedestrian",
    "Vehicle_is_group": "Vehicle",
}


# ---------------------------------------------------------------------------
# Labeltool label loading & condition extraction
# ---------------------------------------------------------------------------


def _active_key(flag_dict: dict) -> str:
    """Return the name of the first True flag in a bool dict, or 'unknown'."""
    for k, v in flag_dict.items():
        if v:
            return k
    return "unknown"

def load_labeltool_label(json_path: str) -> dict:
    """
    Load a labeltool label JSON and flatten it into a simple conditions dict.

    Returns a dict like::

        {
            "weather":        "clear",             # active weather flag
            "daytime":        "night",             # active daytime flag
            "environment":    "dry",               # active environment flag
            "illumination":   "low_dynamic_range", # active illumination flag
            "infrastructure": "in_city",           # active infrastructure flag
            "bad_sensor":     False,
        }

    If the file does not exist or cannot be parsed, returns ``None``.
    """
    if not os.path.isfile(json_path):
        return None
    try:
        with open(json_path, "r") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    conditions = {}

    # weather
    weather = raw.get("weather", {})
    conditions["weather"] = _active_key(weather)

    # daytime
    daytime = raw.get("daytime", {})
    conditions["daytime"] = _active_key(daytime)

    # meta sub-dicts
    meta = raw.get("meta", {})
    conditions["environment"] = _active_key(meta.get("environment", {}))
    conditions["illumination"] = _active_key(meta.get("illumination", {}))
    conditions["infrastructure"] = _active_key(meta.get("infrastructure", {}))

    # bad_sensor flag
    conditions["bad_sensor"] = bool(raw.get("bad_sensor", False))

    return conditions

def collate_fn_stf(batch):
    """
    Collate function for SeeingThroughFog dataset.
    Returns tuple of (images, targets, conditions) for batch processing.
    Handles both 2-tuple (legacy) and 3-tuple (with conditions) items.
    """
    if len(batch[0]) == 3:
        modalities, targets, conditions = zip(*batch)
        return modalities, targets, conditions
    else:
        return tuple(zip(*batch))


# ---------------------------------------------------------------------------
# Lightweight dataset that loads precomputed .pt files
# ---------------------------------------------------------------------------
class PreprocessedSTFDataset(Dataset):
    """
    Loads precomputed (rgb, lidar, radar, target) tensors saved by
    ``preprocess_and_save()``.  No sensor I/O or projection math at runtime.
    """
 
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            raise FileNotFoundError(
                f"Cache directory not found: {self.cache_dir}\n"
                "Run preprocess_and_save() first to generate the cached data."
            )
        self.files = sorted(self.cache_dir.glob("*.pt"))
        if len(self.files) == 0:
            raise RuntimeError(
                f"No .pt files found in {self.cache_dir}. "
                "Run preprocess_and_save() first."
            )
        print(f"[PreprocessedSTFDataset] Found {len(self.files)} cached samples.")
    

    def __len__(self) -> int:
        return len(self.files)
 
    def __getitem__(self, idx: int):
        data = torch.load(self.files[idx], weights_only=False)
        modalities = {
            "rgb":   data["rgb"],
            "lidar": data["lidar"],
            "radar": data["radar"],
        }
        target = data["target"]
        conditions = data.get("conditions", None)
        return modalities, target, conditions
    

class SeeingThroughFogDataset(Dataset):
    """
    Loads images and annotations from the SeeingThroughFog dataset.
    
    Supports:
    - Stereo camera images (left/right)
    - Radar target detections
    - Ground truth labels from camera label files (KITTI-like format)
    - Proper coordinate transformations via calibration data
    - Multi-modal fusion for object detection
    """
    
    def __init__(
        self,
        root_dir: str,
        use_camera: str = "left_lut",
        transform=None,
        calib_from_dir: Optional[str] = None,
    ):
        """
        Args:
            root_dir: Root directory of SeeingThroughFog dataset
            use_camera: Which camera to use ("left", "right", "left_lut", "right_lut")
            transform: Optional image transforms (e.g., torchvision transforms)
            calib_from_dir: Directory containing calibration files. If None, uses root_dir
        """
        self.root_dir = Path(root_dir)
        self.use_camera = use_camera.lower()
        self.transform = transform
        self.visualize_modalities = True

        self.label_dir = self.root_dir / f"labeltool_labels"
        # Set up camera path
        self.camera_dir = self.root_dir / f"cam_stereo_{self.use_camera}"
        if not self.camera_dir.exists():
            raise FileNotFoundError(f"Camera directory not found: {self.camera_dir}")
        
        #Load calibration data
        velodyne_to_camera, camera_to_velodyne, P, R, vtc, radar_to_camera, zero_to_camera = load_calib_data(self.root_dir, 'calib_cam_stereo_left.json', 'calib_tf_tree_full.json', 'lidar_hdl64_s3_roof')

        self.P = P
        self.R = R
        self.radar_to_camera = radar_to_camera
        self.velodyne_to_camera = velodyne_to_camera
        self.vtc = vtc
        self.rtc = np.matmul(np.matmul(self.P, self.R), self.radar_to_camera)
        self.ltc = np.matmul(np.matmul(self.P, self.R), self.velodyne_to_camera)

        self.vel_max = 0.0
        self.intensity_max = 0.0

        # Get list of images
        if self.use_camera in ["left", "right"]:
            self.image_files = sorted(self.camera_dir.glob("*.tiff"))
        else:
            self.image_files = sorted(self.camera_dir.glob("*.png"))

        # Load annotations and BBoxes
        self.annotations = {}
        self.conditions = {}
        for img_file in tqdm(self.image_files, desc="Loading annotations and conditions"):
            self.bbox, self.label= self.get_annotations(img_file, self.root_dir / "gt_labels" / "cam_left_labels_TMP")
            if len(self.bbox) == 0:
                continue  # skip frames with no valid annotations
            self.annotations[img_file] = {"bbox": self.bbox, "label": self.label}

            label_json = self.label_dir / (img_file.stem + ".json")
            cond = load_labeltool_label(str(label_json))
            if cond is not None:
                self.conditions[img_file] = cond
        print(f"Loaded {len(self.annotations)} annotations for {len(self.image_files)} images.")
        print(f"Loaded {len(self.conditions)} labeltool labels (conditions) for {len(self.image_files)} images.")

    def _show_modality_views(
        self,
        rgb_image: np.ndarray,
        lidar_map: np.ndarray,
        radar_map: np.ndarray,
        sample_name: str,
    ) -> None:
        """Render aligned modality views for one sample."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(rgb_image)
        axes[0].set_title(f"RGB: {sample_name}")
        axes[0].axis("off")

        axes[1].imshow(lidar_map[..., 0], cmap="magma", vmin=0.0, vmax=1.0)
        axes[1].set_title("Projected LiDAR (depth)")
        axes[1].axis("off")

        axes[2].imshow(radar_map[..., 0], cmap="viridis", vmin=0.0, vmax=1.0)
        axes[2].set_title("Projected Radar (distance)")
        axes[2].axis("off")

        fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)


    def _rasterize_lidar_to_image(self, lidar_points_2D, pts_3D_yzi, image_hw,
                               max_depth=100.0, max_intensity=255.0):
        """
        Rasterize projected LiDAR points into image-aligned depth + intensity maps.
    
        Args:
            lidar_points_2D: Nx2 projected pixel coordinates (u, v).
            pts_3D_yzi:      3xN array — rows [y_cam, z_cam (depth), intensity].
            image_hw:        (H, W) target canvas size.
            max_depth:       Fixed max depth in meters for normalization.
            max_intensity:   Fixed max intensity value for normalization.
                            HDL-64 returns raw reflectivity 0–255.
    
        Returns:
            (H, W, 2) float32 — channels [depth, intensity], values in [0, 1].
        """
        h, w = image_hw
        depth     = np.zeros((h, w), dtype=np.float32)
        intensity = np.zeros((h, w), dtype=np.float32)
    
        uv = np.round(lidar_points_2D).astype(np.int32)
        x = uv[:, 0]
        y = uv[:, 1]
    
        # pts_3D_yzi rows: [y_cam, z_cam, intensity]
        z = pts_3D_yzi[1, :].astype(np.float32)
        i = pts_3D_yzi[2, :].astype(np.float32)
    
        valid = (x >= 0) & (x < w) & (y >= 0) & (y < h) & (z > 0.0)
        x, y, z, i = x[valid], y[valid], z[valid], i[valid]
    
        if len(x) == 0:
            return np.stack([depth, intensity], axis=-1)
    
        # --- Vectorized z-buffer (sort by depth descending, scatter) --------
        # Writing far points first, then closer points overwrite them,
        # so the final value at each pixel is the nearest point.
        order = np.argsort(-z)
        x, y, z, i = x[order], y[order], z[order], i[order]
    
        # numpy fancy indexing: last write wins → nearest point wins
        depth[y, x]     = z
        intensity[y, x] = i
    
        # --- Fixed-range normalization --------------------------------------
        # Depth: [0, max_depth] → [0, 1]
        depth = np.clip(depth / max_depth, 0.0, 1.0)
    
        # Intensity: [0, max_intensity] → [0, 1]
        # Using fixed range so the network sees consistent values across frames.
        intensity = np.clip(intensity / max_intensity, 0.0, 1.0)
    
        return np.stack([depth, intensity], axis=-1)  # HxWx2

    def _rasterize_radar_to_image(self, radar_points, rtc, image_hw,
                               max_distance=200.0, max_velocity=67.786,
                               blur_sigma=2.0):
        """
        Rasterize projected radar targets into image-aligned maps for CNN input.
    
        Args:
            radar_points: Nx5 array [x, y, z, radial_velocity, distance]
            rtc:          3x4 projection matrix — must match the coordinate
                        space of image_hw (use unscaled self.rtc for original
                        image dims, or scaled_rtc for resize-target dims).
            image_hw:     (H, W) — must match the projection matrix space.
            max_distance: Fixed max distance (meters) for normalization.
            max_velocity: Fixed max absolute velocity (m/s) for normalization.
            blur_sigma:   Gaussian sigma for densifying sparse detections.
    
        Returns:
            HxWx2 float32 map: [distance_map, velocity_map], values in [0, 1].
        """
        h, w = image_hw
        dist_map = np.zeros((h, w), dtype=np.float32)
        vel_map  = np.zeros((h, w), dtype=np.float32)
        # Separate mask to track which pixels have actual detections,
        # so blur doesn't corrupt the normalization.
        hit_map  = np.zeros((h, w), dtype=np.float32)
    
        if radar_points is None or radar_points.size == 0:
            return np.stack([dist_map, vel_map], axis=-1)
    
        # --- Project 3D → 2D -----------------------------------------------
        points_xyz  = radar_points[:, :3].astype(np.float32)
        points_vel  = radar_points[:, 3].astype(np.float32)
        points_dist = radar_points[:, 4].astype(np.float32)
    
        points_h = np.hstack([points_xyz, np.ones((points_xyz.shape[0], 1),
                                                    dtype=np.float32)])
        proj = np.matmul(rtc, points_h.T)  # (3, N)
    
        valid_depth = proj[2, :] > 1e-6
        if not np.any(valid_depth):
            return np.stack([dist_map, vel_map], axis=-1)
    
        proj        = proj[:, valid_depth]
        points_vel  = points_vel[valid_depth]
        points_dist = points_dist[valid_depth]
    
        u = np.round(proj[0, :] / proj[2, :]).astype(np.int32)
        v = np.round(proj[1, :] / proj[2, :]).astype(np.int32)
    
        valid_img = (u >= 0) & (u < w) #& (v >= 0) & (v < h)
        u           = u[valid_img]
        #v           = v[valid_img]
        points_vel  = points_vel[valid_img]
        points_dist = points_dist[valid_img]
    
        # --- Z-buffer: keep nearest detection per pixel ---------------------
        for px, p_dist, p_vel in zip(u, points_dist, points_vel):
            old = dist_map[0, px] # whole column shares the same value
            if old == 0.0 or p_dist < old:
                dist_map[:, px] = p_dist
                vel_map[:, px]  = p_vel
                hit_map[:, px]  = 1.0
    
        # --- Fixed-range normalization --------------------------
        # Distance: [0, max_distance] → [0, 1]
        dist_map = np.clip(dist_map / max_distance, 0.0, 1.0)
    
        # Velocity: [-max_velocity, +max_velocity] → [0, 1], with 0.5 = stationary
        vel_map = np.clip(0.5 * (vel_map / max_velocity) + 0.5, 0.0, 1.0)
    
        # Zero out pixels with no radar hit (velocity 0.5 would be misleading)
        dist_map *= hit_map
        vel_map  *= hit_map
    
        dist_map = np.clip(dist_map, 0.0, 1.0)
        vel_map  = np.clip(vel_map, 0.0, 1.0)
    
        return np.stack([dist_map, vel_map], axis=-1)

    def get_annotations(self, img_file, label_dir):
        img_file = str(img_file)
        img_file = img_file.split("/")[-1]
        object_list = read_label(str(img_file), self.root_dir / "gt_labels" / "cam_left_labels_TMP")
        boxes = []
        labels = []


        for obj in object_list:
            label = obj['identity']
            label = STF_CLASS_ALIASES.get(label, label)
            if label in IGNORED_CLASSES:
                continue  # skip ignored classes entirely
            x1 = float(obj['xleft'])
            y1 = float(obj['ytop'])
            x2 = float(obj['xright'])
            y2 = float(obj['ybottom'])
            visibleRGB = obj['visibleRGB']
            visibleLidar = obj['visibleLidar']
            visibleRadar = obj['visibleRadar']

            if not (visibleRGB or visibleLidar or visibleRadar):
                continue  # skip objects not visible in any modality

            # Validate bounding box
            if x2 <= x1 or y2 <= y1:
                continue
            if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:  # minimum 1px size
                continue
            
            boxes.append([x1, y1, x2, y2])
            labels.append(label)


        if not boxes:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        else:
            return np.array(boxes, dtype=np.float32), labels
        
    # STF FUNCTION
    def py_func_project_3D_to_2D(self, points_3D, P):
        # Project on image
        points_2D = np.matmul(P, np.vstack((points_3D, np.ones([1, np.shape(points_3D)[1]]))))

        # scale projected points
        points_2D[0][:] = points_2D[0][:] / points_2D[2][:]
        points_2D[1][:] = points_2D[1][:] / points_2D[2][:]

        points_2D = points_2D[0:2]
        return points_2D.transpose()
    
    # STF FUNCTION
    def py_func_lidar_projection(self, lidar_points_3D, vtc, velodyne_to_camera, shape):

        img_width = shape[1]
        img_height = shape[0]
        # print img_height, img_width
        lidar_points_3D = lidar_points_3D[:, 0:4] # TOTO JE PROBLEM PRE RADAR

        # Filer away all points behind image plane
        min_x = 2.5
        valid = lidar_points_3D[:, 0] > min_x
        # extend projection matrix to 5d to efficiently parse intensity
        lidar_points_3D = lidar_points_3D[np.where(valid)]
        lidar_points_3D2 = np.ones((lidar_points_3D.shape[0], lidar_points_3D.shape[1] + 1))
        lidar_points_3D2[:, 0:3] = lidar_points_3D[:, 0:3] 
        lidar_points_3D2[:, 4] = lidar_points_3D[:, 3]
        # Extend projection matric to pass trough intensities
        velodyne_to_camera2 = np.zeros((5, 5))
        velodyne_to_camera2[0:4, 0:4] = velodyne_to_camera
        velodyne_to_camera2[4, 4] = 1

        lidar_points_2D = self.py_func_project_3D_to_2D(lidar_points_3D.transpose()[:][0:3], vtc)

        pts_3D = np.matmul(velodyne_to_camera2, lidar_points_3D2.transpose())
        # detelete placeholder 1 axis
        pts_3D = np.delete(pts_3D, 3, axis=0) 

        pts_3D_yzi = pts_3D[1:, :] 

        return lidar_points_2D, pts_3D_yzi

    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int):
        img_path = self.image_files[idx]

        # Load image
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Common resize target (used for all modalities)
        r = resize('shrink')
        scaled_h, scaled_w = list(r.dsize)[::-1]

        # Get radar data
        img_name = img_path.stem
        radar_file = os.path.join(self.root_dir, 'radar_targets',
                                    img_name + '.json')
        
        radar_data = load_radar_points(radar_file)

        # Get LiDAR data
        velo_file_last = os.path.join(self.root_dir, 'lidar_hdl64_last',
                                      img_name + '.bin') 
        lidar_data_last = load_velodyne_scan(velo_file_last)

        lidar_data_last = filter(lidar_data_last, 1.5)

        # --- Rasterize LiDAR at scaled dims ---
        scaled_vtc = np.matmul(r.get_image_scaling(), self.vtc)

        lidar_points_2D, pts_3D_yzi = self.py_func_lidar_projection(
                                            lidar_data_last, 
                                            scaled_vtc, 
                                            self.velodyne_to_camera,
                                            [scaled_h, scaled_w, 3]
            )

        lidar_map = self._rasterize_lidar_to_image(
            lidar_points_2D, pts_3D_yzi, (scaled_h, scaled_w)
        )  # HxWx2

        # --- Rasterize radar at scaled dims ---
        scaled_rtc = np.matmul(r.get_image_scaling(), self.rtc)
        radar_map = self._rasterize_radar_to_image(
            radar_data,
            scaled_rtc,
            (scaled_h, scaled_w),
        )  # HxWx2

        

        # --- Resize RGB to same scaled dims ---
        image_resized = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        image_f = image_resized.astype(np.float32) / 255.0  # HxWx3, [0, 1]

        if self.visualize_modalities:
            canvas = np.full((h, w, 3), dtype=np.uint8)

            canvas[0].imshow(image_f)
            canvas[0].set_title(f"RGB: {img_path.stem}")
            canvas[0].axis("off")


        # --- Scale bounding boxes to match resized image ---
        ann = self.annotations[img_path]
        boxes = ann["bbox"].copy()
        orig_h, orig_w = image.shape[:2]
        scale_x = scaled_w / orig_w
        scale_y = scaled_h / orig_h
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Map class names -> integer ids (+1 because Faster R-CNN reserves 0 for background)
        labels = ann["label"]
        label_ids = [
            STF_CLASSES.index(STF_CLASS_ALIASES.get(lbl, lbl)) + 1
            for lbl in labels
        ]
        
        labels = torch.as_tensor(label_ids, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        # --- Build per-modality tensors (C, H, W) ---
        # Model normalizes internally, so pass raw [0, 1] values
        rgb_tensor   = torch.from_numpy(image_f).permute(2, 0, 1).float()    # (3, H, W)
        lidar_tensor = torch.from_numpy(lidar_map).permute(2, 0, 1).float()  # (2, H, W)
        radar_tensor = torch.from_numpy(radar_map).permute(2, 0, 1).float()  # (2, H, W)

        modalities = {
            "rgb":   rgb_tensor,
            "lidar": lidar_tensor,
            "radar": radar_tensor,
        }

        # Get conditions (labeltool labels) for this frame
        conditions = self.conditions.get(img_path, None)

        return modalities, target, conditions

# ---------------------------------------------------------------------------
# One-time preprocessing: transform all samples and save to disk
# ---------------------------------------------------------------------------
def preprocess_and_save(
    root_dir: str,
    cache_dir: str,
    use_camera: str = "left_lut",
):
    """
    Run every heavy transformation (image load, LiDAR/radar projection,
    rasterization, resize, bbox scaling) once and save each sample as a
    compact .pt file under ``cache_dir``.
 
    Args:
        root_dir:   Root directory of the SeeingThroughFog dataset.
        cache_dir:  Directory where preprocessed .pt files will be saved.
        use_camera: Which camera to use.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
 
    # Build the raw dataset (loads calibration + annotations)
    dataset = SeeingThroughFogDataset(root_dir=root_dir, use_camera=use_camera)
 
    skipped = 0
    saved = 0
    not_used = 0
 
    for idx in tqdm(range(len(dataset)), desc="Preprocessing & saving"):
        img_path = dataset.image_files[idx]
 
        # Skip images that had no valid annotations
        if img_path not in dataset.annotations:
            skipped += 1
            continue
 
        out_file = cache_path / f"{img_path.stem}.pt"
 
        # Skip if already preprocessed (allows resuming)
        if out_file.exists():
            skipped += 1
            continue
 
        try:
            modalities, target, conditions = dataset[idx]
        except Exception as e:
            print(f"[WARN] Skipping {img_path.stem}: {e}")
            not_used += 1
            continue
 
        torch.save(
            {
                "rgb":   modalities["rgb"],
                "lidar": modalities["lidar"],
                "radar": modalities["radar"],
                "target": target,
                "conditions": conditions,
            },
            out_file,
        )
        saved += 1
 
    print(f"Done. Saved {saved} samples, skipped {skipped}, not used {not_used}. "
          f"Cache directory: {cache_path}")

# ---------------------------------------------------------------------------
# Dataloader factory
# ---------------------------------------------------------------------------

def create_dataloaders(
    root_dir: str,
    cache_dir: str = None,
    batch_size: int = 32,
    num_workers: int = 0,
    use_camera: str = "left_lut",
    split_ratio: float = 0.8,
    split_mode: str = "simple",
    num_folds: int = 5,
    fold_index: int = 0,
    seed: int = 42,
):
    """
    Create dataloaders for SeeingThroughFog dataset.
    
    If ``cache_dir`` is provided and contains preprocessed .pt files, loads
    from cache (fast).  Otherwise falls back to the raw dataset (slow).
 
        Supported split modes:
                - "simple": random train/val split using ``split_ratio``.
                - "kfold": standard deterministic k-fold train/val split where
                    val fold = ``fold_index`` and train uses all remaining folds.

    Args:
        root_dir:    Root directory of dataset
        cache_dir:   Directory with preprocessed .pt files (from preprocess_and_save).
                     If None, the raw dataset is used.
        batch_size:  Batch size for dataloaders
        num_workers: Number of worker processes
        use_camera:  Which camera to use
        split_ratio: Train/val split ratio
        split_mode:  "simple" or "kfold"
        num_folds:   Number of folds for k-fold split
        fold_index:  Fold used as validation split in k-fold mode
        seed:        Random seed for deterministic shuffled splits
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    # --- Pick the right dataset class ---
    if cache_dir is not None and Path(cache_dir).exists() and any(Path(cache_dir).glob("*.pt")):
        print(f"[create_dataloaders] Loading preprocessed data from {cache_dir}")
        dataset = PreprocessedSTFDataset(cache_dir=cache_dir)
    else:
        if cache_dir is not None:
            print(f"[create_dataloaders] Cache dir '{cache_dir}' not found or empty, "
                  "falling back to raw dataset.")
        dataset = SeeingThroughFogDataset(
            root_dir=root_dir,
            use_camera=use_camera,
        )

    if split_mode == "kfold":
        if num_folds < 2:
            raise ValueError("num_folds must be at least 2 for train/val splits")
        if len(dataset) < num_folds:
            raise ValueError(
                f"Dataset has {len(dataset)} samples, but num_folds={num_folds}. "
                "Reduce num_folds or use more data."
            )
        if not 0 <= fold_index < num_folds:
            raise ValueError(f"fold_index must be in [0, {num_folds - 1}], got {fold_index}")

        generator = torch.Generator().manual_seed(seed)
        shuffled_indices = torch.randperm(len(dataset), generator=generator).tolist()

        fold_sizes = [len(dataset) // num_folds] * num_folds
        for i in range(len(dataset) % num_folds):
            fold_sizes[i] += 1

        folds = []
        start = 0
        for fold_size in fold_sizes:
            folds.append(shuffled_indices[start:start + fold_size])
            start += fold_size

        val_fold = fold_index
        val_indices = folds[val_fold]

        train_indices = []
        for idx, fold in enumerate(folds):
            if idx != val_fold:
                train_indices.extend(fold)

        train_ds = torch.utils.data.Subset(dataset, train_indices)
        val_ds = torch.utils.data.Subset(dataset, val_indices)
    else:
        # Backward-compatible random train/val split
        train_size = int(split_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_stf,
        num_workers=num_workers,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_stf,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader