"""
Overlay radar detections on an RGB image, coloring points by distance. 
File made with use of Calude and Github Copilot. 

 
Usage:
    from visualize_radar_overlay import overlay_radar_on_rgb
 
    # radar_map: (1024, 1920, 2) float32 — channels [velocity, distance]
    # rgb:       (1024, 1920, 3) uint8 or float32 [0-1]
    overlay = overlay_radar_on_rgb(radar_map, rgb)
    cv2.imwrite("overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
"""
 
import numpy as np
import cv2
import matplotlib.cm as cm
 
 
def overlay_radar_on_rgb(
    radar_points: np.ndarray,
    rtc: np.ndarray,
    rgb: np.ndarray,
    col_width: int = 3,
    cmap_name: str = "turbo",
    alpha: float = 0.6,
    invert_depth: bool = True,
) -> np.ndarray:
    """
    Project raw radar targets onto an RGB image as full-height vertical columns.
 
    Args:
        radar_points: (N, 5) array — columns [x, y, z, velocity, distance].
        rtc:          (3, 4) projection matrix (P @ R @ radar_to_camera).
        rgb:          (H, W, 3) image, uint8 [0-255] or float32 [0-1].
        col_width:    Half-width of each column in pixels (total = 2*col_width+1).
        cmap_name:    Matplotlib colormap name.
        alpha:        Opacity of columns (0.0–1.0).
        invert_depth: If True, close = warm (red), far = cool (blue).
 
    Returns:
        (H, W, 3) uint8 RGB image with radar column overlay.
    """
    if rgb.dtype in (np.float32, np.float64):
        canvas = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8).copy()
    else:
        canvas = rgb.copy()
 
    h, w = canvas.shape[:2]
 
    if radar_points is None or radar_points.size == 0:
        return canvas
 
    # --- Project 3D radar points to 2D ----------------------------------
    xyz = radar_points[:, :3].astype(np.float64)
    dist = radar_points[:, 4].astype(np.float64)
 
    pts_h = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
    proj = rtc @ pts_h.T  # (3, N)
 
    valid = proj[2, :] > 1e-6
    proj = proj[:, valid]
    dist = dist[valid]
 
    if proj.shape[1] == 0:
        return canvas
 
    u = proj[0, :] / proj[2, :]
 
    in_bounds = (u >= 0) & (u < w)
    u = u[in_bounds]
    dist = dist[in_bounds]
 
    if len(u) == 0:
        return canvas
 
    # --- Colormap by distance -------------------------------------------
    d_min, d_max = dist.min(), dist.max()
    if d_max - d_min < 1e-6:
        d_norm = np.full_like(dist, 0.5)
    else:
        d_norm = (dist - d_min) / (d_max - d_min)
 
    if invert_depth:
        d_norm = 1.0 - d_norm
 
    cmap = cm.get_cmap(cmap_name)
    colors = (cmap(d_norm)[:, :3] * 255).astype(np.uint8)
 
    # --- Draw columns (far first so close columns are on top) -----------
    order = np.argsort(-dist)
    u, colors = u[order], colors[order]
 
    overlay = canvas.copy()
    for px, col in zip(u.astype(int), colors):
        x0 = max(px - col_width, 0)
        x1 = min(px + col_width + 1, w)
        overlay[:, x0:x1] = col  # full-height vertical stripe
 
    # Alpha blend columns onto original image
    canvas = cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0)

    return canvas


def render_radar_only(
    radar_points: np.ndarray,
    rtc: np.ndarray,
    image_size: tuple,
    col_width: int = 3,
    cmap_name: str = "turbo",
    bg_color: tuple = (0, 0, 0),
    invert_depth: bool = True,
) -> np.ndarray:
    """
    Render radar targets as full-height vertical columns on a blank canvas.

    Args:
        radar_points: (N, 5) array — columns [x, y, z, velocity, distance].
        rtc:          (3, 4) projection matrix (P @ R @ radar_to_camera).
        image_size:   (H, W) output image size in pixels.
        col_width:    Half-width of each column in pixels (total = 2*col_width+1).
        cmap_name:    Matplotlib colormap name.
        bg_color:     Background RGB color as a (R, G, B) tuple (default black).
        invert_depth: If True, close = warm (red), far = cool (blue).

    Returns:
        (H, W, 3) uint8 RGB image with radar columns on plain background.
    """
    h, w = image_size
    canvas = np.full((h, w, 3), bg_color, dtype=np.uint8)

    if radar_points is None or radar_points.size == 0:
        return canvas

    # --- Project 3D radar points to 2D ----------------------------------
    xyz = radar_points[:, :3].astype(np.float64)
    dist = radar_points[:, 4].astype(np.float64)

    pts_h = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
    proj = rtc @ pts_h.T  # (3, N)

    valid = proj[2, :] > 1e-6
    proj = proj[:, valid]
    dist = dist[valid]

    if proj.shape[1] == 0:
        return canvas

    u = proj[0, :] / proj[2, :]

    in_bounds = (u >= 0) & (u < w)
    u = u[in_bounds]
    dist = dist[in_bounds]

    if len(u) == 0:
        return canvas

    # --- Colormap by distance -------------------------------------------
    d_min, d_max = dist.min(), dist.max()
    if d_max - d_min < 1e-6:
        d_norm = np.full_like(dist, 0.5)
    else:
        d_norm = (dist - d_min) / (d_max - d_min)

    if invert_depth:
        d_norm = 1.0 - d_norm

    cmap = cm.get_cmap(cmap_name)
    colors = (cmap(d_norm)[:, :3] * 255).astype(np.uint8)

    # --- Draw columns (far first so close columns are on top) -----------
    order = np.argsort(-dist)
    u, colors = u[order], colors[order]

    for px, col in zip(u.astype(int), colors):
        x0 = max(px - col_width, 0)
        x1 = min(px + col_width + 1, w)
        canvas[:, x0:x1] = col

    return canvas