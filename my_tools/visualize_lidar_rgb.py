"""
Visualize a radar/LiDAR projection map on top of a camera image.

The projection map has shape (1024, 1920, 2) aligned to the image grid.
Channel 0 and Channel 1 hold sensor values (e.g. depth & intensity,
depth & speed, etc.). Points with non-zero values are overlaid on the image,
colored by the selected channel.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path


def visualize_radar_on_image(
    image_path: Path,
    radar_map: np.ndarray,
    color_channel: int = 0,
    color_label: str = "Depth",
    cmap: str = "turbo",
    point_size: float = 1.0,
    alpha: float = 0.8,
    figsize: tuple = (19.2, 10.24),
    title: str | None = None,
    save_path: Path | None = None,
):
    """
    Overlay radar/LiDAR points on a camera image.

    Parameters
    ----------
    image_path : Path
        Path to the camera image (.png).
    radar_map : np.ndarray
        Projection map of shape (H, W, 2), aligned to the image.
        Non-zero entries are treated as valid points.
    color_channel : int
        Which channel (0 or 1) to use for coloring the points.
    color_label : str
        Label for the colorbar (e.g. "Depth [m]", "Intensity", "Speed [m/s]").
    cmap : str
        Matplotlib colormap name.
    point_size : float
        Marker size for each point.
    alpha : float
        Point transparency.
    figsize : tuple
        Figure size in inches.
    title : str or None
        Plot title.
    save_path : Path or None
        If set, save figure to this path instead of displaying.
    """
    image = plt.imread(str(image_path))

    # Find valid (non-zero) points — a point is valid if at least one channel != 0
    mask = np.any(radar_map != 0, axis=-1)
    rows, cols = np.where(mask)

    values = radar_map[rows, cols, color_channel]

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(image)
    sc = ax.scatter(
        cols, rows,
        c=values,
        s=point_size,
        cmap=cmap,
        alpha=alpha,
        marker=".",
        norm=Normalize(vmin=np.percentile(values, 1),
                       vmax=np.percentile(values, 99)),
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(color_label)

    ax.set_title(title or f"Radar/LiDAR overlay — colored by {color_label}")
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ── Example usage ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    image_path = Path(
        "/home/misko/projects/BP/data/SeeingThroughFog/"
        "SeeingThroughFog/cam_stereo_left_lut/"
        "2019-05-02_17-21-16_02190.png"
    )
    radar_map_path = Path("path/to/your/radar_map.npy")  # <-- replace

    # Load the radar/LiDAR map — adjust to your format:
    #   np.load("file.npy")
    #   np.fromfile("file.bin", dtype=np.float32).reshape(1024, 1920, 2)
    radar_map = np.load(str(radar_map_path))

    print(f"Radar map shape: {radar_map.shape}")
    print(f"Channel 0 range: {radar_map[..., 0].min():.2f} – {radar_map[..., 0].max():.2f}")
    print(f"Channel 1 range: {radar_map[..., 1].min():.2f} – {radar_map[..., 1].max():.2f}")
    print(f"Non-zero points:  {np.any(radar_map != 0, axis=-1).sum()}")

    # Visualize colored by channel 0 (e.g. depth)
    visualize_radar_on_image(
        image_path=image_path,
        radar_map=radar_map,
        color_channel=0,
        color_label="Depth [m]",
        cmap="turbo",
        point_size=1.5,
    )

    # Visualize colored by channel 1 (e.g. intensity / speed)
    visualize_radar_on_image(
        image_path=image_path,
        radar_map=radar_map,
        color_channel=1,
        color_label="Intensity",
        cmap="inferno",
        point_size=1.5,
    )
