import sys
from pathlib import Path
# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for debugging

from tools.DatasetViewer.lib.read import load_velodyne_scan
from tools.DatasetViewer.lib.read import load_calib_data
from tools.ProjectionTools.Lidar2RGB.lib.utils import filter, \
    find_closest_neighbors, find_missing_points, transform_coordinates, project_pointcloud
from tools.ProjectionTools.Lidar2RGB.lib.visi import plot_spherical_scatter_plot, plot_image_projection
from tools.CreateTFRecords.generic_tf_tools.resize import resize
import matplotlib.pyplot as plt
import cv2
# import cv2
import numpy as np

import os
import argparse


def parsArgs():
    parser = argparse.ArgumentParser(description='Lidar 2d projection tool')
    parser.add_argument('--root', '-r', help='Enter the root folder')
    parser.add_argument('--lidar_type', '-t', help='Enter the root folder', default='lidar_hdl64',
                        choices=['lidar_hdl64', 'lidar_vlp32'])
    args = parser.parse_args()

    return args


def _find_corresponding_image(root, sample_id):
    candidates = [
        os.path.join(root, 'cam_stereo_left_lut', sample_id + '.png'),
        os.path.join(root, 'cam_stereo_left', sample_id + '.tiff'),
        os.path.join(root, 'cam_stereo_right_lut', sample_id + '.png'),
        os.path.join(root, 'cam_stereo_right', sample_id + '.tiff'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _check_image_resolution(root, sample_id):
    """Check actual image resolution"""
    img_path = _find_corresponding_image(root, sample_id)
    if img_path:
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
            return w, h
    return None


def plot_image_and_lidar_side_by_side(pointcloud, vtc, velodyne_to_camera, root, sample_id, title=None):
    r = resize('default')
    lidar_image = project_pointcloud(
        pointcloud,
        np.matmul(r.get_image_scaling(), vtc),
        velodyne_to_camera,
        list(r.dsize)[::-1] + [3],
        init=np.zeros(list(r.dsize)[::-1] + [3]),
        draw_big_circle=True,
    )

    image_path = _find_corresponding_image(root, sample_id)
    if image_path is None:
        plot_image_projection(pointcloud, vtc, velodyne_to_camera, title=title)
        return

    rgb = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if rgb is None:
        plot_image_projection(pointcloud, vtc, velodyne_to_camera, title=title)
        return

    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    h, w = lidar_image.shape[:2]
    if rgb.shape[0] != h or rgb.shape[1] != w:
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].imshow(rgb)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')

    axes[1].imshow(lidar_image)
    axes[1].set_title('LiDAR Projection')
    axes[1].axis('off')

    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    plt.close('all')  # Don't show, just close after drawing


interesting_samples = [
    '2018-02-06_14-25-51_00400',
    '2019-09-11_16-39-41_01770',
    '2018-02-12_07-16-32_00100',
    '2018-10-29_16-42-03_00560',
]

echos = [
    ['last', 'strongest'],
]

if __name__ == '__main__':

    args = parsArgs()

    velodyne_to_camera, camera_to_velodyne, P, R, vtc, radar_to_camera, zero_to_camera = load_calib_data(
        args.root, name_camera_calib='calib_cam_stereo_left.json', tf_tree='calib_tf_tree_full.json',
        velodyne_name='lidar_hdl64_s3_roof' if args.lidar_type == 'lidar_hdl64' else 'lidar_vlp32_roof')

    # Check actual image resolution
    first_sample = interesting_samples[0]
    actual_res = _check_image_resolution(args.root, first_sample)
    if actual_res:
        actual_w, actual_h = actual_res
        print(f"Actual image resolution: {actual_w}x{actual_h}")
        print(f"Expected (from resize 'default'): 1920x1024")
        if actual_w != 1920 or actual_h != 1024:
            print(f"  ⚠ Resolution mismatch! Calibration may be for different resolution.")

    for interesting_sample in interesting_samples:
        velo_file_last = os.path.join(args.root, args.lidar_type + '_' + echos[0][0],
                                      interesting_sample + '.bin')
        velo_file_strongest = os.path.join(args.root, args.lidar_type + '_' + echos[0][1],
                                           interesting_sample + '.bin')
        lidar_data_last = load_velodyne_scan(velo_file_last)
        lidar_data_strongest = load_velodyne_scan(velo_file_strongest)

        print('last shape:', lidar_data_last.shape)
        lidar_data_last = filter(lidar_data_last, 1.5)
        print('strongest shape:', lidar_data_strongest.shape)
        lidar_data_strongest = filter(lidar_data_strongest, 1.5)

        remaining_last, remaining_strong = find_missing_points(lidar_data_last, lidar_data_strongest)
        valid = find_closest_neighbors(transform_coordinates(remaining_strong), transform_coordinates(remaining_last))

        plot_spherical_scatter_plot(lidar_data_last, pattern='hot', title='Spherical Plot Last Echo')
        plt.close('all')
        plot_spherical_scatter_plot(lidar_data_strongest, pattern='cool', title='Spherical Plot Strongest Echo')
        plt.close('all')

        print(len(remaining_strong), '/', len(lidar_data_strongest), len(remaining_last), '/', len(lidar_data_last))
        print("intensity_mean", np.mean(lidar_data_strongest[:, 3]), np.mean(lidar_data_last[:, 3]))
        print("intensity_mean", np.mean(remaining_strong[:, 3]), np.mean(remaining_last[:, 3]))

        plot_spherical_scatter_plot(remaining_last, pattern='hot', plot_show=False)
        plt.close('all')
        plot_spherical_scatter_plot(remaining_strong, pattern='cool', title='Not matching echos')
        plt.close('all')

        plot_image_and_lidar_side_by_side(
            lidar_data_last,
            vtc,
            velodyne_to_camera,
            args.root,
            interesting_sample,
            title='Camera + LiDAR Projection Last Echo',
        )
        plot_image_and_lidar_side_by_side(
            lidar_data_strongest,
            vtc,
            velodyne_to_camera,
            args.root,
            interesting_sample,
            title='Camera + LiDAR Projection Strongest Echo',
        )