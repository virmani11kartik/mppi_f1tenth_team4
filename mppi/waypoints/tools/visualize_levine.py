#!/usr/bin/env python3
"""
visualize_levine.py

Usage:
    python visualize_levine.py

Visualize levin.csv waypoints over levin.png background image using map metadata.
"""
import os
import csv
import matplotlib.pyplot as plt
import yaml

def main():
    # Determine file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    levin_dir = os.path.join(script_dir, '..', 'race')
    csv_path = os.path.join(levin_dir, 'map_edited_waypoints.csv')
    img_path = os.path.join(levin_dir, 'map_edited.png')
    yaml_path = os.path.join(levin_dir, 'map_edited.yaml')
    # levin_dir = os.path.join(script_dir, '..', 'levine')
    # csv_path = os.path.join(levin_dir, 'levine.csv')
    # img_path = os.path.join(levin_dir, 'levine.png')
    # yaml_path = os.path.join(levin_dir, 'levine.yaml')
    # Load map metadata for resolution and origin
    with open(yaml_path, 'r') as f:
        map_info = yaml.safe_load(f)
    resolution = map_info.get('resolution', 1.0)
    origin = map_info.get('origin', [0.0, 0.0, 0.0])
    origin_x, origin_y = origin[0], origin[1]

    # Load background image
    img = plt.imread(img_path)
    height, width = img.shape[0], img.shape[1]

    # Compute image extent in world coordinates
    xmin = origin_x
    ymin = origin_y
    xmax = origin_x + width * resolution
    ymax = origin_y + height * resolution

    # Create plot
    fig, ax = plt.subplots()
    ax.imshow(img, extent=[xmin, xmax, ymin, ymax], origin='upper')

    # Load waypoints CSV manually due to pandas error
    xs, ys = [], []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(
            (line for line in f if not line.startswith('#')),
            fieldnames=[
                's_m', 'x_m', 'y_m', 'psi_rad', 'kappa_radpm',
                'vx_mps', 'ax_mps2', 'w_tr_right_m', 'w_tr_left_m'
            ],
            delimiter=';'
        )
        for row in reader:
            xs.append(float(row['x_m']))
            ys.append(float(row['y_m']))
    # Plot individual waypoints as red dots
    ax.scatter(xs, ys, c='r', s=5, label='waypoints')

    # Finalize plot appearance
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    plt.title('Levine Waypoints Visualization')
    plt.show()

if __name__ == '__main__':
    main() 