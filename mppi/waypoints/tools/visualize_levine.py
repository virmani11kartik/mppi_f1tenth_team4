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
import numpy as np
import yaml
import matplotlib.patches as patches

def main():
    # Determine file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # levin_dir = os.path.join(script_dir, '..', 'race')
    # csv_path = os.path.join(levin_dir, 'map_edited_waypoints.csv')
    # img_path = os.path.join(levin_dir, 'map_edited.png')
    # yaml_path = os.path.join(levin_dir, 'map_edited.yaml')
    levin_dir = os.path.join(script_dir, '..', 'levine')
    csv_path = os.path.join(levin_dir, 'levine.csv')
    img_path = os.path.join(levin_dir, 'levine.png')
    yaml_path = os.path.join(levin_dir, 'levine.yaml')
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

    # Create single figure with map
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img, extent=[xmin, xmax, ymin, ymax], origin='upper')

    # Load waypoints CSV manually due to pandas error
    s_list, xs, ys, psi_list, kappa_list, vx_list, ax_list, w_tr_right_list, w_tr_left_list = [], [], [], [], [], [], [], [], []
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
            s_list.append(float(row['s_m']))
            xs.append(float(row['x_m']))
            ys.append(float(row['y_m']))
            psi_list.append(float(row['psi_rad']))
            kappa_list.append(float(row['kappa_radpm']))
            vx_list.append(float(row['vx_mps']))
            ax_list.append(float(row['ax_mps2']))
            w_tr_right_list.append(float(row['w_tr_right_m']))
            w_tr_left_list.append(float(row['w_tr_left_m']))
    # Plot waypoints
    ax.scatter(xs, ys, c='r', s=2, label='waypoints')
    
    # Convert lists to numpy arrays for easier operations
    xs_np = np.array(xs)
    ys_np = np.array(ys)
    psi_np = np.array(psi_list)
    kappa_np = np.array(kappa_list)
    vx_np = np.array(vx_list)
    s_np = np.array(s_list)
    
    # Draw road boundaries based on track width
    track_width = 0.75  # w_tr_right_m and w_tr_left_m values
    left_xs, left_ys = [], []
    right_xs, right_ys = [], []
    
    # Calculate road boundaries
    for i in range(len(xs)):
        # Unit normal vector
        nx = -np.sin(psi_list[i])
        ny = np.cos(psi_list[i])
        
        # Left and right track boundaries
        left_xs.append(xs[i] + track_width * nx)
        left_ys.append(ys[i] + track_width * ny)
        right_xs.append(xs[i] - track_width * nx)
        right_ys.append(ys[i] - track_width * ny)
    
    # Plot track boundaries
    ax.plot(left_xs, left_ys, 'b-', linewidth=1, alpha=0.7, label='left boundary')
    ax.plot(right_xs, right_ys, 'g-', linewidth=1, alpha=0.7, label='right boundary')
    
    # Select sample points for annotations (every N points)
    sample_step = len(xs) // 10  # Adjust to show more or fewer annotations
    
    # Annotate curvature at sample points
    for i in range(0, len(xs), sample_step):
        if i >= len(xs):
            break
        ax.annotate(f"κ={kappa_list[i]:.2f}", 
                    xy=(xs[i], ys[i]), 
                    xytext=(xs[i]+0.5, ys[i]+0.5),
                    fontsize=8,
                    arrowprops=dict(arrowstyle="->", color='purple'))
    
    # Annotate cumulative distance at sample points
    for i in range(0, len(xs), 2*sample_step):
        if i >= len(xs):
            break
        ax.annotate(f"s={s_list[i]:.1f}m", 
                    xy=(xs[i], ys[i]), 
                    xytext=(xs[i]-0.5, ys[i]-0.5),
                    fontsize=8,
                    arrowprops=dict(arrowstyle="->", color='orange'))
    
    # Draw velocity vectors (arrow length proportional to speed)
    for i in range(0, len(xs), sample_step):
        if i >= len(xs):
            break
        # Direction vector based on heading
        dx = np.cos(psi_list[i]) * vx_list[i] * 0.5  # Scale factor for visibility
        dy = np.sin(psi_list[i]) * vx_list[i] * 0.5
        ax.arrow(xs[i], ys[i], dx, dy, 
                head_width=0.2, head_length=0.3, fc='red', ec='red')
    
    # Add a custom legend entry for velocity
    ax.plot([0], [0], 'r-', label='velocity (arrow length = magnitude)')
    
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.set_title('Levine Waypoints with Curvature, Distance and Velocity')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main() 