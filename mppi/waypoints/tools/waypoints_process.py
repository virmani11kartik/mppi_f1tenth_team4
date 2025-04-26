#!/usr/bin/env python3
import argparse
import csv
import math
import datetime
import hashlib
import os
import numpy as np


def convert(input_path, output_path):
    # read input CSV with comma-separated header x_m,y_m,theta_rad,velocity_m_s
    with open(input_path, newline='') as f:
        reader = csv.DictReader(f)
        xs, ys, thetas, vels = [], [], [], []
        for row in reader:
            xs.append(float(row['x_m']))
            ys.append(float(row['y_m']))
            thetas.append(float(row['theta_rad']))
            vels.append(float(row['velocity_m_s']))

    # compute cumulative s_m (arc length)
    ss = [0.0]
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i-1]
        dy = ys[i] - ys[i-1]
        ss.append(ss[-1] + math.hypot(dx, dy))

    # Resample to approximately 0.09m spacing
    ds = 0.09
    max_s = ss[-1]
    new_ss = np.arange(0.0, max_s, ds)
    if new_ss[-1] < max_s:
        new_ss = np.append(new_ss, max_s)
    # linear interpolation for x, y, velocity
    xs = np.interp(new_ss, ss, xs)
    ys = np.interp(new_ss, ss, ys)
    # unwrap theta for smooth interpolation then wrap back
    thetas_unwrapped = np.unwrap(thetas)
    thetas_interp = np.interp(new_ss, ss, thetas_unwrapped)
    # compute curvature kappa = dtheta/ds
    kappas = np.gradient(thetas_interp, new_ss)
    thetas = (thetas_interp + np.pi) % (2 * np.pi) - np.pi
    vels = np.interp(new_ss, ss, vels)
    ss = new_ss.tolist()

    # build body lines with semicolon-separated values
    body_lines = []
    for s, x, y, th, k, v in zip(ss, xs, ys, thetas, kappas, vels):
        line = f"{s:.18e};{x:.18e};{y:.18e};{th:.18e};{k:.18e};{v:.18e};{0.0:.18e};{0.75:.18e};{0.75:.18e}"
        body_lines.append(line)

    # compute MD5 of content
    content_bytes = "\n".join(body_lines).encode('utf-8')
    md5sum = hashlib.md5(content_bytes).hexdigest()

    # write output file with headers
    with open(output_path, 'w', newline='') as f:
        f.write('# ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')
        f.write('# ' + md5sum + '\n')
        f.write('# s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2; w_tr_right_m; w_tr_left_m\n')
        for line in body_lines:
            f.write(line + '\n')


def main():
    script_dir = os.path.dirname(__file__)
    default_input = os.path.normpath(os.path.join(
        script_dir, '..','..','..','..','slam-and-pure-pursuit-team4','pure_pursuit','tools','waypoints','map1_edited_waypoints.csv'))
    default_output = os.path.normpath(os.path.join(
        script_dir, '..','race','map1_edited_waypoints.csv'))
    parser = argparse.ArgumentParser(description='Convert map3_levine_waypoints.csv to levine.csv format')
    parser.add_argument('-i', '--input', default=default_input, help='input CSV path (comma-separated)')
    parser.add_argument('-o', '--output', default=default_output, help='output CSV path (semicolon-separated)')
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == '__main__':
    main()
