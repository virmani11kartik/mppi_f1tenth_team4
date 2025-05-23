#!/usr/bin/env python3
import argparse
import csv
import math
import datetime
import hashlib
import os


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

    # build body lines with semicolon-separated values
    body_lines = []
    for s, x, y, th, v in zip(ss, xs, ys, thetas, vels):
        line = f"{s:.18e};{x:.18e};{y:.18e};{th:.18e};{0.0:.18e};{v:.18e};{0.0:.18e};{0.0:.18e};{0.0:.18e}"
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
        script_dir, '..','..','..','..','slam-and-pure-pursuit-team4','pure_pursuit','tools','waypoints','map_edited_waypoints.csv'))
    default_output = os.path.normpath(os.path.join(
        script_dir, '..','map_edited_waypoints.csv'))
    parser = argparse.ArgumentParser(description='Convert map3_levine_waypoints.csv to levine.csv format')
    parser.add_argument('-i', '--input', default=default_input, help='input CSV path (comma-separated)')
    parser.add_argument('-o', '--output', default=default_output, help='output CSV path (semicolon-separated)')
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == '__main__':
    main()
