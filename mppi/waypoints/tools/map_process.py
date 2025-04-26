#!/usr/bin/env python3

import argparse
import numpy as np
from PIL import Image

def convert_to_trinary(input_path, output_path, thresholds=(75, 180)):
    """
    Convert a grayscale image to trinary mode (3 values only).
    
    Args:
        input_path: Path to input grayscale image
        output_path: Path to save the trinary image
        thresholds: Tuple of thresholds (low, high) to determine the three values
    """
    # Open the grayscale image
    img = Image.open(input_path).convert('L')
    img_array = np.array(img)
    
    # Create a trinary image
    trinary_array = np.zeros_like(img_array)
    
    # Apply thresholds
    low_threshold, high_threshold = thresholds
    
    # 0 (black) for values below low_threshold
    # 127 (gray) for values between low_threshold and high_threshold
    # 255 (white) for values above high_threshold
    trinary_array[img_array < low_threshold] = 0
    trinary_array[(img_array >= low_threshold) & (img_array < high_threshold)] = 127
    trinary_array[img_array >= high_threshold] = 255
    
    # Convert back to image and save
    trinary_img = Image.fromarray(trinary_array.astype(np.uint8))
    trinary_img.save(output_path)
    
    print(f"Converted image saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert grayscale image to trinary mode')
    parser.add_argument('input', help='Input grayscale image path')
    parser.add_argument('output', help='Output trinary image path')
    parser.add_argument('--low', type=int, default=75, help='Low threshold (default: 75)')
    parser.add_argument('--high', type=int, default=180, help='High threshold (default: 180)')
    
    args = parser.parse_args()
    
    convert_to_trinary(args.input, args.output, (args.low, args.high))

if __name__ == "__main__":
    main()
