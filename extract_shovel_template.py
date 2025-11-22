#!/usr/bin/env python3
"""
Helper script to extract just the shovel icon portion from a full logo template.
This creates a second template for use with --second-template and --require-both.
"""

import cv2
import argparse
import sys
import numpy as np


def extract_shovel_region(template_path: str, output_path: str, 
                          x: int = None, y: int = None, w: int = None, h: int = None):
    """
    Extract the shovel region from a template image.
    
    Args:
        template_path: Path to full template image
        output_path: Path to save extracted shovel region
        x, y, w, h: Coordinates and dimensions of shovel region (optional, will prompt if not provided)
    """
    img = cv2.imread(template_path)
    if img is None:
        print(f"Error: Could not load image from {template_path}")
        return False
    
    h_img, w_img = img.shape[:2]
    print(f"Template image size: {w_img}x{h_img}")
    
    # If coordinates not provided, show image and let user select
    if x is None or y is None or w is None or h is None:
        print("\nPlease select the shovel region in the window that opens.")
        print("Click and drag to select the area, then press SPACE or ENTER to confirm.")
        print("Press ESC to cancel.")
        
        # Create a copy for selection
        img_copy = img.copy()
        drawing = False
        ix, iy = -1, -1
        fx, fy = -1, -1
        
        def mouse_callback(event, mouse_x, mouse_y, flags, param):
            nonlocal drawing, ix, iy, fx, fy, img_copy
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                ix, iy = mouse_x, mouse_y
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    img_copy = img.copy()
                    cv2.rectangle(img_copy, (ix, iy), (mouse_x, mouse_y), (0, 255, 0), 2)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                fx, fy = mouse_x, mouse_y
                cv2.rectangle(img_copy, (ix, iy), (fx, fy), (0, 255, 0), 2)
        
        cv2.namedWindow('Select Shovel Region')
        cv2.setMouseCallback('Select Shovel Region', mouse_callback)
        
        while True:
            cv2.imshow('Select Shovel Region', img_copy)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("Cancelled.")
                cv2.destroyAllWindows()
                return False
            elif key == 13 or key == 32:  # ENTER or SPACE
                if ix != -1 and iy != -1 and fx != -1 and fy != -1:
                    x = min(ix, fx)
                    y = min(iy, fy)
                    w = abs(fx - ix)
                    h = abs(fy - iy)
                    cv2.destroyAllWindows()
                    break
        
        if w == 0 or h == 0:
            print("Error: Invalid selection")
            return False
    
    # Validate coordinates
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    w = max(1, min(w, w_img - x))
    h = max(1, min(h, h_img - y))
    
    # Extract region
    shovel_region = img[y:y+h, x:x+w]
    
    print(f"\nExtracted region: x={x}, y={y}, w={w}, h={h}")
    print(f"Shovel template size: {w}x{h}")
    
    # Save extracted region
    cv2.imwrite(output_path, shovel_region)
    print(f"Saved shovel template to: {output_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Extract just the shovel icon from a full logo template",
        epilog="""
Examples:
  # Interactive selection (recommended)
  python extract_shovel_template.py target.png shovel.png
  
  # With coordinates
  python extract_shovel_template.py target.png shovel.png --x 100 --y 20 --w 50 --h 50
        """
    )
    
    parser.add_argument('template', help='Input template image (full logo)')
    parser.add_argument('output', help='Output image file for shovel template')
    parser.add_argument('--x', type=int, help='X coordinate of shovel region')
    parser.add_argument('--y', type=int, help='Y coordinate of shovel region')
    parser.add_argument('--w', type=int, help='Width of shovel region')
    parser.add_argument('--h', type=int, help='Height of shovel region')
    
    args = parser.parse_args()
    
    if not extract_shovel_region(args.template, args.output, args.x, args.y, args.w, args.h):
        sys.exit(1)
    
    print(f"\nNow you can use both templates:")
    print(f"  python video_logo_scanner.py scan --folder ./videos --template {args.template} --second-template {args.output} --require-both")


if __name__ == "__main__":
    main()

