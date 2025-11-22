#!/usr/bin/env python3
"""
Helper script to extract a frame from a video for use as a template.
"""

import cv2
import argparse
import sys


def extract_frame(video_path: str, output_path: str, frame_number: int = 0):
    """Extract a frame from a video."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_number >= total_frames:
        print(f"Error: Frame {frame_number} is out of range. Video has {total_frames} frames.")
        cap.release()
        return False
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Could not read frame {frame_number}")
        return False
    
    cv2.imwrite(output_path, frame)
    print(f"Extracted frame {frame_number} to: {output_path}")
    print(f"Total frames in video: {total_frames}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Extract a frame from a video for use as a template",
        epilog="""
Examples:
  # Extract first frame
  python extract_frame.py video.mp4 logo.png
  
  # Extract frame at specific position
  python extract_frame.py video.mp4 logo.png --frame 100
        """
    )
    
    parser.add_argument('video', help='Input video file')
    parser.add_argument('output', help='Output image file (PNG or JPG)')
    parser.add_argument('--frame', type=int, default=0,
                       help='Frame number to extract (default: 0)')
    
    args = parser.parse_args()
    
    if not extract_frame(args.video, args.output, args.frame):
        sys.exit(1)


if __name__ == "__main__":
    main()

