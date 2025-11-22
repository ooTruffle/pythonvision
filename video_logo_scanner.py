#!/usr/bin/env python3
"""
Video Logo Scanner - Scans videos for a specific logo and allows bulk file operations.
"""

import os
import cv2
import numpy as np
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import json


class VideoLogoScanner:
    """Scans videos for a specific logo using template matching."""
    
    def __init__(self, template_path: Optional[str] = None, threshold: float = 0.7,
                 corners: List[str] = None, corner_size: float = 0.2,
                 second_template_path: Optional[str] = None, require_both: bool = False):
        """
        Initialize the scanner.
        
        Args:
            template_path: Path to the logo template image. If None, will try to use screenshot.
            threshold: Matching threshold (0.0 to 1.0). Higher = stricter matching.
            corners: List of corners to check: 'top-left', 'top-right', 'bottom-left', 'bottom-right'
                    If None, checks all corners.
            corner_size: Fraction of frame dimensions to use for corner region (0.0-1.0, default: 0.2 = 20%)
            second_template_path: Path to a second template (e.g., shovel icon). If provided with require_both=True,
                                 both templates must be found.
            require_both: If True and second_template_path is provided, require both templates to be found.
        """
        self.threshold = threshold
        self.template = None
        self.second_template = None
        self.require_both = require_both
        self.corner_size = corner_size
        
        # Default to checking all corners if none specified
        if corners is None:
            corners = ['top-left', 'top-right', 'bottom-left', 'bottom-right']
        self.corners = corners
        
        if template_path and os.path.exists(template_path):
            self.template = cv2.imread(template_path, cv2.IMREAD_COLOR)
            if self.template is None:
                raise ValueError(f"Could not load template image from {template_path}")
            # Convert to grayscale for template matching
            self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        else:
            print("Warning: No template image provided. Please provide a screenshot/template image.")
            print("You can extract a frame from a video that contains the logo and use it as template.")
        
        # Load second template if provided
        if second_template_path and os.path.exists(second_template_path):
            self.second_template = cv2.imread(second_template_path, cv2.IMREAD_COLOR)
            if self.second_template is None:
                raise ValueError(f"Could not load second template image from {second_template_path}")
            self.second_template_gray = cv2.cvtColor(self.second_template, cv2.COLOR_BGR2GRAY)
        elif require_both:
            print("Warning: require_both is True but no second template provided.")
    
    def extract_template_from_video(self, video_path: str, frame_number: int = 0) -> Optional[np.ndarray]:
        """Extract a frame from a video to use as template."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return frame
        return None
    
    def extract_corner_region(self, frame: np.ndarray, corner: str) -> Optional[np.ndarray]:
        """
        Extract a corner region from a frame.
        
        Args:
            frame: Full frame image
            corner: Corner name: 'top-left', 'top-right', 'bottom-left', 'bottom-right'
            
        Returns:
            Corner region as numpy array, or None if invalid corner
        """
        h, w = frame.shape[:2]
        corner_w = int(w * self.corner_size)
        corner_h = int(h * self.corner_size)
        
        if corner == 'top-left':
            return frame[0:corner_h, 0:corner_w]
        elif corner == 'top-right':
            return frame[0:corner_h, w-corner_w:w]
        elif corner == 'bottom-left':
            return frame[h-corner_h:h, 0:corner_w]
        elif corner == 'bottom-right':
            return frame[h-corner_h:h, w-corner_w:w]
        else:
            return None
    
    def detect_logo_in_frame(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Detect logo in a single frame using template matching.
        Only searches in the specified corner regions.
        If require_both is True, both templates must be found.
        
        Returns:
            (found, confidence): Tuple of (bool, float) indicating if logo was found and confidence.
        """
        if self.template is None or self.template_gray is None:
            return False, 0.0
        
        max_confidence = 0.0
        found = False
        
        # Check each specified corner
        for corner in self.corners:
            corner_region = self.extract_corner_region(frame, corner)
            if corner_region is None:
                continue
            
            # Convert corner region to grayscale
            corner_gray = cv2.cvtColor(corner_region, cv2.COLOR_BGR2GRAY)
            
            # Check first template
            if corner_gray.shape[0] < self.template_gray.shape[0] or \
               corner_gray.shape[1] < self.template_gray.shape[1]:
                # Template too large for this corner, skip
                continue
            
            # Perform template matching on corner region for first template
            result = cv2.matchTemplate(corner_gray, self.template_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            first_found = max_val >= self.threshold
            first_confidence = max_val
            
            # If require_both is True, check second template
            if self.require_both and self.second_template is not None:
                if corner_gray.shape[0] < self.second_template_gray.shape[0] or \
                   corner_gray.shape[1] < self.second_template_gray.shape[1]:
                    # Second template too large, skip
                    continue
                
                # Check second template
                result2 = cv2.matchTemplate(corner_gray, self.second_template_gray, cv2.TM_CCOEFF_NORMED)
                _, max_val2, _, max_loc2 = cv2.minMaxLoc(result2)
                
                second_found = max_val2 >= self.threshold
                second_confidence = max_val2
                
                # Both must be found
                if first_found and second_found:
                    found = True
                    # Use average confidence of both matches
                    combined_confidence = (first_confidence + second_confidence) / 2.0
                    max_confidence = max(max_confidence, combined_confidence)
            else:
                # Only first template required
                if first_found:
                    found = True
                    max_confidence = max(max_confidence, first_confidence)
        
        return found, max_confidence
    
    def scan_video(self, video_path: str, sample_frames: int = 10) -> Tuple[bool, float, int]:
        """
        Scan a video file for the logo.
        
        Args:
            video_path: Path to video file
            sample_frames: Number of frames to sample from the video
            
        Returns:
            (found, max_confidence, frames_checked): Tuple indicating if logo was found
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, 0.0, 0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames == 0:
            cap.release()
            return False, 0.0, 0
        
        # Sample frames evenly throughout the video
        frame_indices = np.linspace(0, total_frames - 1, min(sample_frames, total_frames), dtype=int)
        
        max_confidence = 0.0
        logo_found = False
        frames_checked = 0
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            frames_checked += 1
            found, confidence = self.detect_logo_in_frame(frame)
            
            if found:
                logo_found = True
                max_confidence = max(max_confidence, confidence)
        
        cap.release()
        return logo_found, max_confidence, frames_checked
    
    def scan_folder(self, folder_path: str, extensions: List[str] = None, 
                   sample_frames: int = 10, verbose: bool = True,
                   corners: List[str] = None, corner_size: float = None) -> List[Tuple[str, float, int]]:
        """
        Scan all videos in a folder.
        
        Args:
            folder_path: Path to folder containing videos
            extensions: List of video file extensions to scan (default: common video formats)
            sample_frames: Number of frames to sample per video
            verbose: Print progress information
            corners: Override corners to check (optional)
            corner_size: Override corner size fraction (optional)
            
        Returns:
            List of tuples: (video_path, max_confidence, frames_checked) for videos containing the logo
        """
        # Allow overriding corner settings per scan
        original_corners = self.corners
        original_size = self.corner_size
        
        if corners is not None:
            self.corners = corners
        if corner_size is not None:
            self.corner_size = corner_size
        
        if extensions is None:
            extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.webm']
        
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Folder does not exist: {folder_path}")
        
        matching_videos = []
        all_videos = []
        
        # Collect all video files
        for ext in extensions:
            all_videos.extend(folder.glob(f'*{ext}'))
            all_videos.extend(folder.glob(f'*{ext.upper()}'))
        
        total = len(all_videos)
        if verbose:
            print(f"Found {total} video file(s) to scan...")
        
        for idx, video_path in enumerate(all_videos, 1):
            if verbose:
                print(f"[{idx}/{total}] Scanning: {video_path.name}...", end=' ', flush=True)
            
            try:
                found, confidence, frames_checked = self.scan_video(str(video_path), sample_frames)
                
                if found:
                    matching_videos.append((str(video_path), confidence, frames_checked))
                    if verbose:
                        print(f"✓ FOUND (confidence: {confidence:.2f})")
                else:
                    if verbose:
                        print(f"✗ Not found")
            except Exception as e:
                if verbose:
                    print(f"✗ Error: {str(e)}")
        
        # Restore original corner settings
        self.corners = original_corners
        self.corner_size = original_size
        
        return matching_videos


def save_results(results: List[Tuple[str, float, int]], output_file: str = "scan_results.json"):
    """Save scan results to a JSON file."""
    data = {
        "total_matches": len(results),
        "videos": [
            {
                "path": path,
                "confidence": float(conf),
                "frames_checked": frames
            }
            for path, conf, frames in results
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


def load_results(input_file: str = "scan_results.json") -> List[str]:
    """Load video paths from a results file."""
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    return [video["path"] for video in data["videos"]]


def move_files(file_paths: List[str], destination: str, create_subdirs: bool = False):
    """Move files to destination folder."""
    dest_path = Path(destination)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    moved = 0
    errors = 0
    
    for file_path in file_paths:
        src = Path(file_path)
        if not src.exists():
            print(f"Warning: File not found: {file_path}")
            errors += 1
            continue
        
        if create_subdirs:
            # Preserve directory structure
            rel_path = src.relative_to(src.parents[len(src.parents) - 2])
            dest_file = dest_path / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            dest_file = dest_path / src.name
        
        try:
            shutil.move(str(src), str(dest_file))
            print(f"Moved: {src.name} -> {dest_file}")
            moved += 1
        except Exception as e:
            print(f"Error moving {src.name}: {str(e)}")
            errors += 1
    
    print(f"\nMoved {moved} file(s), {errors} error(s)")


def main():
    parser = argparse.ArgumentParser(
        description="Scan videos for a logo and manage matching files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan videos with a template image (checks all corners)
  python video_logo_scanner.py scan --folder ./videos --template logo.png
  
  # Scan only top-right corner (where logo typically appears)
  python video_logo_scanner.py scan --folder ./videos --template logo.png --corners top-right
  
  # Require both text and shovel icon (use full logo template + shovel template)
  python video_logo_scanner.py scan --folder ./videos --template logo.png --second-template shovel.png --require-both
  
  # Scan with custom threshold and corner size
  python video_logo_scanner.py scan --folder ./videos --template logo.png --threshold 0.8 --corner-size 0.25
  
  # Move matching videos to a folder
  python video_logo_scanner.py move --results scan_results.json --destination ./matched_videos
  
  # Move specific files
  python video_logo_scanner.py move --files video1.mp4 video2.mp4 --destination ./matched_videos
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Scan videos for logo')
    scan_parser.add_argument('--folder', required=True, help='Folder containing videos')
    scan_parser.add_argument('--template', help='Path to logo template image (screenshot)')
    scan_parser.add_argument('--threshold', type=float, default=0.7, 
                            help='Matching threshold (0.0-1.0, default: 0.7)')
    scan_parser.add_argument('--sample-frames', type=int, default=10,
                            help='Number of frames to sample per video (default: 10)')
    scan_parser.add_argument('--output', default='scan_results.json',
                            help='Output file for results (default: scan_results.json)')
    scan_parser.add_argument('--quiet', action='store_true', help='Suppress progress output')
    scan_parser.add_argument('--corners', nargs='+', 
                            choices=['top-left', 'top-right', 'bottom-left', 'bottom-right'],
                            help='Corners to check (default: all corners). Example: --corners top-right')
    scan_parser.add_argument('--corner-size', type=float, default=0.2,
                            help='Fraction of frame size for corner region (0.0-1.0, default: 0.2 = 20%%)')
    scan_parser.add_argument('--second-template', 
                            help='Path to second template image (e.g., shovel icon). Use with --require-both.')
    scan_parser.add_argument('--require-both', action='store_true',
                            help='Require both templates to be found (useful for text + icon detection)')
    
    # Move command
    move_parser = subparsers.add_parser('move', help='Move matching videos')
    move_parser.add_argument('--results', help='JSON file with scan results')
    move_parser.add_argument('--files', nargs='+', help='Specific files to move')
    move_parser.add_argument('--destination', required=True, help='Destination folder')
    move_parser.add_argument('--preserve-structure', action='store_true',
                            help='Preserve directory structure when moving')
    
    args = parser.parse_args()
    
    if args.command == 'scan':
        if not args.template:
            print("Error: --template is required for scanning")
            print("Please provide a screenshot or template image of the logo.")
            return
        
        scanner = VideoLogoScanner(
            template_path=args.template, 
            threshold=args.threshold,
            corners=args.corners,
            corner_size=args.corner_size,
            second_template_path=args.second_template,
            require_both=args.require_both
        )
        
        print(f"Scanning videos in: {args.folder}")
        print(f"Template: {args.template}")
        if args.second_template:
            print(f"Second template: {args.second_template}")
        if args.require_both:
            print(f"Mode: Requiring BOTH templates to be found")
        print(f"Threshold: {args.threshold}")
        print(f"Sample frames per video: {args.sample_frames}")
        print(f"Corners to check: {', '.join(scanner.corners)}")
        print(f"Corner region size: {args.corner_size * 100:.0f}% of frame")
        print("-" * 60)
        
        results = scanner.scan_folder(
            args.folder, 
            sample_frames=args.sample_frames,
            verbose=not args.quiet
        )
        
        print("-" * 60)
        print(f"\nFound {len(results)} video(s) containing the logo:")
        for path, conf, frames in results:
            print(f"  - {Path(path).name} (confidence: {conf:.2f}, frames checked: {frames})")
        
        if results:
            save_results(results, args.output)
            print(f"\nTo move these files, run:")
            print(f"  python video_logo_scanner.py move --results {args.output} --destination <folder>")
    
    elif args.command == 'move':
        if args.results:
            file_paths = load_results(args.results)
        elif args.files:
            file_paths = args.files
        else:
            print("Error: Either --results or --files must be provided")
            return
        
        print(f"Moving {len(file_paths)} file(s) to: {args.destination}")
        move_files(file_paths, args.destination, create_subdirs=args.preserve_structure)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

