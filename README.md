# Video Logo Scanner

A Python script that scans videos in a folder to detect a specific logo (like the ooTruffle logo) and allows bulk moving of matching files.

## Features

- Scans all videos in a folder for a specific logo
- **Corner-specific detection**: Only searches in corner regions where logos typically appear (faster and more accurate)
- Uses template matching to detect the logo in video frames
- Saves results to JSON for easy management
- Bulk move matching videos to a destination folder
- Configurable matching threshold, frame sampling, and corner regions

## Installation

1. Install Python 3.7 or higher
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Prepare Template Image(s)

First, you need a screenshot or image of the logo you want to detect. You can:
- Take a screenshot from a video that contains the logo
- Extract a frame from a video using the script
- Use any image file containing the logo

**For better accuracy (text + icon detection):**
If your logo has multiple elements (like text + an icon), you can use dual-template matching:
1. Use your full logo as the main template
2. Extract just the icon portion (e.g., shovel) as a second template using `extract_shovel_template.py`
3. Use `--require-both` to ensure both elements are present

### Step 2: Scan Videos

Scan all videos in a folder:
```bash
python video_logo_scanner.py scan --folder ./videos --template logo.png
```

Options:
- `--folder`: Folder containing videos to scan
- `--template`: Path to the logo template image (required)
- `--threshold`: Matching threshold (0.0-1.0, default: 0.7). Higher = stricter matching
- `--sample-frames`: Number of frames to sample per video (default: 10)
- `--output`: Output JSON file for results (default: scan_results.json)
- `--quiet`: Suppress progress output
- `--corners`: Which corners to check: `top-left`, `top-right`, `bottom-left`, `bottom-right` (default: all corners)
- `--corner-size`: Fraction of frame size for corner region (0.0-1.0, default: 0.2 = 20%)
- `--second-template`: Path to a second template image (e.g., just the shovel icon)
- `--require-both`: Require BOTH templates to be found (prevents false positives from text-only matches)

Examples:
```bash
# Scan only top-right corner (where logo typically appears in kill feed)
python video_logo_scanner.py scan --folder ./videos --template logo.png --corners top-right

# Require both text and shovel icon (prevents false positives)
python video_logo_scanner.py scan --folder ./videos --template logo.png --second-template shovel.png --require-both

# Scan with custom threshold and corner size
python video_logo_scanner.py scan --folder ./videos --template logo.png --threshold 0.8 --sample-frames 20 --corner-size 0.25

# Scan multiple specific corners
python video_logo_scanner.py scan --folder ./videos --template logo.png --corners top-right top-left
```

### Step 3: Move Matching Videos

After scanning, move all matching videos:
```bash
python video_logo_scanner.py move --results scan_results.json --destination ./matched_videos
```

Or move specific files:
```bash
python video_logo_scanner.py move --files video1.mp4 video2.mp4 --destination ./matched_videos
```

Options:
- `--results`: JSON file with scan results
- `--files`: Specific files to move (space-separated)
- `--destination`: Destination folder (required)
- `--preserve-structure`: Preserve directory structure when moving

## How It Works

1. **Corner Region Extraction**: Only searches in specified corner regions (e.g., top-right for kill feed logos)
2. **Template Matching**: Uses OpenCV's template matching to find the logo in corner regions
3. **Frame Sampling**: Instead of checking every frame, it samples frames evenly throughout each video
4. **Confidence Scoring**: Each match is scored by confidence (0.0 to 1.0)
5. **Results Storage**: Matching videos are saved to a JSON file with their paths and confidence scores

**Why corner detection?** Logos in videos (like player names in kill feeds) typically appear in fixed corner positions. By only searching corners, the script is:
- **Faster**: Searches ~20% of each frame instead of 100%
- **More Accurate**: Reduces false positives from similar patterns elsewhere in the frame
- **More Reliable**: Focuses on where the logo actually appears

## Tips

- **Better Template**: Use a clear, high-quality screenshot of just the logo for best results
- **Corner Selection**: 
  - For kill feed logos (like ooTruffle), use `--corners top-right`
  - If unsure, scan all corners first, then narrow down based on results
- **Corner Size**: 
  - Default 20% works for most cases
  - Increase to 25-30% if logo is larger or positioned further from corner
  - Decrease to 15% if logo is very small and close to corner edge
- **Threshold Tuning**: 
  - Lower threshold (0.5-0.6): More matches, but may include false positives
  - Higher threshold (0.8-0.9): Fewer matches, but more accurate
- **Frame Sampling**: More frames = more thorough but slower scanning
- **Video Formats**: Supports common formats: .mp4, .avi, .mov, .mkv, .flv, .wmv, .m4v, .webm

## Example Workflow

```bash
# 1. Extract a frame from a video to create template (optional)
python extract_frame.py video_with_logo.mp4 logo_template.png --frame 100

# 2a. (Optional) Extract just the shovel icon for dual-template matching
python extract_shovel_template.py logo_template.png shovel_template.png

# 2b. Scan videos (checking only top-right corner where logo appears)
# Option A: Single template (simpler, but may match text-only)
python video_logo_scanner.py scan --folder ./my_videos --template logo_template.png --corners top-right

# Option B: Dual template (more accurate, requires both text and shovel)
python video_logo_scanner.py scan --folder ./my_videos --template logo_template.png --second-template shovel_template.png --require-both --corners top-right

# 3. Review results in scan_results.json

# 4. Move matching videos
python video_logo_scanner.py move --results scan_results.json --destination ./videos_with_logo
```

## Troubleshooting

- **No matches found**: Try lowering the threshold or using a better template image
- **Too many false positives**: 
  - Use `--require-both` with a second template to require both text and icon
  - Increase the threshold
  - Improve the template quality
- **Finding text but not icon**: Use `--second-template` and `--require-both` to require both elements
- **Slow scanning**: Reduce the `--sample-frames` value
- **Template not loading**: Ensure the image file exists and is in a supported format (PNG, JPG, etc.)

