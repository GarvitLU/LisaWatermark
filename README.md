# Video Watermarking Script

This Python script adds watermarks to videos that change position every X seconds throughout the entire video. The watermark "Created using LisaApp.in – AI-Powered Course Builder" will appear continuously and change position at regular intervals, preserving the original audio.

## Features

- Add watermarks with custom text
- Continuous watermark throughout the video
- **Automatic watermark calculation**: Number of watermarks is calculated based on video duration
- **One watermark every 10 seconds** (or custom interval) by default
- Position changes every X seconds (configurable)
- Random positioning (top-left, top-right, bottom-left, bottom-right, center)
- Preserves original audio using FFmpeg
- Semi-transparent watermark with background
- Support for multiple video formats (mp4, avi, mov, mkv, wmv)

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Usage

```bash
python video_watermark.py input_video.mp4 output_video.mp4
```

#### Optional Parameters:
- `--watermark-text`: Custom watermark text (default: "Created using LisaApp.in\nAI-Powered Course Builder")
- `--change-interval`: Interval in seconds between position changes (default: 10.0)
- `--font-size`: Font size for watermark text (default: 18)

#### Examples:

```bash
# Basic usage
python video_watermark.py AI_Powered_Visualizations.mp4 watermarked_video.mp4

# Custom watermark text
python video_watermark.py input.mp4 output.mp4 --watermark-text "Created using LisaApp.in\nAI-Powered Course Builder"

# Change position every 15 seconds
python video_watermark.py input.mp4 output.mp4 --change-interval 15.0

# Custom font size
python video_watermark.py input.mp4 output.mp4 --font-size 14
```

### API Usage (for Postman)

1. Start the API server:
```bash
python api_wrapper.py
```

2. The API will be available at `http://localhost:5000`

#### API Endpoints:

**POST /watermark**
- Upload a video file and add watermarks
- Form data:
  - `video`: Video file (required)
  - `watermark_text`: Watermark text (optional, default: "Created using LisaApp.in\nAI-Powered Course Builder")
  - `change_interval`: Interval in seconds between position changes (optional, default: 10.0)
  - `font_size`: Font size for watermark text (optional, default: 18)

**Response includes:**
- `num_watermarks`: Automatically calculated number of watermarks based on video duration
- `video_duration`: Duration of the video in seconds
- `watermarks_per_minute`: Number of watermarks per minute
- `timestamps`: Array of timestamps when watermark positions change

**GET /download/<filename>**
- Download the processed video file

**GET /health**
- Health check endpoint

#### Postman Setup:

1. Create a new POST request to `http://localhost:5000/watermark`
2. Set the request body to `form-data`
3. Add the following fields:
   - `video` (type: File) - Select your video file
   - `watermark_text` (type: Text) - Optional watermark text (use \n for line breaks)
   - `change_interval` (type: Text) - Optional interval in seconds between position changes

4. Send the request and you'll get a JSON response with the download URL

## How It Works

1. **Video Analysis**: The script reads the input video and extracts properties (fps, resolution, duration)
2. **Watermark Calculation**: Automatically calculates the number of watermarks based on video duration
   - One watermark every 10 seconds (or custom interval)
   - Minimum of 1 watermark for any video
3. **Position Changes**: Generates timestamps for watermark position changes every X seconds
4. **Random Positions**: For each position change, selects a random position from 5 possible locations
5. **Watermark Creation**: Creates a semi-transparent watermark with the specified text
6. **Frame Processing**: Processes each frame and adds watermarks continuously
7. **Audio Preservation**: Uses FFmpeg to merge the processed video with original audio
8. **Output**: Saves the processed video with watermarks and original audio

## Watermark Positions

The watermark can appear in these positions:
- Top-left corner
- Top-right corner  
- Bottom-left corner
- Bottom-right corner
- Center of the video

**Position Selection Logic:**
- Positions change every 10 seconds (or custom interval)
- **No consecutive repeats**: The same position will never appear twice in a row
- **Random selection**: Each position change randomly selects from the 5 available positions
- **Varied distribution**: All positions are used throughout the video

## Supported Video Formats

- MP4
- AVI
- MOV
- MKV
- WMV

## Requirements

- Python 3.7+
- OpenCV
- Pillow (PIL)
- NumPy
- Flask (for API usage)

## Notes

- The watermark has a semi-transparent black background for better visibility
- Each watermark appears for the specified duration (default: 3 seconds)
- The script shows progress during processing
- Output videos are saved in MP4 format 