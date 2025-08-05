# Video Watermarking API

A FastAPI-based service for adding watermarks to videos with dynamic position changes. This service automatically uploads processed videos to AWS S3.

## Features

- **Dynamic Watermark Positioning**: Watermarks change position every specified interval (default: 10 seconds)
- **Fixed Position Option**: Option to use a fixed watermark position
- **Multi-line Text Support**: Support for multi-line watermark text
- **S3 Integration**: Automatic upload to AWS S3 with configurable bucket
- **Video Format Support**: Supports MP4, AVI, MOV, MKV, WMV formats
- **Audio Preservation**: Maintains original audio in processed videos
- **Health Monitoring**: Built-in health check and S3 connectivity testing

## FastAPI Conversion

This service has been converted from Flask to FastAPI for better compatibility with Modal and improved performance. Key changes include:

- **Async/Await Support**: All endpoints are now async for better performance
- **Automatic API Documentation**: FastAPI provides automatic OpenAPI/Swagger documentation
- **Better Type Safety**: Enhanced type checking and validation
- **CORS Middleware**: Built-in CORS support for cross-origin requests
- **Improved Error Handling**: Better HTTP exception handling

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables (create a `.env` file):
```env
AWS_S3_BUCKET_NAME=your-s3-bucket-name
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
```

## Usage

### Starting the Server

```bash
python api_wrapper.py
```

Or using uvicorn directly:
```bash
uvicorn api_wrapper:app --host 0.0.0.0 --port 5001
```

### API Endpoints

#### 1. Add Watermark (`POST /watermark`)

Add watermarks to a video file.

**Parameters:**
- `video` (file): Video file to process
- `watermark_text` (string, optional): Text for watermark (default: "Created using LisaApp.in\nAI-Powered Course Builder")
- `change_interval` (float, optional): Interval in seconds between position changes (default: 10.0)
- `font_size` (int, optional): Font size for watermark text (default: 18)
- `fixed_position` (string, optional): Fixed position ('top-left', 'top-right', 'bottom-left', 'bottom-right', 'center')

**Example Request:**
```bash
curl -X POST "http://localhost:5001/watermark" \
  -H "Content-Type: multipart/form-data" \
  -F "video=@input_video.mp4" \
  -F "watermark_text=Your Custom Text" \
  -F "change_interval=5.0" \
  -F "font_size=20" \
  -F "fixed_position=bottom-right"
```

**Response:**
```json
{
  "success": true,
  "message": "Video watermarked successfully and uploaded to S3",
  "input_file": "input_video.mp4",
  "output_file": "watermarked_input_video.mp4",
  "s3_url": "https://bucket.s3.region.amazonaws.com/watermarked-videos/watermarked_input_video.mp4",
  "download_url": "https://bucket.s3.region.amazonaws.com/watermarked-videos/watermarked_input_video.mp4",
  "parameters": {
    "watermark_text": "Your Custom Text",
    "change_interval": 5.0,
    "font_size": 20,
    "fixed_position": "bottom-right",
    "video_duration": 120.5,
    "num_watermarks": 25,
    "watermarks_per_minute": 12.0,
    "timestamps": [0.0, 5.0, 10.0, ...]
  }
}
```

#### 2. Health Check (`GET /health`)

Check service health and S3 configuration status.

**Response:**
```json
{
  "status": "healthy",
  "message": "Video watermarking API is running",
  "s3_status": "configured",
  "s3_bucket": "your-bucket-name"
}
```

#### 3. Test S3 (`GET /test-s3`)

Test S3 connectivity and bucket access.

**Response:**
```json
{
  "success": true,
  "message": "S3 connection successful",
  "bucket": "your-bucket-name",
  "region": "us-east-1",
  "access_key_id": "AKIA..."
}
```

#### 4. Download File (`GET /download/{filename}`)

Download processed video files (fallback when S3 is not configured).

#### 5. Check Video (`GET /check-video/{filename}`)

Check video file properties using ffprobe.

## Watermark Positioning

The service supports two positioning modes:

### Dynamic Positioning (Default)
- Watermarks change position every specified interval
- 5 possible positions: corners and center
- No consecutive repeats (same position never appears twice in a row)
- Random selection ensures varied distribution

### Fixed Positioning
- Use a single position throughout the video
- Options: 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'center'

## Testing

Run the test script to verify the service:

```bash
python test_fastapi.py
```

## API Documentation

FastAPI automatically generates interactive API documentation. Visit:
- Swagger UI: `http://localhost:5001/docs`
- ReDoc: `http://localhost:5001/redoc`

## Modal Compatibility

This FastAPI version is designed to be more compatible with Modal deployment:

- **Async Support**: Better handling of concurrent requests
- **Type Safety**: Enhanced validation and error handling
- **Performance**: Improved response times and resource usage
- **Documentation**: Automatic API documentation generation

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `AWS_S3_BUCKET_NAME` | S3 bucket name for uploads | Yes |
| `AWS_REGION` | AWS region (default: us-east-1) | No |
| `AWS_ACCESS_KEY_ID` | AWS access key | Yes |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | Yes |

## Dependencies

- `fastapi`: Web framework
- `uvicorn[standard]`: ASGI server
- `python-multipart`: File upload support
- `opencv-python`: Video processing
- `Pillow`: Image processing
- `numpy`: Numerical operations
- `boto3`: AWS S3 integration
- `python-dotenv`: Environment variable management 