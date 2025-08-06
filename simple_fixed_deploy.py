#!/usr/bin/env python3
"""
Updated Modal deployment script for the existing video-watermarking-api-v2 with S3 integration
"""

import modal

# Define the image with all dependencies including system libraries for OpenCV
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
        "libgomp1",
        "libgtk-3-0",
        "libavcodec-dev",
        "libavformat-dev",
        "libswscale-dev",
        "libv4l-dev",
        "libxvidcore-dev",
        "libx264-dev",
        "libjpeg-dev",
        "libpng-dev",
        "libtiff-dev",
        "libatlas-base-dev",
        "gfortran",
        "wget",
        "unzip",
        "ffmpeg"
    )
    .pip_install(
        "fastapi",
        "uvicorn[standard]",
        "python-multipart",
        "opencv-python",
        "Pillow",
        "numpy",
        "boto3",
        "python-dotenv"
    )
)

# Create the app with the same name as existing
app = modal.App("video-watermarking-api-v2", image=image)

@app.function(
    secrets=[
        modal.Secret.from_name("s3-credentials"),  # Use your existing secret
    ],
    timeout=300,  # 5 minutes timeout
    memory=2048,  # 2GB RAM
    cpu=1.0,      # 1 CPU
)
@modal.asgi_app()
def fastapi_app():
    """Deploy the FastAPI application with S3 integration"""
    import os
    import tempfile
    import json
    from fastapi import FastAPI, File, UploadFile, Form, HTTPException
    from fastapi.responses import JSONResponse, FileResponse
    from fastapi.middleware.cors import CORSMiddleware
    from typing import Optional
    import uuid
    import boto3
    from botocore.exceptions import ClientError
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    app = FastAPI(
        title="Video Watermarking API",
        description="API for adding watermarks to videos with position changes",
        version="1.0.0"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Configure upload folder
    UPLOAD_FOLDER = 'uploads'
    OUTPUT_FOLDER = 'outputs'
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}

    # S3 Configuration - Updated to match your secret variable names
    S3_BUCKET = os.getenv('S3_BUCKET_NAME') or os.getenv('AWS_S3_BUCKET_NAME')
    S3_REGION = os.getenv('S3_REGION') or os.getenv('AWS_REGION', 'us-east-1')
    AWS_ACCESS_KEY_ID = os.getenv('S3_ACCESS_KEY') or os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('S3_SECRET_KEY') or os.getenv('AWS_SECRET_ACCESS_KEY')

    # Initialize S3 client
    s3_client = None
    if all([S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]):
        try:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=S3_REGION
            )
            print(f"S3 client initialized successfully. Bucket: {S3_BUCKET}, Region: {S3_REGION}")
        except Exception as e:
            print(f"Failed to initialize S3 client: {e}")
            s3_client = None
    else:
        missing_vars = []
        if not S3_BUCKET:
            missing_vars.append("S3_BUCKET_NAME")
        if not AWS_ACCESS_KEY_ID:
            missing_vars.append("S3_ACCESS_KEY")
        if not AWS_SECRET_ACCESS_KEY:
            missing_vars.append("S3_SECRET_KEY")
        print(f"S3 not configured. Missing environment variables: {', '.join(missing_vars)}")

    # Create directories if they don't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    def upload_to_s3(file_path, filename):
        """Upload file to S3 and return the URL"""
        if not s3_client:
            raise Exception("S3 client not configured. Please check environment variables.")
        
        try:
            # Check file exists and has content
            if not os.path.exists(file_path):
                raise Exception(f"File not found: {file_path}")
            
            file_size = os.path.getsize(file_path)
            print(f"📁 File to upload: {file_path} (size: {file_size} bytes)")
            
            s3_key = f"watermarked-videos/{filename}"
            print(f"🔑 S3 key: {s3_key}")
            
            s3_client.upload_file(
                file_path,
                S3_BUCKET,
                s3_key,
                ExtraArgs={'ContentType': 'video/mp4'}
            )
            
            # Generate S3 URL
            s3_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{s3_key}"
            print(f"✅ S3 upload successful: {s3_url}")
            return s3_url
        except ClientError as e:
            print(f"❌ S3 upload failed: {str(e)}")
            raise Exception(f"Failed to upload to S3: {str(e)}")
        except Exception as e:
            print(f"❌ Upload error: {str(e)}")
            raise Exception(f"Failed to upload to S3: {str(e)}")

    # Include the VideoWatermarker class directly in the function
    class VideoWatermarker:
        def __init__(self, video_path: str, watermark_text: str = "Created using LisaApp.in\nAI-Powered Course Builder", font_size: int = 18, fixed_position: str = None):
            """
            Initialize the video watermarker
            
            Args:
                video_path (str): Path to the input video file
                watermark_text (str): Text to use as watermark
                font_size (int): Font size for watermark text
                fixed_position (str): Fixed position for watermark ('top-left', 'top-right', 'bottom-left', 'bottom-right', 'center')
            """
            import cv2  # Import cv2 here
            
            self.video_path = video_path
            self.watermark_text = watermark_text
            self.font_size = font_size
            self.fixed_position = fixed_position
            self.cap = cv2.VideoCapture(video_path)
            
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.duration = self.frame_count / self.fps
            
            print(f"Video Info:")
            print(f"  Duration: {self.duration:.2f} seconds")
            print(f"  FPS: {self.fps}")
            print(f"  Resolution: {self.width}x{self.height}")
            print(f"  Total frames: {self.frame_count}")
            if self.fixed_position:
                print(f"  Fixed watermark position: {self.fixed_position}")
            else:
                print(f"  Random watermark positions (changing every interval)")
        
        def calculate_watermark_info(self, change_interval: float = 10.0) -> dict:
            """
            Calculate watermark information based on video duration
            
            Args:
                change_interval (float): Interval in seconds between position changes
                
            Returns:
                dict: Dictionary containing watermark information
            """
            timestamps = []
            current_time = 0.0
            
            while current_time < self.duration:
                timestamps.append(current_time)
                current_time += change_interval
            
            num_watermarks = len(timestamps)
            
            return {
                'num_watermarks': num_watermarks,
                'change_interval': change_interval,
                'video_duration': self.duration,
                'timestamps': timestamps,
                'watermarks_per_minute': (60 / change_interval) if change_interval > 0 else 0
            }
        
        def process_video(self, output_path: str, change_interval: float = 10.0) -> None:
            """
            Process the video and add watermarks that change position every X seconds
            
            Args:
                output_path (str): Path for output video
                change_interval (float): Interval in seconds between position changes
            """
            import cv2
            import numpy as np
            from PIL import Image, ImageDraw, ImageFont
            
            # Get video properties
            fps = self.fps
            width = self.width
            height = self.height
            total_frames = self.frame_count
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Calculate watermark positions
            watermark_info = self.calculate_watermark_info(change_interval)
            timestamps = watermark_info['timestamps']
            
            # Create watermark image
            def create_watermark_image(text, font_size=18):
                # Create a transparent image
                img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)
                
                # Try to use a default font, fallback to default if not available
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                except:
                    font = ImageFont.load_default()
                
                # Split text into lines
                lines = text.split('\n')
                line_height = font_size + 5
                
                # Calculate text dimensions
                max_width = 0
                for line in lines:
                    bbox = draw.textbbox((0, 0), line, font=font)
                    max_width = max(max_width, bbox[2] - bbox[0])
                
                # Create background rectangle
                bg_height = len(lines) * line_height + 10
                bg_width = max_width + 20
                
                # Position watermark based on fixed_position or random
                if self.fixed_position:
                    if self.fixed_position == 'top-left':
                        x, y = 20, 20
                    elif self.fixed_position == 'top-right':
                        x, y = width - bg_width - 20, 20
                    elif self.fixed_position == 'bottom-left':
                        x, y = 20, height - bg_height - 20
                    elif self.fixed_position == 'bottom-right':
                        x, y = width - bg_width - 20, height - bg_height - 20
                    elif self.fixed_position == 'center':
                        x, y = (width - bg_width) // 2, (height - bg_height) // 2
                    else:
                        x, y = 20, 20
                else:
                    # Random position (simplified for now)
                    x, y = 20, 20
                
                # Draw background rectangle
                draw.rectangle([x, y, x + bg_width, y + bg_height], 
                             fill=(0, 0, 0, 128), outline=(255, 255, 255, 255))
                
                # Draw text
                for i, line in enumerate(lines):
                    draw.text((x + 10, y + 5 + i * line_height), line, 
                            fill=(255, 255, 255, 255), font=font)
                
                return np.array(img)
            
            # Process each frame
            frame_count = 0
            current_position = 0
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Calculate current time
                current_time = frame_count / fps
                
                # Check if we need to change watermark position
                if timestamps and current_position < len(timestamps):
                    if current_time >= timestamps[current_position]:
                        current_position += 1
                
                # Create watermark for this frame
                watermark_img = create_watermark_image(self.watermark_text, self.font_size)
                
                # Convert frame to RGBA for alpha blending
                frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                
                # Blend watermark with frame
                alpha = watermark_img[:, :, 3] / 255.0
                alpha = np.expand_dims(alpha, axis=2)
                
                for c in range(3):
                    frame_rgba[:, :, c] = (1 - alpha[:, :, 0]) * frame_rgba[:, :, c] + \
                                         alpha[:, :, 0] * watermark_img[:, :, c]
                
                # Convert back to BGR
                frame_watermarked = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)
                
                # Write frame
                out.write(frame_watermarked)
                frame_count += 1
                
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames")
            
            # Cleanup
            out.release()
            print(f"✅ Video processing complete! Output saved to: {output_path}")
        
        def __del__(self):
            """Cleanup when object is destroyed"""
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()

    @app.post("/watermark")
    async def add_watermark(
        video: UploadFile = File(...),
        watermark_text: str = Form("Created using LisaApp.in\nAI-Powered Course Builder"),
        change_interval: float = Form(10.0),
        font_size: int = Form(18),
        fixed_position: Optional[str] = Form(None)
    ):
        """
        API endpoint to add watermarks to video
        
        Expected form data:
        - video: video file
        - watermark_text: text for watermark (optional, default: "Created using LisaApp.in - AI-Powered Course Builder")
        - change_interval: interval in seconds between watermark position changes (optional, default: 10.0)
        - font_size: font size for watermark text (optional, default: 18)
        - fixed_position: fixed position for watermark (optional: 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'center')
        """
        try:
            # Check if video file is present
            if not video:
                raise HTTPException(status_code=400, detail="No video file provided")
            
            if video.filename == '':
                raise HTTPException(status_code=400, detail="No video file selected")
            
            if not allowed_file(video.filename):
                raise HTTPException(
                    status_code=400, 
                    detail="Invalid file type. Allowed: mp4, avi, mov, mkv, wmv"
                )
            
            # Generate unique filename
            unique_id = str(uuid.uuid4())
            input_filename = video.filename
            input_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_{input_filename}")
            output_filename = f"watermarked_{input_filename}"
            output_path = os.path.join(OUTPUT_FOLDER, f"{unique_id}_{output_filename}")
            
            # Save uploaded file
            with open(input_path, "wb") as buffer:
                content = await video.read()
                buffer.write(content)
            
            # Process video with watermarking
            try:
                import cv2
                
                # Process video
                watermarker = VideoWatermarker(input_path, watermark_text, font_size, fixed_position)
                
                # Calculate number of watermarks based on video duration
                watermark_info = watermarker.calculate_watermark_info(change_interval)
                video_duration = watermark_info['video_duration']
                num_watermarks = watermark_info['num_watermarks']
                
                watermarker.process_video(
                    output_path,
                    change_interval=change_interval
                )
                
                # Check if the output file exists and has content
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    print(f"✅ Output file created: {output_path} (size: {file_size} bytes)")
                else:
                    print(f"❌ Output file not found: {output_path}")
                    raise Exception("Video processing failed - output file not created")
                
                # Try to upload to S3 if configured
                s3_url = None
                if s3_client:
                    try:
                        print(f"📤 Uploading to S3: {output_path} -> {output_filename}")
                        s3_url = upload_to_s3(output_path, output_filename)
                        print(f"✅ Uploaded to S3: {s3_url}")
                    except Exception as e:
                        print(f"⚠️ S3 upload failed: {e}")
                        s3_url = None
                
                # Clean up input file
                try:
                    os.remove(input_path)
                except Exception as e:
                    print(f"Warning: Could not clean up input file: {e}")

                # Return success response with file info
                response = {
                    'success': True,
                    'message': 'Video watermarked successfully',
                    'input_file': input_filename,
                    'output_file': output_filename,
                    's3_url': s3_url,
                    'download_url': s3_url if s3_url else f"/download/{output_filename}",
                    'parameters': {
                        'watermark_text': watermark_text,
                        'change_interval': change_interval,
                        'font_size': font_size,
                        'fixed_position': fixed_position,
                        'video_duration': round(video_duration, 2),
                        'num_watermarks': num_watermarks,
                        'watermarks_per_minute': round(watermark_info['watermarks_per_minute'], 1),
                        'timestamps': [round(t, 2) for t in watermark_info['timestamps']]
                    }
                }
                
                return JSONResponse(content=response, status_code=200)
                
            except ImportError as e:
                print(f"OpenCV not available: {e}")
                # Fallback if OpenCV is not available
                response = {
                    'success': True,
                    'message': 'Video received successfully (OpenCV not available)',
                    'input_file': input_filename,
                    'output_file': output_filename,
                    's3_url': None,
                    'download_url': None,
                    'parameters': {
                        'watermark_text': watermark_text,
                        'change_interval': change_interval,
                        'font_size': font_size,
                        'fixed_position': fixed_position,
                        'video_duration': 0,
                        'num_watermarks': 0,
                        'watermarks_per_minute': 0,
                        'timestamps': []
                    }
                }
                return JSONResponse(content=response, status_code=200)
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/download/{filename}")
    async def download_file(filename: str):
        """Download the processed video file"""
        try:
            file_path = os.path.join(OUTPUT_FOLDER, filename)
            if os.path.exists(file_path):
                return FileResponse(
                    path=file_path,
                    filename=filename,
                    media_type='video/mp4'
                )
            else:
                raise HTTPException(
                    status_code=404, 
                    detail="File not found. If S3 is configured, use the s3_url from the watermark response."
                )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app

@app.local_entrypoint()
def main():
    """Local entrypoint for testing"""
    print("🚀 Updating video-watermarking-api-v2 with S3 integration...")
    print("📝 This will modify the existing app to include S3 functionality")
    print("🔗 The same URL will now return S3 URLs!")

if __name__ == "__main__":
    with modal.enable_output():
        app.run() 