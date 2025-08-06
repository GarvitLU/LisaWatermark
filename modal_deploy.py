#!/usr/bin/env python3
"""
Modal deployment script for the FastAPI video watermarking service
"""

import modal

# Define the image with all dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi",
    "uvicorn[standard]",
    "python-multipart",
    "opencv-python",
    "Pillow",
    "numpy",
    "boto3",
    "python-dotenv"
)

# Create the app
app = modal.App("video-watermarking-api", image=image)

@app.function(
    secrets=[
        modal.Secret.from_name("aws-credentials"),  # Create this in Modal dashboard
    ],
    timeout=300,  # 5 minutes timeout
    memory=2048,  # 2GB RAM
    cpu=1.0,      # 1 CPU
)
@modal.asgi_app()
def fastapi_app():
    """Deploy the FastAPI application with only the watermark endpoint"""
    import os
    import tempfile
    import json
    from fastapi import FastAPI, File, UploadFile, Form, HTTPException
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from typing import Optional
    from video_watermark import VideoWatermarker
    import uuid
    import boto3
    from botocore.exceptions import ClientError
    from dotenv import load_dotenv

    # Load environment variables from .env file
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

    # S3 Configuration
    S3_BUCKET = os.getenv('AWS_S3_BUCKET_NAME') or os.getenv('S3_BUCKET')
    S3_REGION = os.getenv('AWS_REGION') or os.getenv('S3_REGION', 'us-east-1')
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

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
            missing_vars.append("AWS_S3_BUCKET_NAME or S3_BUCKET")
        if not AWS_ACCESS_KEY_ID:
            missing_vars.append("AWS_ACCESS_KEY_ID")
        if not AWS_SECRET_ACCESS_KEY:
            missing_vars.append("AWS_SECRET_ACCESS_KEY")
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
            
            # Upload to S3
            print(f"📤 Uploading to S3: {output_path} -> {output_filename}")
            s3_url = upload_to_s3(output_path, output_filename)
            print(f"✅ Uploaded to S3: {s3_url}")
            
            # Clean up local files
            try:
                os.remove(input_path)
                os.remove(output_path)
            except Exception as e:
                print(f"Warning: Could not clean up local files: {e}")

            # Return success response with file info
            response = {
                'success': True,
                'message': 'Video watermarked successfully and uploaded to S3',
                'input_file': input_filename,
                'output_file': output_filename,
                's3_url': s3_url,
                'download_url': s3_url,  # Keep for backward compatibility
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
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app

@app.local_entrypoint()
def main():
    """Local entrypoint for testing"""
    print("Video Watermarking API deployed to Modal!")
    print("Access the API at the provided URL")
    print("API Documentation will be available at /docs")

if __name__ == "__main__":
    app.run() 