#!/usr/bin/env python3
"""
Working Modal deployment using the original api_wrapper.py and video_watermark.py files
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
        "python-dotenv",
        "requests"
    )
    .add_local_file("api_wrapper.py", "/root/api_wrapper.py")
    .add_local_file("video_watermark.py", "/root/video_watermark.py")
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
    """Deploy the FastAPI application using the original working files"""
    import sys
    import os
    
    # Add the root directory to Python path
    sys.path.append("/root")
    
    # Import the original working API wrapper
    from api_wrapper import app
    
    return app

@app.local_entrypoint()
def main():
    """Local entrypoint for testing"""
    print("🚀 Deploying with original working files...")
    print("📝 Using api_wrapper.py and video_watermark.py exactly as they are")
    print("🔗 This should work perfectly with visible watermarks!")

if __name__ == "__main__":
    with modal.enable_output():
        app.run() 