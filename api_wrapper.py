import os
import tempfile
import json
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from video_watermark import VideoWatermarker
import uuid
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

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

@app.route('/watermark', methods=['POST'])
def add_watermark():
    """
    API endpoint to add watermarks to video
    
    Expected form data:
    - video: video file
    - watermark_text: text for watermark (optional, default: "Created using LisaApp.in - AI-Powered Course Builder")
    - change_interval: interval in seconds between watermark position changes (optional, default: 10.0)
    - font_size: font size for watermark text (optional, default: 18)
    
    The number of watermarks is automatically calculated based on video duration:
    - One watermark every 10 seconds (or specified change_interval)
    - Minimum of 1 watermark for any video
    
    Position Selection:
    - 5 possible positions: corners and center
    - No consecutive repeats (same position never appears twice in a row)
    - Random selection ensures varied distribution
    """
    try:
        # Check if video file is present
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        
        if video_file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400
        
        if not allowed_file(video_file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv, wmv'}), 400
        
        # Get parameters
        watermark_text = request.form.get('watermark_text', 'Created using LisaApp.in\nAI-Powered Course Builder')
        change_interval = float(request.form.get('change_interval', 10.0))
        font_size = int(request.form.get('font_size', 18))
        
        # Calculate number of watermarks based on video duration
        # We'll determine this after loading the video
        
        # Generate unique filename
        unique_id = str(uuid.uuid4())
        input_filename = secure_filename(video_file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_{input_filename}")
        output_filename = f"watermarked_{input_filename}"
        output_path = os.path.join(OUTPUT_FOLDER, f"{unique_id}_{output_filename}")
        
        # Save uploaded file
        video_file.save(input_path)
        
        # Process video
        watermarker = VideoWatermarker(input_path, watermark_text, font_size)
        
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
                'video_duration': round(video_duration, 2),
                'num_watermarks': num_watermarks,
                'watermarks_per_minute': round(watermark_info['watermarks_per_minute'], 1),
                'timestamps': [round(t, 2) for t in watermark_info['timestamps']]
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download the processed video file (fallback for when S3 is not configured)"""
    try:
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found. If S3 is configured, use the s3_url from the watermark response.'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    s3_status = 'configured' if s3_client else 'not_configured'
    return jsonify({
        'status': 'healthy', 
        'message': 'Video watermarking API is running',
        's3_status': s3_status,
        's3_bucket': S3_BUCKET if S3_BUCKET else None
    }), 200

@app.route('/test-s3', methods=['GET'])
def test_s3():
    """Test S3 connectivity and bucket access"""
    if not s3_client:
        return jsonify({
            'error': 'S3 client not configured',
            'missing_vars': [var for var, val in {
                'AWS_S3_BUCKET_NAME': os.getenv('AWS_S3_BUCKET_NAME'),
                'AWS_REGION': os.getenv('AWS_REGION'),
                'AWS_ACCESS_KEY_ID': AWS_ACCESS_KEY_ID,
                'AWS_SECRET_ACCESS_KEY': AWS_SECRET_ACCESS_KEY
            }.items() if not val]
        }), 400
    
    try:
        # Test bucket access
        s3_client.head_bucket(Bucket=S3_BUCKET)
        return jsonify({
            'success': True,
            'message': 'S3 connection successful',
            'bucket': S3_BUCKET,
            'region': S3_REGION,
            'access_key_id': AWS_ACCESS_KEY_ID[:10] + '...' if AWS_ACCESS_KEY_ID else None
        }), 200
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        return jsonify({
            'error': f'S3 connection failed: {error_code}',
            'message': error_message,
            'bucket': S3_BUCKET,
            'region': S3_REGION
        }), 400
    except Exception as e:
        return jsonify({
            'error': f'S3 connection failed: {str(e)}',
            'bucket': S3_BUCKET,
            'region': S3_REGION
        }), 400

@app.route('/check-video/<filename>', methods=['GET'])
def check_video(filename):
    """Check video file properties using ffprobe"""
    try:
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        import subprocess
        
        # Use ffprobe to get video information
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', file_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            import json
            video_info = json.loads(result.stdout)
            
            # Extract useful information
            format_info = video_info.get('format', {})
            streams = video_info.get('streams', [])
            
            video_streams = [s for s in streams if s.get('codec_type') == 'video']
            audio_streams = [s for s in streams if s.get('codec_type') == 'audio']
            
            return jsonify({
                'success': True,
                'file_path': file_path,
                'file_size': format_info.get('size'),
                'duration': format_info.get('duration'),
                'video_streams': len(video_streams),
                'audio_streams': len(audio_streams),
                'video_codec': video_streams[0].get('codec_name') if video_streams else None,
                'audio_codec': audio_streams[0].get('codec_name') if audio_streams else None,
                'resolution': f"{video_streams[0].get('width')}x{video_streams[0].get('height')}" if video_streams else None,
                'fps': video_streams[0].get('r_frame_rate') if video_streams else None
            }), 200
        else:
            return jsonify({
                'error': 'Failed to analyze video',
                'ffprobe_error': result.stderr
            }), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 