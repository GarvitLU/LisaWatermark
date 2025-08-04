import os
import tempfile
import json
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from video_watermark import VideoWatermarker
import uuid

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
        
        # Return success response with file info
        response = {
            'success': True,
            'message': 'Video watermarked successfully',
            'input_file': input_filename,
            'output_file': output_filename,
            'download_url': f'/download/{unique_id}_{output_filename}',
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
    """Download the processed video file"""
    try:
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Video watermarking API is running'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 