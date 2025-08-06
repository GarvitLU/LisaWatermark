import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import os
from typing import List, Tuple
import argparse

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
    
    def create_watermark_image(self, text: str, font_size: int = 18, color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """
        Create a watermark image with the given text (supports multi-line)
        
        Args:
            text (str): Text to render (can be multi-line with \n)
            font_size (int): Font size
            color (tuple): RGB color tuple
            
        Returns:
            np.ndarray: Watermark image as numpy array
        """
        print(f"DEBUG: Creating watermark with font_size={font_size}")
        
        # Split text into lines
        lines = text.split('\n')
        
        # Try to use a default font, fallback to default if not available
        font = None
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
            print(f"DEBUG: Loaded arial.ttf with size {font_size}")
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
                print(f"DEBUG: Loaded Arial.ttf with size {font_size}")
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                    print(f"DEBUG: Loaded DejaVuSans.ttf with size {font_size}")
                except:
                    font = ImageFont.load_default()
                    print(f"DEBUG: Using default font with size {font_size}")
        
        # Calculate text dimensions
        max_width = 0
        total_height = 0
        
        for line in lines:
            bbox = ImageDraw.Draw(Image.new('RGBA', (1, 1))).textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
            line_height = bbox[3] - bbox[1]
            max_width = max(max_width, line_width)
            total_height += line_height
            print(f"DEBUG: Line '{line}' - width={line_width}, height={line_height}")
        
        print(f"DEBUG: Total dimensions - width={max_width}, height={total_height}")
        
        # Add proportional padding based on font size
        padding = max(6, font_size // 8)  # Proportional padding: minimum 6px, or 1/8 of font size
        bottom_padding = padding + 4  # Extra padding at the bottom
        img_width = max_width + padding * 2
        img_height = total_height + padding + bottom_padding
        
        print(f"DEBUG: Image dimensions - width={img_width}, height={img_height}")
        
        # Create image
        text_img = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_img)
        
        # Add semi-transparent background
        text_draw.rectangle([0, 0, img_width, img_height], fill=(0, 0, 0, 128))
        
        # Draw text
        y_offset = padding
        for line in lines:
            text_draw.text((padding, y_offset), line, font=font, fill=color)
            bbox = text_draw.textbbox((0, 0), line, font=font)
            y_offset += bbox[3] - bbox[1]
        
        # Convert to numpy array
        return np.array(text_img)
    
    def get_random_position(self, watermark_width: int, watermark_height: int, exclude_position: Tuple[int, int] = None) -> Tuple[int, int]:
        """
        Get a random position for the watermark, avoiding only the previous position to prevent consecutive repeats
        
        Args:
            watermark_width (int): Width of watermark
            watermark_height (int): Height of watermark
            exclude_position (tuple): Previous position to avoid (prevents consecutive repeats)
            
        Returns:
            Tuple[int, int]: (x, y) position
        """
        # Define possible positions (corners and center)
        positions = [
            (10, 10),  # Top left
            (self.width - watermark_width - 10, 10),  # Top right
            (10, self.height - watermark_height - 10),  # Bottom left
            (self.width - watermark_width - 10, self.height - watermark_height - 10),  # Bottom right
            ((self.width - watermark_width) // 2, (self.height - watermark_height) // 2),  # Center
        ]
        
        # If we need to exclude the previous position, remove it from the list
        if exclude_position and exclude_position in positions:
            available_positions = [pos for pos in positions if pos != exclude_position]
        else:
            available_positions = positions
        
        return random.choice(available_positions)
    
    def get_fixed_position(self, watermark_width: int, watermark_height: int) -> Tuple[int, int]:
        """
        Get a fixed position for the watermark based on the specified position
        
        Args:
            watermark_width (int): Width of watermark
            watermark_height (int): Height of watermark
            
        Returns:
            Tuple[int, int]: (x, y) position
        """
        if not self.fixed_position:
            return self.get_random_position(watermark_width, watermark_height)
        
        # Define fixed positions
        positions = {
            'top-left': (10, 10),
            'top-right': (self.width - watermark_width - 10, 10),
            'bottom-left': (10, self.height - watermark_height - 10),
            'bottom-right': (self.width - watermark_width - 10, self.height - watermark_height - 10),
            'center': ((self.width - watermark_width) // 2, (self.height - watermark_height) // 2)
        }
        
        if self.fixed_position.lower() not in positions:
            print(f"Warning: Invalid position '{self.fixed_position}'. Using 'bottom-right' instead.")
            return positions['bottom-right']
        
        return positions[self.fixed_position.lower()]
    
    def get_position_change_timestamps(self, change_interval: float = 10.0) -> List[float]:
        """
        Generate timestamps for watermark position changes every X seconds
        
        Args:
            change_interval (float): Interval in seconds between position changes
            
        Returns:
            List[float]: List of timestamps for position changes
        """
        timestamps = []
        current_time = 0.0
        
        while current_time < self.duration:
            timestamps.append(current_time)
            current_time += change_interval
        
        return timestamps
    
    def calculate_watermark_info(self, change_interval: float = 10.0) -> dict:
        """
        Calculate watermark information based on video duration
        
        Args:
            change_interval (float): Interval in seconds between position changes
            
        Returns:
            dict: Dictionary containing watermark information
        """
        timestamps = self.get_position_change_timestamps(change_interval)
        num_watermarks = len(timestamps)
        
        return {
            'num_watermarks': num_watermarks,
            'change_interval': change_interval,
            'video_duration': self.duration,
            'timestamps': timestamps,
            'watermarks_per_minute': (60 / change_interval) if change_interval > 0 else 0
        }
    
    def add_watermark_to_frame(self, frame: np.ndarray, watermark_img: np.ndarray, 
                              position: Tuple[int, int]) -> np.ndarray:
        """
        Add watermark to a single frame
        
        Args:
            frame (np.ndarray): Input frame
            watermark_img (np.ndarray): Watermark image
            position (tuple): (x, y) position
            
        Returns:
            np.ndarray: Frame with watermark
        """
        x, y = position
        h, w = watermark_img.shape[:2]
        
        # Ensure position is within frame bounds
        if x + w > self.width:
            x = self.width - w
        if y + h > self.height:
            y = self.height - h
        
        # Convert frame to RGBA for alpha blending
        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        
        # Create a region of interest
        roi = frame_rgba[y:y+h, x:x+w]
        
        # Blend watermark with frame
        alpha = watermark_img[:, :, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=2)
        
        # Blend RGB channels
        for c in range(3):
            roi[:, :, c] = (1 - alpha[:, :, 0]) * roi[:, :, c] + alpha[:, :, 0] * watermark_img[:, :, c]
        
        # Convert back to BGR
        return cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)
    
    def process_video(self, output_path: str, change_interval: float = 10.0) -> None:
        """
        Process the video and add watermarks that change position every X seconds
        
        Args:
            output_path (str): Path for output video
            change_interval (float): Interval in seconds between position changes
        """
        # Get position change timestamps and watermark info
        watermark_info = self.calculate_watermark_info(change_interval)
        timestamps = watermark_info['timestamps']
        
        print(f"Watermark Configuration:")
        print(f"  Video duration: {watermark_info['video_duration']:.2f} seconds")
        if self.fixed_position:
            print(f"  Fixed position: {self.fixed_position}")
        else:
            print(f"  Change interval: {watermark_info['change_interval']} seconds")
            print(f"  Number of watermarks: {watermark_info['num_watermarks']}")
            print(f"  Watermarks per minute: {watermark_info['watermarks_per_minute']:.1f}")
            print(f"  Position change timestamps: {[f'{t:.2f}s' for t in timestamps]}")
        
        # Create watermark image
        watermark_img = self.create_watermark_image(self.watermark_text, self.font_size)
        watermark_height, watermark_width = watermark_img.shape[:2]
        
        # Try different codecs in order of preference
        codecs_to_try = [
            ('mp4v', 'mp4v'),  # MP4V codec
            ('avc1', 'avc1'),  # AVC1 codec
            ('XVID', 'XVID'),  # XVID codec
            ('MJPG', 'MJPG'),  # Motion JPEG
        ]
        
        temp_output = output_path.replace('.mp4', '_temp.mp4')
        out = None
        
        # Try each codec until one works
        for codec_name, fourcc_code in codecs_to_try:
            try:
                print(f"Trying codec: {codec_name}")
                fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
                out = cv2.VideoWriter(temp_output, fourcc, self.fps, (self.width, self.height))
                
                if out.isOpened():
                    print(f"✅ Successfully opened video writer with codec: {codec_name}")
                    break
                else:
                    print(f"❌ Failed to open video writer with codec: {codec_name}")
                    if out:
                        out.release()
            except Exception as e:
                print(f"❌ Error with codec {codec_name}: {e}")
                if out:
                    out.release()
                continue
        
        if not out or not out.isOpened():
            raise ValueError(f"Could not create output video with any codec. Tried: {[c[0] for c in codecs_to_try]}")
        
        # Process each frame
        current_frame = 0
        
        if self.fixed_position:
            # Use fixed position throughout the video
            current_position = self.get_fixed_position(watermark_width, watermark_height)
            print(f"Using fixed position: {self.fixed_position} at {current_position}")
        else:
            # Use random positions that change over time
            current_position = self.get_random_position(watermark_width, watermark_height)
            previous_position = None
            current_timestamp_index = 0
        
        print("Processing video...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            current_time = current_frame / self.fps
            
            # Check if we need to change watermark position (only for random mode)
            if not self.fixed_position and (current_timestamp_index < len(timestamps) and 
                current_time >= timestamps[current_timestamp_index]):
                
                previous_position = current_position
                current_position = self.get_random_position(watermark_width, watermark_height, previous_position)
                print(f"Changing watermark position at {current_time:.2f}s from {previous_position} to {current_position}")
                current_timestamp_index += 1
            
            # Add watermark to frame
            frame = self.add_watermark_to_frame(frame, watermark_img, current_position)
            
            out.write(frame)
            current_frame += 1
            
            # Progress indicator
            if current_frame % (self.fps * 5) == 0:  # Every 5 seconds
                progress = (current_frame / self.frame_count) * 100
                print(f"Progress: {progress:.1f}%")
        
        # Cleanup video writer
        self.cap.release()
        out.release()
        
        # Merge with original audio using ffmpeg
        print("Merging with original audio...")
        import subprocess
        
        try:
            # Use ffmpeg to merge video with original audio
            cmd = [
                'ffmpeg', '-y',  # Overwrite output file
                '-i', temp_output,  # Input video (without audio)
                '-i', self.video_path,  # Input original video (with audio)
                '-c:v', 'libx264',  # Use H.264 codec for video
                '-c:a', 'aac',  # Use AAC codec for audio
                '-preset', 'fast',  # Faster encoding
                '-crf', '23',  # Good quality
                '-map', '0:v:0',  # Use video from first input
                '-map', '1:a:0',  # Use audio from second input
                output_path  # Output file
            ]
            
            print(f"Running FFmpeg command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Remove temporary file
                import os
                os.remove(temp_output)
                print(f"✅ Video processing complete! Output saved to: {output_path}")
                print(f"✅ Audio merged successfully")
            else:
                print(f"❌ Warning: Could not merge audio. Using video without audio.")
                print(f"FFmpeg error: {result.stderr}")
                print(f"FFmpeg stdout: {result.stdout}")
                # If ffmpeg fails, just rename the temp file
                import os
                os.rename(temp_output, output_path)
                print(f"⚠️ Video processing complete! Output saved to: {output_path} (no audio)")
                
        except FileNotFoundError:
            print("❌ Warning: FFmpeg not found. Using video without audio.")
            import os
            os.rename(temp_output, output_path)
            print(f"⚠️ Video processing complete! Output saved to: {output_path} (no audio)")
        except Exception as e:
            print(f"❌ Warning: Error merging audio: {e}")
            import os
            os.rename(temp_output, output_path)
            print(f"⚠️ Video processing complete! Output saved to: {output_path} (no audio)")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

def main():
    parser = argparse.ArgumentParser(description='Add watermarks to video that change position every X seconds')
    parser.add_argument('input_video', help='Path to input video file')
    parser.add_argument('output_video', help='Path to output video file')
    parser.add_argument('--watermark-text', default='Created using LisaApp.in\nAI-Powered Course Builder', help='Watermark text')
    parser.add_argument('--change-interval', type=float, default=10.0, help='Interval in seconds between position changes')
    parser.add_argument('--font-size', type=int, default=18, help='Font size for watermark text')
    parser.add_argument('--fixed-position', choices=['top-left', 'top-right', 'bottom-left', 'bottom-right', 'center'], 
                       help='Fixed watermark position (if specified, position will not change)')
    
    args = parser.parse_args()
    
    try:
        # Create watermarker
        watermarker = VideoWatermarker(args.input_video, args.watermark_text, args.font_size, args.fixed_position)
        
        # Show watermark information
        watermark_info = watermarker.calculate_watermark_info(args.change_interval)
        print(f"\nWatermark Summary:")
        print(f"  Video duration: {watermark_info['video_duration']:.2f} seconds")
        if args.fixed_position:
            print(f"  Fixed position: {args.fixed_position}")
        else:
            print(f"  Change interval: {watermark_info['change_interval']} seconds")
            print(f"  Number of watermarks: {watermark_info['num_watermarks']}")
            print(f"  Watermarks per minute: {watermark_info['watermarks_per_minute']:.1f}")
        
        # Process video
        watermarker.process_video(
            args.output_video,
            change_interval=args.change_interval
        )
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 