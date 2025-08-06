# Modal Deployment Guide for Video Watermarking API

This guide will help you deploy the FastAPI video watermarking service to Modal and get a public URL.

## Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **AWS S3 Bucket**: Create an S3 bucket for storing watermarked videos
3. **AWS Credentials**: Get your AWS access key and secret key

## Step 1: Install Modal CLI

```bash
pip install modal
```

## Step 2: Authenticate with Modal

```bash
modal token new
```

Follow the instructions to authenticate with your Modal account.

## Step 3: Set up AWS Credentials in Modal

1. Go to the [Modal Dashboard](https://modal.com/secrets)
2. Click "Create Secret"
3. Name it `aws-credentials`
4. Add the following environment variables:
   ```
   AWS_S3_BUCKET_NAME=your-s3-bucket-name
   AWS_REGION=us-east-1
   AWS_ACCESS_KEY_ID=your-aws-access-key
   AWS_SECRET_ACCESS_KEY=your-aws-secret-key
   ```

## Step 4: Deploy to Modal

Run the deployment script:

```bash
python modal_deploy.py
```

This will:
- Build the Docker image with all dependencies
- Deploy the FastAPI application to Modal
- Provide you with a public URL

## Step 5: Get Your Public URL

After deployment, Modal will provide you with a public URL like:
```
https://your-app-name--your-username.modal.run
```

## Step 6: Test Your API

### Using Postman

1. **Create a new POST request**
2. **Set URL**: `https://your-app-name--your-username.modal.run/watermark`
3. **Set Body to form-data**:
   - `video` (File): Your video file
   - `watermark_text` (Text): "Created using LisaApp.in\nAI-Powered Course Builder"
   - `change_interval` (Text): "10.0"
   - `font_size` (Text): "18"
   - `fixed_position` (Text): "bottom-right" (optional)

### Using curl

```bash
curl -X POST "https://your-app-name--your-username.modal.run/watermark" \
  -H "Content-Type: multipart/form-data" \
  -F "video=@your_video.mp4" \
  -F "watermark_text=Created using LisaApp.in\nAI-Powered Course Builder" \
  -F "change_interval=10.0" \
  -F "font_size=18" \
  -F "fixed_position=bottom-right"
```

## Expected Response

```json
{
  "success": true,
  "message": "Video watermarked successfully and uploaded to S3",
  "input_file": "your_video.mp4",
  "output_file": "watermarked_your_video.mp4",
  "s3_url": "https://your-bucket.s3.us-east-1.amazonaws.com/watermarked-videos/watermarked_your_video.mp4",
  "download_url": "https://your-bucket.s3.us-east-1.amazonaws.com/watermarked-videos/watermarked_your_video.mp4",
  "parameters": {
    "watermark_text": "Created using LisaApp.in\nAI-Powered Course Builder",
    "change_interval": 10.0,
    "font_size": 18,
    "fixed_position": "bottom-right",
    "video_duration": 120.5,
    "num_watermarks": 13,
    "watermarks_per_minute": 6.0,
    "timestamps": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0]
  }
}
```

## API Documentation

Once deployed, you can access the interactive API documentation at:
- **Swagger UI**: `https://your-app-name--your-username.modal.run/docs`
- **ReDoc**: `https://your-app-name--your-username.modal.run/redoc`

## Troubleshooting

### Common Issues

1. **S3 Upload Fails**: Check your AWS credentials and bucket permissions
2. **Video Processing Fails**: Ensure the video format is supported (MP4, AVI, MOV, MKV, WMV)
3. **Timeout Errors**: Large videos may take longer to process. The timeout is set to 5 minutes.

### Checking Logs

You can view logs in the Modal dashboard:
1. Go to [Modal Dashboard](https://modal.com/apps)
2. Click on your app
3. View the logs for debugging

### Redeploying

To update your deployment:

```bash
python modal_deploy.py
```

Modal will automatically update the deployment with your changes.

## Cost Optimization

- The service uses 2GB RAM and 1 CPU
- Timeout is set to 5 minutes
- Consider adjusting these based on your needs
- Modal charges based on actual usage time

## Security Notes

- The API accepts files from any origin (CORS is enabled for all origins)
- Consider adding authentication if needed
- S3 bucket should have appropriate permissions
- Consider using presigned URLs for S3 uploads in production 