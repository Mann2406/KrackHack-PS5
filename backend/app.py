import cv2
import numpy as np
import yt_dlp
from flask import Flask, render_template, Response, request
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO(r"C:\Users\Mahesh\OneDrive\Desktop\Real time Traffic Analysis\BDD100K_YOLOv8\yolov8n_train\weights\best.pt")

app = Flask(__name__)

def get_youtube_stream_url(youtube_url):
    """Extract the direct YouTube video URL using yt-dlp."""
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'quiet': True
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=False)
            return info_dict.get('url')  # Return the direct video stream URL
    except Exception as e:
        print(f"Error extracting YouTube URL: {e}")
        return None

def generate_frames(youtube_url):
    """Streams and processes YouTube video using OpenCV instead of FFmpeg."""
    stream_url = get_youtube_stream_url(youtube_url)
    if not stream_url:
        print("Failed to retrieve video stream.")
        return

    cap = cv2.VideoCapture(stream_url)  # Open the YouTube stream with OpenCV
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break  # Stop if no frames are received

        # Run YOLOv8 object detection
        results = model.predict(frame, conf=0.25)

        for result in results:
            frame = result.plot()  # Draw bounding boxes and labels

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()  # Release video capture when done

@app.route('/')
def index():
    """Render the frontend UI."""
    return render_template('index.html')

@app.route('/youtube_feed')
def youtube_feed():
    youtube_url = request.args.get('youtube_url')
    if not youtube_url:
        return "YouTube URL not provided", 400
    return Response(generate_frames(youtube_url), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
