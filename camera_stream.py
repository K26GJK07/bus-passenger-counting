#!/usr/bin/env python3
"""
カメラ映像をブラウザでリアルタイム確認するスクリプト
Windows ブラウザで http://[RPiのIP]:5000 にアクセスして確認
"""
from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import time

app = Flask(__name__)
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    controls={"FrameDurationLimits": (33333, 33333)}
)
picam2.configure(config)
picam2.start()
time.sleep(1)

def generate_frames():
    while True:
        frame = picam2.capture_array()
        # RGB → BGR に変換（OpenCV用）
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', bgr)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'
               + buffer.tobytes()
               + b'\r\n')

@app.route('/')
def index():
    return '''
    <html>
    <head><title>Camera Stream</title></head>
    <body style="background:#000; margin:0; display:flex; 
                 justify-content:center; align-items:center; height:100vh;">
        <img src="/video_feed" style="max-width:100%;">
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("ブラウザで以下にアクセスしてください:")
    import socket
    ip = socket.gethostbyname(socket.gethostname())
    print(f"  http://{ip}:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)