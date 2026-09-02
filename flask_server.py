#!/usr/bin/env python3
"""
flask_server.py
バス乗降客カウンターのWeb配信専用プロセス

bus_shared.py の共有メモリ・統計ファイルから
データを読み込んでブラウザに配信する。

起動方法:
    検出プロセス（nbotcount.py または imxbotcount.py）を起動した後、
    別のターミナルで以下を実行:
        python3 flask_server.py

ブラウザアクセス:
    http://[Raspberry PiのIP]:8080/
"""

import sys
import time
import socket
import cv2
import threading
from flask import Flask, Response, jsonify
from bus_shared import FrameBuffer, StatsFile

# ============================================================
# 設定
# ============================================================
FLASK_PORT       = 8080
MJPEG_QUALITY    = 75
MJPEG_INTERVAL   = 0.033   # 約30FPS

# ============================================================
# 共有メモリへの接続
# ============================================================
print("=" * 55)
print("Bus Passenger Counter - Flask Web Server")
print("=" * 55)
print("検出プロセスの共有メモリに接続中...")

try:
    frame_buffer = FrameBuffer.attach()
    print("✓ 共有メモリに接続成功")
except FileNotFoundError:
    print("✖ 共有メモリが見つかりません")
    print("  先に検出プロセス（nbotcount.py または imxbotcount.py）を起動してください")
    sys.exit(1)

app = Flask(__name__)

# ============================================================
# Flask ルート
# ============================================================
@app.route('/')
def index():
    return '''
    <!doctype html>
    <html>
    <head>
        <title>乗降客カウンター</title>
        <meta charset="UTF-8">
        <style>
            body {
                margin:0; padding:20px;
                background:linear-gradient(135deg,#1e3c72 0%,#2a5298 100%);
                color:white; font-family:'Segoe UI',sans-serif;
            }
            .container { max-width:1200px; margin:0 auto; text-align:center; }
            h1 { font-size:2.2em; margin-bottom:10px; }
            .subtitle { font-size:1.1em; color:#a8d8ff; margin-bottom:20px; }
            .video {
                max-width:100%; height:auto;
                border:3px solid #4a90e2; border-radius:12px;
                box-shadow:0 8px 16px rgba(0,0,0,0.3);
            }
            .stats {
                margin-top:20px;
                background:rgba(255,255,255,0.1); padding:20px;
                border-radius:12px;
            }
            .stats-grid {
                display:grid;
                grid-template-columns:repeat(auto-fit,minmax(130px,1fr));
                gap:12px; margin-top:12px;
            }
            .stat-item {
                background:rgba(255,255,255,0.15); padding:15px;
                border-radius:8px;
            }
            .stat-value { font-size:2em; font-weight:bold; color:#4affff; }
            .stat-label { font-size:0.9em; color:#c0e0ff; margin-top:4px; }
            .system-badge {
                display:inline-block; padding:4px 12px;
                background:rgba(74,255,255,0.2); border-radius:12px;
                font-size:0.9em; margin-top:8px;
            }
        </style>
        <script>
            function updateStats() {
                fetch('/stats').then(r=>r.json()).then(data=>{
                    document.getElementById('entry-value').textContent    = data.entry_count;
                    document.getElementById('exit-value').textContent     = data.exit_count;
                    document.getElementById('current-value').textContent  = data.current_count;
                    document.getElementById('photos-value').textContent   = data.photo_count;
                    document.getElementById('cam-fps-value').textContent  = data.cam_fps.toFixed(1);
                    document.getElementById('imx-fps-value').textContent  = data.imx_fps.toFixed(1);
                    document.getElementById('person-value').textContent   = data.person_count;
                    document.getElementById('system-type').textContent    = data.system_type;
                });
            }
            setInterval(updateStats, 1000);
            window.onload = updateStats;
        </script>
    </head>
    <body>
        <div class="container">
            <h1>🚌 乗降客カウンター</h1>
            <div class="subtitle">
                Edge AI Bus Passenger Counter
                <div class="system-badge">System: <span id="system-type">-</span></div>
            </div>
            <img src="/video_feed" class="video">
            <div class="stats">
                <div style="font-size:1.3em;margin-bottom:10px;">📊 統計情報</div>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value" id="entry-value">-</div>
                        <div class="stat-label">乗車</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="exit-value">-</div>
                        <div class="stat-label">降車</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="current-value">-</div>
                        <div class="stat-label">現在</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="photos-value">-</div>
                        <div class="stat-label">写真</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="person-value">-</div>
                        <div class="stat-label">検出中</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="cam-fps-value">-</div>
                        <div class="stat-label">CAM FPS</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="imx-fps-value">-</div>
                        <div class="stat-label">IMX FPS</div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    '''


@app.route('/stats')
def stats_route():
    return jsonify(StatsFile.read())


@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            try:
                frame = frame_buffer.read()
                if frame is not None:
                    ret, jpeg = cv2.imencode(
                        '.jpg', frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), MJPEG_QUALITY]
                    )
                    if ret:
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                               + jpeg.tobytes() + b'\r\n')
            except Exception as e:
                print(f"Frame read error: {e}")
            time.sleep(MJPEG_INTERVAL)

    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ============================================================
# メイン
# ============================================================
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


def main():
    ip = get_local_ip()
    print("=" * 55)
    print(f"✓ Flask Web Server 起動")
    print(f"✓ アクセスURL: http://{ip}:{FLASK_PORT}/")
    print("=" * 55)
    print("Ctrl+C で終了\n")

    try:
        app.run(host="0.0.0.0", port=FLASK_PORT, threaded=True, debug=False)
    except KeyboardInterrupt:
        print("\nFlask Web Server を終了します")
    finally:
        frame_buffer.close()


if __name__ == "__main__":
    main()
