import sys
import collections
import os
import cv2
import threading
import time
import numpy as np
import argparse
import datetime
import socket
from collections import deque
from picamera2 import Picamera2
from flask import Flask, Response

# ============================================================
# 1. Python 3.10+ & NumPy 互換性対策
# ============================================================
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int

if not hasattr(collections, 'Mapping'):
    import collections.abc
    collections.Mapping        = collections.abc.Mapping
    collections.MutableMapping = collections.abc.MutableMapping
    collections.Sequence       = collections.abc.Sequence

# ============================================================
# 2. BoT-SORT パス設定とインポート
# ============================================================
BOT_SORT_PATH = "/home/yaslab/bus-passenger-counting/BoT-SORT"
sys.path.insert(0, BOT_SORT_PATH)
sys.path.insert(0, "/home/yaslab/bus-passenger-counting")   # hailo_reid.py の場所
os.chdir(BOT_SORT_PATH)

from tracker.bot_sort import BoTSORT

# ============================================================
# 3. Hailo YOLO 推論ラッパー
# ============================================================
from hailo_platform import (
    HEF, VDevice, HailoStreamInterface,
    InferVStreams, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, FormatType,
)

class HailoYOLODetector:
    """yolov11s.hef を使った人物検出クラス"""

    def __init__(self, hef_path: str, vdevice: VDevice, conf_threshold: float = 0.5):
        self.conf_threshold = conf_threshold
        self.hef    = HEF(hef_path)
        self.target = vdevice   # 共有VDeviceを使用
        configure_params = ConfigureParams.create_from_hef(
            hef=self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_group        = self.target.configure(self.hef, configure_params)[0]
        self.network_group_params = self.network_group.create_params()
        self.input_params  = InputVStreamParams.make(
            self.network_group, quantized=False, format_type=FormatType.UINT8)
        self.output_params = OutputVStreamParams.make(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32)
        self.input_key = list(self.input_params.keys())[0]
        self._pipeline = None
        self._ctx      = None
        print(f"✓ HailoYOLO ロード完了: {hef_path}")

    def open(self):
        """推論パイプラインを開く（スケジューラ使用時はactivate不要）"""
        self._pipeline = InferVStreams(
            self.network_group, self.input_params, self.output_params)
        self._pipeline.__enter__()  

    def close(self):
        if self._pipeline: self._pipeline.__exit__(None, None, None)

    def predict(self, frame: np.ndarray):
        """
        BGR フレームを受け取り (boxes, scores) を返す
        boxes : ndarray (N, 4) xyxy
        scores: ndarray (N,)
        """
        orig_h, orig_w = frame.shape[:2]
        # BGR → RGB に変換してリサイズ
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (640, 640)).astype(np.uint8)
        raw = self._pipeline.infer(
            {self.input_key: np.expand_dims(resized, axis=0)}
        )
        return self._postprocess(raw, orig_w, orig_h)

    def _postprocess(self, raw_output: dict, orig_w: int, orig_h: int):
        boxes_list, scores_list = [], []
        for _, output_data in raw_output.items():
            # Hailo NMS 出力: list[batch] → list[80クラス] → ndarray(N,5)
            batch = output_data[0] if isinstance(output_data, list) else output_data
            for class_id, class_dets in enumerate(batch):
                if class_id != 0:          # person クラスのみ
                    continue
                if class_dets is None or len(class_dets) == 0:
                    continue
                for det in class_dets:
                    score = float(det[4])
                    if score < self.conf_threshold:
                        continue
                    y_min, x_min, y_max, x_max = det[0], det[1], det[2], det[3]
                    boxes_list.append([
                        float(x_min) * orig_w, float(y_min) * orig_h,
                        float(x_max) * orig_w, float(y_max) * orig_h,
                    ])
                    scores_list.append(score)
        if boxes_list:
            return np.array(boxes_list, dtype=np.float32), np.array(scores_list, dtype=np.float32)
        return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32)


# ============================================================
# 4. 基本設定
# ============================================================
#YOLO_HEF_PATH     = "/home/yaslab/models/yolov11s.hef"
YOLO_HEF_PATH     = "/home/yaslab/models/yolov11m.hef"
REID_HEF_PATH     = "/home/yaslab/models/repvgg_a0_person_reid_2048.hef"
CONF_THRESHOLD    = 0.5
#TARGET_RESOLUTION = (640, 480)
TARGET_RESOLUTION = (1280, 720)
#TARGET_RESOLUTION = (1920, 1080)

#境界線
ZONE_BOUNDARY = 0.60

# カメラの向き（取り付け状態に合わせて変更）
FLIP_HORIZONTAL = True
FLIP_VERTICAL   = True

# ============================================================
# 5. Re-ID 設定
# ============================================================
ENABLE_REID             = True
# REID_INFERENCE_INTERVAL = 5
# REID_MAX_DETECTIONS     = 8

# ============================================================
# 6. 入退室カウンター設定
# ============================================================
ENABLE_PHOTO_CAPTURE = True
PHOTO_SAVE_DIR       = "/home/yaslab/bus-passenger-counting/entry_photos/"
PHOTO_QUALITY        = 95
CAPTURE_DELAY        = 0.5

# ============================================================
# 7. カウント管理変数
# ============================================================
entry_count              = 0
exit_count               = 0
current_count            = 0
disappear_threshold      = 2.0
track_states             = {}
track_history            = {}
display_buffer           = {}
stable_tracks            = {}
track_stability_threshold = 3
photo_capture_queue      = []
photo_count              = 0
last_photo_time          = 0
fps_counter              = deque(maxlen=30)
last_frame_time          = time.time()
current_fps              = 0.0
frame_count              = 0
# reid_frame_counter       = 0
detection_history        = deque(maxlen=10)
track_count_history      = deque(maxlen=15)
display_count_history    = deque(maxlen=15)

# Flask 用
latest_frame = None
frame_lock   = threading.Lock()
flask_ready  = False
server_url   = None

# ============================================================
# 8. BoT-SORT 初期化
# ============================================================
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track_high_thresh",  type=float, default=0.5)
    parser.add_argument("--track_low_thresh",   type=float, default=0.1)
    parser.add_argument("--new_track_thresh",   type=float, default=0.6)
    parser.add_argument("--track_buffer",       type=int,   default=20)
    parser.add_argument("--match_thresh",       type=float, default=0.8)
    parser.add_argument("--device",             type=str,   default="cpu")
    parser.add_argument("--cmc_method",         type=str,   default="None")
    parser.add_argument("--name",               type=str,   default="BoTSORT")
    parser.add_argument("--ablation",           action="store_true", default=False)
    parser.add_argument("--with_reid",          action="store_true", default=False)
    parser.add_argument("--fast_reid_config",   type=str,   default="")
    parser.add_argument("--fast_reid_weights",  type=str,   default="")
    parser.add_argument("--proximity_thresh",   type=float, default=0.5)
    parser.add_argument("--appearance_thresh",  type=float, default=0.25)
    parser.add_argument("--mot20",              action="store_true", default=False)
    return parser.parse_known_args()[0]

bot_args           = get_args()
bot_args.with_reid = True    # Hailo Re-ID を有効化

# ============================================================
# 9. 共有 VDevice 作成（Hailo-8 は1つなので必ず共有する）
# ============================================================
from hailo_platform import HailoSchedulingAlgorithm

print("Initializing Hailo-8 VDevice (with scheduler)...")
params = VDevice.create_params()
params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
params.multi_process_service = True   # ← hailort_service経由で接続
shared_vdevice = VDevice(params)
print("✓ Shared VDevice created (scheduler: ROUND_ROBIN).")

# Re-IDモデルにも shared_vdevice を渡すためカスタム引数として設定
bot_args.hailo_vdevice = shared_vdevice
tracker = BoTSORT(bot_args, frame_rate=30)
print("✓ BoT-SORT initialized (Hailo Re-ID enabled)")

# ============================================================
# 10. YOLO モデル初期化
# ============================================================
print("Loading yolov11s on Hailo-8...")
model = HailoYOLODetector(YOLO_HEF_PATH, vdevice=shared_vdevice, conf_threshold=CONF_THRESHOLD)
print("✓ yolov11s loaded successfully.")

# ============================================================
# 10. Picamera2 初期化
# ============================================================
def init_camera():
    print("Initializing Picamera2...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": TARGET_RESOLUTION, "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)
    print("✓ Picamera2 initialized.")
    return picam2

# ============================================================
# 11. 反転処理
# ============================================================
def apply_flip(frame: np.ndarray) -> np.ndarray:
    if FLIP_HORIZONTAL and FLIP_VERTICAL:
        return cv2.flip(frame, -1)
    elif FLIP_HORIZONTAL:
        return cv2.flip(frame, 1)
    elif FLIP_VERTICAL:
        return cv2.flip(frame, 0)
    return frame

# ============================================================
# 12. 写真保存機能
# ============================================================
def setup_photo_directory():
    if ENABLE_PHOTO_CAPTURE:
        try:
            os.makedirs(PHOTO_SAVE_DIR, exist_ok=True)
            print(f"写真保存先: {PHOTO_SAVE_DIR}")
            return True
        except Exception as e:
            print(f"写真ディレクトリ作成エラー: {e}")
            return False
    return True

def schedule_photo_capture(track_id, entry_time, action="Entry"):
    if not ENABLE_PHOTO_CAPTURE:
        return
    photo_capture_queue.append({
        "track_id":     track_id,
        "capture_time": entry_time + CAPTURE_DELAY,
        "entry_time":   entry_time,
        "action":       action      # ← 入室/退室を記録
    })
    icon = "🚪" if action == "Entry" else "👋"
    print(f"📸 {icon} 写真撮影をスケジュール: ID={track_id}, {CAPTURE_DELAY}秒後")

def capture_entry_photo(frame, track_id, entry_time, action="Entry"):
    global photo_count
    if not ENABLE_PHOTO_CAPTURE:
        return False
    try:
        prefix    = "entry" if action == "Entry" else "exit"   # ← ファイル名を変更
        timestamp = datetime.datetime.fromtimestamp(entry_time)
        base      = (f"{prefix}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
                     f"_ID{track_id:03d}_{photo_count+1:04d}")
#                     f"{TARGET_RESOLUTION[0]}x{TARGET_RESOLUTION[1]}")
        filepath  = os.path.join(PHOTO_SAVE_DIR, f"{base}.jpg")
        success   = cv2.imwrite(filepath, frame, [
            cv2.IMWRITE_JPEG_QUALITY,     PHOTO_QUALITY,
            cv2.IMWRITE_JPEG_OPTIMIZE,    1,
            cv2.IMWRITE_JPEG_PROGRESSIVE, 1
        ])
        if success:
            photo_count += 1
            file_size    = os.path.getsize(filepath) / 1024
            print(f"✓ 写真保存成功: {os.path.basename(filepath)} ({file_size:.1f}KB)")
            return True
        print(f"✗ 写真保存失敗: {filepath}")
        return False
    except Exception as e:
        print(f"写真保存エラー: {e}")
        return False

def process_photo_capture_queue(frame, current_time):
    global photo_capture_queue
    if not photo_capture_queue:
        return
    completed = []
    for i, info in enumerate(photo_capture_queue):
        if current_time >= info["capture_time"]:
            capture_entry_photo(frame, info["track_id"], info["entry_time"], info.get("action", "Entry"))
            completed.append(i)
    for i in reversed(completed):
        photo_capture_queue.pop(i)

# ============================================================
# 13. FPS 計測
# ============================================================
def calculate_fps():
    global fps_counter, last_frame_time, current_fps
    current_time    = time.time()
    frame_time      = current_time - last_frame_time
    last_frame_time = current_time
    if 0.01 < frame_time < 1.0:
        fps_counter.append(frame_time)
    if fps_counter:
        avg         = sum(fps_counter) / len(fps_counter)
        current_fps = 1.0 / avg if avg > 0 else 0.0
    return current_fps

# ============================================================
# 14. ログ記録
# ============================================================
def log_entry_exit(action, track_id, cnt):
    now     = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    log_str = f"{now_str} - {action} ID:{track_id:03d} (Current: {cnt}人)"
    icon    = "🚪" if action == "Entry" else "👋"
    print(f"{icon} {log_str}")
    try:
        with open("count.txt", "a", encoding="utf-8") as f:
            f.write(log_str + "\n")
    except Exception as e:
        print(f"ログ保存エラー: {e}")
    # 入室・退室どちらも撮影
    schedule_photo_capture(track_id, now.timestamp(), action)

# ============================================================
# 15. ネットワーク
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

def open_browser_when_ready():
    global flask_ready, server_url
    print("🌐 Flaskサーバーの起動を待機中...")
    while not flask_ready:
        time.sleep(0.5)
    time.sleep(1.5)
    print(f"🌐 ブラウザを自動起動: {server_url}")
    try:
        import webbrowser
        webbrowser.open(server_url)
        print("✓ ブラウザを開きました")
    except Exception as e:
        print(f"⚠ ブラウザ自動起動失敗: {e}")
        print(f"   手動でアクセス: {server_url}")

# ============================================================
# 16. Flask 設定
# ============================================================
app = Flask(__name__)

@app.route('/')
def index():
    return '''
    <!doctype html>
    <html>
    <head>
        <title>入退室カウンター - BoT-SORT (Raspberry Pi)</title>
        <meta charset="UTF-8">
        <style>
            body {
                margin: 0; padding: 20px;
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .container { max-width: 1200px; margin: 0 auto; text-align: center; }
            h1 { font-size: 2.5em; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
            .subtitle { font-size: 1.2em; color: #a8d8ff; margin-bottom: 30px; }
            .video {
                max-width: 100%; height: auto;
                border: 3px solid #4a90e2; border-radius: 12px;
                box-shadow: 0 8px 16px rgba(0,0,0,0.3);
            }
            .stats {
                margin-top: 30px; font-size: 24px;
                background: rgba(255,255,255,0.1); padding: 20px;
                border-radius: 12px; backdrop-filter: blur(10px);
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px; margin-top: 15px;
            }
            .stat-item {
                background: rgba(255,255,255,0.15); padding: 15px;
                border-radius: 8px; transition: transform 0.2s;
            }
            .stat-item:hover { transform: translateY(-3px); background: rgba(255,255,255,0.25); }
            .stat-value { font-size: 2em; font-weight: bold; color: #4affff; }
            .stat-label { font-size: 0.9em; color: #c0e0ff; margin-top: 5px; }
        </style>
        <script>
            function updateStats() {
                fetch('/stats').then(r => r.json()).then(data => {
                    document.getElementById('entry-value').textContent   = data.entry;
                    document.getElementById('exit-value').textContent    = data.exit;
                    document.getElementById('current-value').textContent = data.current;
                    document.getElementById('photos-value').textContent  = data.photos;
                    document.getElementById('fps-value').textContent     = data.fps;
                });
            }
            setInterval(updateStats, 1000);
            window.onload = updateStats;
        </script>
    </head>
    <body>
        <div class="container">
            <h1>🚌 乗降客計数システム</h1>
            <div class="subtitle">YOLO11 + BoT-SORT with Re-ID on Raspberry Pi 5 + Hailo-8</div>
            <img src="/video_feed" class="video" alt="リアルタイム映像">
            <div class="stats">
                <div style="font-size: 1.5em; margin-bottom: 15px;">📊 統計情報</div>
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
                        <div class="stat-value" id="fps-value">-</div>
                        <div class="stat-label">FPS</div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/stats')
def stats():
    from flask import jsonify
    return jsonify({
        'entry':   entry_count,
        'exit':    exit_count,
        'current': current_count,
        'photos':  photo_count,
        'fps':     f"{current_fps:.1f}"
    })

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if latest_frame is not None:
                    ret, jpeg = cv2.imencode(
                        '.jpg', latest_frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 75]
                    )
                    if ret:
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                               + jpeg.tobytes() + b'\r\n')
            time.sleep(0.05)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ============================================================
# 17. メイン処理
# ============================================================
def process_frame(frame):
    """frame: BGR フレーム"""
    global latest_frame, frame_count, entry_count, exit_count, current_count
    global track_states, track_history, display_buffer
    global stable_tracks, detection_history, track_count_history, display_count_history

    frame_count        += 1
    # reid_frame_counter += 1
    h, w = frame.shape[:2]
    current_time = time.time()
    calculate_fps()

    # ゾーン定義
    # door_x1         = int(w * 0.35)
    # door_x2         = int(w * 0.63)
    # room_zone_left  = (0, door_x1)
    # room_zone_right = (door_x2, w)
    boundary_x    = int(w * ZONE_BOUNDARY)  # 境界線X座標
    inside_zone   = (0, boundary_x)         # 左側：inside
    door_zone     = (boundary_x, w)         # 右側：door

    # ── YOLO 推論（Hailo） ──────────────────────────────────
    boxes, scores = model.predict(frame)

    if len(boxes) > 0:
        detections = np.column_stack((boxes, scores))
    else:
        detections = np.empty((0, 5), dtype=np.float32)

    detection_history.append(len(detections))

    # ── Re-ID 適応制御 ──────────────────────────────────────
    tracker.with_reid = ENABLE_REID

    # ── トラッキング実行 ────────────────────────────────────
    try:
        online_targets = tracker.update(detections, frame)
    except Exception as e:
        print(f"Tracker error: {e}")
        online_targets = []

    track_count_history.append(len(online_targets))

    current_ids = set()
    now = datetime.datetime.now().timestamp()

    # ── 軌跡履歴更新 ────────────────────────────────────────
    Max_HISTORY_LENGTH = 30
    current_track_ids  = set(
        t.track_id for t in online_targets if hasattr(t, 'track_id')
    )
    for track_id in list(track_history.keys()):
        if track_id not in current_track_ids:
            if track_history[track_id]:
                last_frame_num = track_history[track_id][-1]["frame"]
                if frame_count - last_frame_num > 60:
                    del track_history[track_id]
                    stable_tracks.pop(track_id, None)

    # ── 軌跡描画 ────────────────────────────────────────────
    for track_id in current_track_ids:
        if track_id not in track_history or len(track_history[track_id]) < 2:
            continue
        trail = track_history[track_id][-Max_HISTORY_LENGTH:]
        for i in range(1, len(trail)):
            pt1_box   = trail[i - 1]["box"]
            pt2_box   = trail[i]["box"]
            pt1       = ((pt1_box[0] + pt1_box[2]) // 2, (pt1_box[1] + pt1_box[3]) // 2)
            pt2       = ((pt2_box[0] + pt2_box[2]) // 2, (pt2_box[1] + pt2_box[3]) // 2)
            intensity = int(255 * (i / len(trail)))
            cv2.line(frame, pt1, pt2, (0, intensity, intensity), 2)

    # ── トラック処理・入退室判定 ─────────────────────────────
    for t in online_targets:
        track_id             = t.track_id
        x1, y1, w_box, h_box = map(int, t.tlwh)
        x2, y2               = x1 + w_box, y1 + h_box
        cx                   = (x1 + x2) // 2

        current_ids.add(track_id)

        if track_id not in track_history:
            track_history[track_id] = []
        track_history[track_id].append({
            "frame": frame_count, "box": (x1, y1, x2, y2), "timestamp": now
        })
        track_history[track_id] = track_history[track_id][-30:]

        if len(track_history[track_id]) >= track_stability_threshold:
            stable_tracks[track_id] = True

        if track_id in stable_tracks:
            if track_id in display_buffer:
                old = display_buffer[track_id]["box"]
                a   = 0.7
                x1  = int(a * x1 + (1 - a) * old[0])
                y1  = int(a * y1 + (1 - a) * old[1])
                x2  = int(a * x2 + (1 - a) * old[2])
                y2  = int(a * y2 + (1 - a) * old[3])
                display_buffer[track_id].update({
                    "box": (x1, y1, x2, y2), "last_seen": frame_count,
                    "confidence": min(display_buffer[track_id].get("confidence", 0.5) + 0.1, 1.0)
                })
            else:
                display_buffer[track_id] = {
                    "box": (x1, y1, x2, y2), "last_seen": frame_count, "confidence": 0.5
                }

        # ゾーン判定
        # if door_x1 <= cx <= door_x2:
        #     current_zone = "door"
        # elif (room_zone_left[0] <= cx < room_zone_left[1]) or \
        #      (room_zone_right[0] < cx <= room_zone_right[1]):
        #     current_zone = "room"
        # else:
        #     current_zone = "none"
        if cx < boundary_x:
            current_zone = "inside"
        else:
            current_zone = "door"

        # 状態管理
        state = track_states.get(track_id, {
            "entered": False, "exited": False, "was_inside": False,
            "last_zone": "none", "first_zone": None,
            "last_seen": now, "ever_detected": False,
        })

        if not state["ever_detected"]:
            state["first_zone"] = current_zone
        if current_zone == "inside":
            state["was_inside"] = True

        # 入室判定
        if (not state["entered"] and
                state["first_zone"] == "door" and
                current_zone == "inside"):
            entry_count   += 1
            current_count += 1
            state["entered"] = True
            log_entry_exit("Entry", track_id, current_count)

        state["last_zone"]     = current_zone
        state["last_seen"]     = now
        state["ever_detected"] = True
        track_states[track_id] = state

        # 描画（安定トラックのみ）
        if track_id in stable_tracks:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id:03d}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    display_count_history.append(len(display_buffer))

    # ── 退室判定 ────────────────────────────────────────────
    for track_id, state in list(track_states.items()):
        if track_id not in current_ids:
            time_since = now - state["last_seen"]
            if (not state["exited"] and
                    state["was_inside"] and
                    state["last_zone"] == "door" and
                    time_since > disappear_threshold):
                exit_count    += 1
                current_count  = max(0, current_count - 1)
                state["exited"] = True
                log_entry_exit("Exit", track_id, current_count)
                track_states[track_id] = state

    # ── ゾーン描画 ──────────────────────────────────────────
    cv2.line(frame, (boundary_x, 0), (boundary_x, h), (0, 255, 255), 2)
    cv2.putText(frame, "Inside", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "Door", (boundary_x + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Re-ID 状態・FPS 表示
    reid_status = "ON" if (ENABLE_REID and tracker.with_reid) else "OFF"
    reid_color  = (0, 255, 0) if tracker.with_reid else (128, 128, 128)
    cv2.putText(frame, f"ReID: {reid_status}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, reid_color, 2)
    cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    # 写真撮影キュー処理
    process_photo_capture_queue(frame, current_time)

    with frame_lock:
        latest_frame = frame.copy()


# ============================================================
# 18. メイン実行
# ============================================================
def main():
    global ENABLE_PHOTO_CAPTURE, flask_ready, server_url

    print("=" * 60)
    print("入退室カウンター (YOLO11 + BoT-SORT on Raspberry Pi 5 + Hailo-8)")
    print("=" * 60)
    print(f"Re-ID機能: {'ON (常時)' if ENABLE_REID else 'OFF'}")
    print(f"写真撮影機能: {'ON' if ENABLE_PHOTO_CAPTURE else 'OFF'}")
    print(f"解像度: {TARGET_RESOLUTION[0]}x{TARGET_RESOLUTION[1]}")
    print("=" * 60)

    if not setup_photo_directory():
        ENABLE_PHOTO_CAPTURE = False

    local_ip   = get_local_ip()
    server_url = f"http://{local_ip}:8080/"

    flask_thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=8080, threaded=True, debug=False),
        daemon=True
    )
    flask_thread.start()
    time.sleep(1.0)
    flask_ready = True

    browser_thread = threading.Thread(target=open_browser_when_ready, daemon=True)
    browser_thread.start()

    # カメラ起動
    picam2 = init_camera()

    # Hailo YOLO パイプラインを開く
    model.open()

    print("=" * 60)
    print(f"✓ システム起動完了")
    print(f"✓ アクセスURL: {server_url}")
    print("=" * 60)
    print("操作方法:")
    print("  Ctrl+C : プログラム終了")
    print("  ブラウザ: リアルタイム映像と統計表示")
    print("=" * 60)

    try:
        while True:
            # RGB888 で取得 → 反転 → BGR に変換して処理
            frame = picam2.capture_array()
            frame = apply_flip(frame)
            #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            process_frame(frame)

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("プログラムを終了します")
        print("=" * 60)
        print("📊 最終統計:")
        print(f"  総入室数  : {entry_count}人")
        print(f"  総退室数  : {exit_count}人")
        print(f"  現在の在室: {current_count}人")
        if ENABLE_PHOTO_CAPTURE:
            print(f"  撮影枚数  : {photo_count}枚")
            if photo_count > 0:
                print(f"  保存場所  : {PHOTO_SAVE_DIR}")
                try:
                    total_size = sum(
                        os.path.getsize(os.path.join(PHOTO_SAVE_DIR, f))
                        for f in os.listdir(PHOTO_SAVE_DIR) if f.endswith('.jpg')
                    )
                    print(f"  総サイズ  : {total_size/1024/1024:.2f} MB")
                except:
                    pass
        print("=" * 60)

    except Exception as e:
        print(f"✖ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

    finally:
        model.close()
        picam2.stop()
        print("✓ クリーンアップ完了")


if __name__ == "__main__":
    main()