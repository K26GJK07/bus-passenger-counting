#!/usr/bin/env python3
"""
テスト：IMX500 (Raspberry Pi AI Camera) による人物検出テスト
network.rpk (yolo11n) を使い、IMX500内蔵NPUで人物検出を行います。

実行方法:
    python3 test_imx500.py

ブラウザで映像確認:
    http://[Raspberry PiのIP]:5000
"""

import numpy as np
import cv2
import time
import threading
import socket
from flask import Flask, Response
from picamera2 import Picamera2, MappedArray
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

# ── 設定 ────────────────────────────────────────────────────────────────
NETWORK_RPK_PATH = "/home/yaslab/models/yolo11n_imx_model/network.rpk"
CONF_THRESHOLD   = 0.3        # 検出閾値
CAMERA_SIZE      = (1280, 720)
FLASK_PORT       = 5000
MODEL_INPUT_SIZE = 640        # yolo11nの入力サイズ

# カメラの向き（取り付け状態に合わせて変更）
FLIP_HORIZONTAL = True
FLIP_VERTICAL   = True
# ────────────────────────────────────────────────────────────────────────

# ── グローバル変数 ────────────────────────────────────────────────────────
imx500          = None
picam2          = None
app             = Flask(__name__)
latest_frame    = None
frame_lock      = threading.Lock()
last_detections = []   # outputs=Noneのとき前回結果を使いまわす
stats           = {"fps": 0.0, "persons": 0, "frame_count": 0}

# FPS計測用
fps_counter     = []
last_fps_time   = time.time()
real_fps        = 0.0


# ── Detection クラス ──────────────────────────────────────────────────────
class Detection:
    """
    IMX500検出結果を保持するクラス
    box は imx500.convert_inference_coords の出力 = (x, y, w, h) ピクセル座標
    """
    def __init__(self, coords, category, conf, metadata):
        self.category = category
        self.conf     = conf
        # convert_inference_coords → (x, y, w, h) ピクセル座標を返す
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)


# ── 反転処理 ─────────────────────────────────────────────────────────────
def apply_flip(frame: np.ndarray) -> np.ndarray:
    if FLIP_HORIZONTAL and FLIP_VERTICAL:
        return cv2.flip(frame, -1)
    elif FLIP_HORIZONTAL:
        return cv2.flip(frame, 1)
    elif FLIP_VERTICAL:
        return cv2.flip(frame, 0)
    return frame


# ── 検出結果パース ────────────────────────────────────────────────────────
def parse_detections(metadata):
    """
    IMX500の推論結果から人物検出結果を解析
    outputs[0]: boxes     (300, 4) [y1, x1, y2, x2] ピクセル座標
    outputs[1]: scores    (300,)
    outputs[2]: class_ids (300,)
    """
    global last_detections

    outputs = imx500.get_outputs(metadata, add_batch=True)
    if outputs is None:
        # 推論結果がないフレームは前回の結果を使いまわす
        return last_detections

    boxes   = outputs[0][0]   # (300, 4)
    scores  = outputs[1][0]   # (300,)
    classes = outputs[2][0]   # (300,)

    # [y1, x1, y2, x2] → [x1, y1, x2, y2] に並び替えてモデル入力サイズで正規化
    # convert_inference_coords に渡す形式に合わせる
    boxes_reordered = boxes[:, [1, 0, 3, 2]] / MODEL_INPUT_SIZE

    detections = []
    for box, score, cls in zip(boxes_reordered, scores, classes):
        if score > CONF_THRESHOLD and int(cls) == 0:  # person のみ
            detections.append(Detection(box, cls, score, metadata))

    last_detections = detections
    return detections


# ── コールバック（フレームごとに呼ばれる） ────────────────────────────────
def draw_callback(request, stream="main"):
    global latest_frame
    fps = 0.0

    # 実際のフレームレートを計測
    now = time.time()
    fps_counter.append(now)
    fps_counter[:] = [t for t in fps_counter if now - t < 1.0]# 1秒以内のフレームのみ
    real_fps = len(fps_counter)  # 1秒間のフレーム数 = 実FPS

    with MappedArray(request, stream) as m:
        frame    = m.array.copy()   # 実際はBGR順
        metadata = request.get_metadata()
        h, w     = frame.shape[:2]

        t_start    = time.time()
        detections = parse_detections(metadata)

        # ── ①先に反転処理 ──────────────────────────────────
        frame = apply_flip(frame)

        # ── ②反転後のフレームに座標変換して描画 ────────────
        for det in detections:
            try:
                x, y, bw, bh = det.box
                x1 = int(x)
                y1 = int(y)
                x2 = int(x + bw)
                y2 = int(y + bh)

                # 反転に合わせて座標を変換
                if FLIP_HORIZONTAL and FLIP_VERTICAL:
                    x1, y1, x2, y2 = w - x2, h - y2, w - x1, h - y1
                elif FLIP_HORIZONTAL:
                    x1, x2 = w - x2, w - x1
                elif FLIP_VERTICAL:
                    y1, y2 = h - y2, h - y1

                # 座標クランプ
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                if x2 > x1 and y2 > y1:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, f"P {det.conf:.2f}",
                                (x1, max(y1 - 10, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except Exception as e:
                print(f"  [描画エラー] {e} box={det.box}")

        # ── ③FPS・検出数を描画 ─────────────────────────────
        elapsed = time.time() - t_start
        fps     = 1.0 / elapsed if elapsed > 0 else 0.0
        cv2.putText(frame, f"FPS:{real_fps:.0f} P:{len(detections)}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)

        # ── ④色変換不要でそのままFlaskへ ──────────────────
        with frame_lock:
            latest_frame = frame.copy()

    stats["fps"]          = fps
    stats["persons"]      = len(detections)
    stats["frame_count"] += 1
    if stats["frame_count"] % 30 == 0:
        print(f"  FPS: {real_fps:5.1f} | 人物検出数: {len(detections)}")


# ── Flask ─────────────────────────────────────────────────────────────────
def generate_mjpeg():
    """MJPEGストリームを生成"""
    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.05)
                continue
            _, buffer = cv2.imencode('.jpg', latest_frame,
                                     [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'
               + buffer.tobytes()
               + b'\r\n')
        time.sleep(0.033)


@app.route('/')
def index():
    ip = socket.gethostbyname(socket.gethostname())
    return f'''
    <html>
    <head>
        <title>IMX500 人物検出テスト</title>
        <style>
            body {{ background:#111; color:#eee; font-family:sans-serif;
                   display:flex; flex-direction:column;
                   align-items:center; padding:20px; }}
            h2   {{ color:#0f0; }}
            img  {{ border:2px solid #0f0; border-radius:8px; max-width:100%; }}
            p    {{ color:#aaa; font-size:14px; }}
        </style>
    </head>
    <body>
        <h2>🎯 IMX500 人物検出テスト (yolo11n)</h2>
        <img src="/video_feed">
        <p>接続中: {ip}:{FLASK_PORT} ／ 検出閾値: {CONF_THRESHOLD}</p>
        <p>映像が反転している場合は FLIP_HORIZONTAL / FLIP_VERTICAL を変更してください</p>
    </body>
    </html>
    '''


@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ── メイン ────────────────────────────────────────────────────────────────
def main():
    global imx500, picam2

    print("=" * 55)
    print("IMX500 人物検出テスト (yolo11n)")
    print("=" * 55)
    print(f"モデル   : {NETWORK_RPK_PATH}")
    print(f"解像度   : {CAMERA_SIZE[0]}x{CAMERA_SIZE[1]}")
    print(f"検出閾値 : {CONF_THRESHOLD}")
    print("=" * 55)

    # ── IMX500初期化 ──────────────────────────────────────
    print("\nIMX500にモデルをロード中...")
    print("（初回は数分かかる場合があります）")
    imx500     = IMX500(NETWORK_RPK_PATH)
    intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
    intrinsics.task           = "object detection"
    intrinsics.inference_rate = 30
    intrinsics.update_with_defaults()
    print("✓ IMX500モデルロード完了")

    # ── カメラ初期化 ──────────────────────────────────────
    print("\nPicamera2を初期化中...")
    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        main={"size": CAMERA_SIZE, "format": "RGB888"},
        buffer_count=12,
        controls={"FrameRate": 30}
    )

    # コールバック登録
    picam2.pre_callback = draw_callback

    # カメラ起動
    picam2.start(config, show_preview=False)
    print("✓ カメラ起動完了")

    # ── アクセス先表示 ────────────────────────────────────
    ip = socket.gethostbyname(socket.gethostname())
    print(f"\n{'='*55}")
    print(f"  ブラウザでアクセス → http://{ip}:{FLASK_PORT}")
    print(f"{'='*55}\n")
    print("終了するには Ctrl+C を押してください\n")

    # ── Flask起動（メインスレッド） ───────────────────────
    try:
        app.run(host='0.0.0.0', port=FLASK_PORT, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
        print("\nテスト終了")


if __name__ == "__main__":
    main()