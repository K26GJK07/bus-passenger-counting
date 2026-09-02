#!/usr/bin/env python3
"""
テスト①：yolov11s による人物検出テスト（映像確認付き）
Hailo-8 上で yolov11s.hef を使い、Picamera2 のフレームから人物を検出します。

実行方法:
    python3 test_yolo.py

ブラウザで映像確認:
    http://[Raspberry PiのIP]:5000

確認ポイント:
    - ブラウザで映像が表示されること
    - 人物の周囲に緑のバウンディングボックスが表示されること
    - ターミナルに FPS と検出数が表示されること
"""

import numpy as np
import cv2
import time
import threading
import socket
from flask import Flask, Response
from picamera2 import Picamera2
from hailo_platform import (
    HEF,
    VDevice,
    HailoStreamInterface,
    InferVStreams,
    ConfigureParams,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType,
)

# ── 設定 ────────────────────────────────────────────────────────────────
HEF_PATH        = "/home/yaslab/models/yolov11s.hef"
CONF_THRESHOLD  = 0.5        # 信頼度スコアの閾値
TARGET_SIZE     = (640, 640) # yolov11s の入力サイズ
CAMERA_SIZE     = (640, 480) # カメラ解像度
FLASK_PORT      = 5000

# カメラの向きを調整（取り付け状態に合わせて True/False を変更）
FLIP_HORIZONTAL = True   # 左右反転
FLIP_VERTICAL   = True   # 上下反転
# ────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

# スレッド間で共有するフレーム
latest_frame = None
frame_lock   = threading.Lock()
stats        = {"fps": 0.0, "persons": 0, "frame_count": 0}


# ── カメラ初期化 ──────────────────────────────────────────────────────────
def init_camera():
    print("[1/3] カメラを初期化中...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": CAMERA_SIZE, "format": "RGB888"},
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)
    print("✓ カメラ初期化完了")
    return picam2


# ── Hailo モデルロード ────────────────────────────────────────────────────
def load_hailo_model(hef_path: str):
    print(f"[2/3] モデルをロード中: {hef_path}")
    hef = HEF(hef_path)
    target = VDevice()
    configure_params = ConfigureParams.create_from_hef(
        hef=hef,
        interface=HailoStreamInterface.PCIe
    )
    network_group        = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()

    input_params = InputVStreamParams.make(
        network_group,
        quantized=False,
        format_type=FormatType.UINT8
    )
    output_params = OutputVStreamParams.make(
        network_group,
        quantized=False,
        format_type=FormatType.FLOAT32
    )
    print("✓ モデルロード完了")
    return target, hef, network_group, network_group_params, input_params, output_params


# ── 前処理 ───────────────────────────────────────────────────────────────
def preprocess(frame: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    カメラフレームをモデル入力形式に変換
    カメラは RGB888 で取得 → そのままモデルへ（色変換不要）
    """
    resized = cv2.resize(frame, target_size)
    return resized.astype(np.uint8)


# ── 後処理 ───────────────────────────────────────────────────────────────
def parse_detections(raw_output: dict, orig_w: int, orig_h: int, conf_threshold: float):
    """
    Hailo NMS BY CLASS の出力をパース
    構造: dict → list[batch] → list[80クラス] → ndarray(N, 5)
    各検出: [y_min, x_min, y_max, x_max, score]（正規化座標）
    """
    detections = []
    for _, output_data in raw_output.items():

        # output_data は list[1]（バッチサイズ1）
        batch = output_data[0]  # list[80クラス]

        for class_id, class_detections in enumerate(batch):
            # class_detections は ndarray(N, 5) または空
            if class_detections is None:
                continue
            if not hasattr(class_detections, '__len__') or len(class_detections) == 0:
                continue

            for det in class_detections:
                score = float(det[4])
                if score < conf_threshold:
                    continue
                y_min, x_min, y_max, x_max = float(det[0]), float(det[1]), float(det[2]), float(det[3])
                detections.append({
                    "class_id": class_id,
                    "score":    score,
                    "box": (
                        int(x_min * orig_w),
                        int(y_min * orig_h),
                        int(x_max * orig_w),
                        int(y_max * orig_h),
                    )
                })
    return detections


# ── 描画 ─────────────────────────────────────────────────────────────────
def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    """人物（class_id=0）のバウンディングボックスを描画"""
    for det in detections:
        if det["class_id"] != 0:
            continue
        x1, y1, x2, y2 = det["box"]
        score = det["score"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"person {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


# ── 反転処理 ─────────────────────────────────────────────────────────────
def apply_flip(frame: np.ndarray) -> np.ndarray:
    """カメラの取り付け向きに合わせて反転"""
    if FLIP_HORIZONTAL and FLIP_VERTICAL:
        return cv2.flip(frame, -1)   # 上下左右同時反転
    elif FLIP_HORIZONTAL:
        return cv2.flip(frame, 1)    # 左右のみ
    elif FLIP_VERTICAL:
        return cv2.flip(frame, 0)    # 上下のみ
    return frame


# ── 推論ループ（別スレッドで動作） ───────────────────────────────────────
def inference_loop(picam2, network_group, ng_params, input_params, output_params):
    global latest_frame, stats

    print("\n[3/3] 推論ループ開始")
    print("ブラウザで映像を確認しながらカメラの前に立ってください。\n")

    with InferVStreams(network_group, input_params, output_params) as pipeline:
        with network_group.activate(ng_params):
            while True:
                t_start = time.time()

                # フレーム取得（RGB888）
                frame = picam2.capture_array()
                orig_h, orig_w = frame.shape[:2]

                # 反転補正
                frame = apply_flip(frame)

                # 前処理（RGB のままモデルへ）
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_data = preprocess(rgb_frame, TARGET_SIZE)
                input_batch = {
                    list(input_params.keys())[0]: np.expand_dims(input_data, axis=0)
                }

                # Hailo 推論
                raw_output = pipeline.infer(input_batch)

                # 後処理
                detections        = parse_detections(raw_output, orig_w, orig_h, CONF_THRESHOLD)
                person_detections = [d for d in detections if d["class_id"] == 0]

                # ── 表示用フレームの作成 ──
                # Picamera2 の RGB888 → OpenCV の BGR に変換（色を正しく表示するため）
                display = frame.copy()
                display = draw_detections(display, person_detections)

                # FPS・検出数をフレームに描画
                elapsed = time.time() - t_start
                fps     = 1.0 / elapsed if elapsed > 0 else 0
                cv2.putText(display,
                            f"FPS: {fps:.1f}  Persons: {len(person_detections)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

                # 共有フレームを更新
                with frame_lock:
                    latest_frame = display.copy()

                # ターミナルに統計を表示（30フレームごと）
                stats["fps"]          = fps
                stats["persons"]      = len(person_detections)
                stats["frame_count"] += 1
                if stats["frame_count"] % 30 == 0:
                    print(f"  FPS: {fps:5.1f} | 人物検出数: {len(person_detections)}")


# ── Flask ルーティング ────────────────────────────────────────────────────
def generate_mjpeg():
    """MJPEGストリームを生成してブラウザに送る"""
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
        <title>YOLO 人物検出テスト</title>
        <style>
            body {{ background:#111; color:#eee; font-family:sans-serif;
                   display:flex; flex-direction:column;
                   align-items:center; padding:20px; margin:0; }}
            h2   {{ color:#0f0; }}
            img  {{ border:2px solid #0f0; border-radius:8px; max-width:100%; }}
            p    {{ color:#aaa; font-size:14px; }}
        </style>
    </head>
    <body>
        <h2>🎯 yolov11s 人物検出テスト</h2>
        <img src="/video_feed">
        <p>接続中: {ip}:{FLASK_PORT} ／ 検出閾値: {CONF_THRESHOLD}</p>
        <p>映像が反転している場合は test_yolo.py の
           FLIP_HORIZONTAL / FLIP_VERTICAL を変更してください</p>
    </body>
    </html>
    '''


@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ── メイン ───────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("テスト①：yolov11s 人物検出テスト（映像確認付き）")
    print("=" * 50)

    # カメラ・モデル初期化
    picam2 = init_camera()
    target, hef, network_group, ng_params, input_params, output_params = \
        load_hailo_model(HEF_PATH)

    # 推論を別スレッドで起動
    t = threading.Thread(
        target=inference_loop,
        args=(picam2, network_group, ng_params, input_params, output_params),
        daemon=True
    )
    t.start()

    # アクセス先をターミナルに表示
    ip = socket.gethostbyname(socket.gethostname())
    print(f"\n{'='*50}")
    print(f"  ブラウザでアクセス → http://{ip}:{FLASK_PORT}")
    print(f"{'='*50}\n")
    print("終了するには Ctrl+C を押してください\n")

    # Flask サーバー起動（メインスレッド）
    try:
        app.run(host='0.0.0.0', port=FLASK_PORT, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
        print("\nテスト終了")


if __name__ == "__main__":
    main()