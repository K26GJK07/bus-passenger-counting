import sys
import collections
import os
import cv2
import threading
import time
import numpy as np
import argparse
import datetime
import statistics
from collections import deque
from picamera2 import Picamera2

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
YOLO_HEF_PATH     = "/home/yaslab/models/yolov11n.hef"
#YOLO_HEF_PATH     = "/home/yaslab/models/yolov11s.hef"
#YOLO_HEF_PATH     = "/home/yaslab/models/yolov11m.hef"
REID_HEF_PATH     = "/home/yaslab/models/repvgg_a0_person_reid_2048.hef"
CONF_THRESHOLD    = 0.5
#TARGET_RESOLUTION = (640, 480)
TARGET_RESOLUTION = (800, 600)
#TARGET_RESOLUTION = (1280, 720)
#TARGET_RESOLUTION = (1920, 1080)

#境界線
ZONE_BOUNDARY = 0.4

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
# （写真撮影機能は廃止）

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
fps_counter              = deque(maxlen=30)
last_frame_time          = time.time()
current_fps              = 0.0
frame_count              = 0
# reid_frame_counter       = 0
detection_history        = deque(maxlen=10)
track_count_history      = deque(maxlen=15)
display_count_history    = deque(maxlen=15)

# ============================================================
# 7.1 セッション（実行単位）ログ設定
# ============================================================
# 録画・検出ログ・処理時間ログ・入退室ログ・最終サマリーを
# 実行1回ごとのフォルダにまとめて保存する（上書き防止）
SESSION_BASE_DIR = "/home/yaslab/sessions/"

ENABLE_RECORDING     = True
RECORD_FOURCC        = "mp4v"   # Pi5にH.264ハードウェアエンコーダーは無いため軽量なmp4vを既定に
                                 # （libx264が使える環境なら "avc1" に変更可。ただしCPU負荷は増加）
RECORD_FPS           = 15       # 動画ファイルの再生fps（実際のフレーム到着間隔とは独立）
RECORD_QUEUE_MAXSIZE = 60       # キュー上限（=約4秒分@15fps）。超過時は古いフレームを破棄

import queue

# セッション開始時に init_session() が実際のパスを設定する
SESSION_DIR         = None
SESSION_START_TIME  = None
RECORDING_FILE_PATH = None
TIMING_LOG_PATH     = None
DETECTION_LOG_PATH  = None
COUNT_LOG_PATH       = None
SUMMARY_LOG_PATH     = None

record_queue          = None    # queue.Queue（フレーム＋メタ情報）
record_thread         = None
video_writer          = None
recording_frame_count = 0     # 実際に書き込んだフレーム数
recording_dropped     = 0     # キュー溢れで破棄したフレーム数
detection_log_file    = None  # 検出ログCSVファイルハンドル

def init_session():
    """実行1回分のセッションフォルダとログパスを準備する"""
    global SESSION_DIR, SESSION_START_TIME
    global RECORDING_FILE_PATH, TIMING_LOG_PATH, DETECTION_LOG_PATH
    global COUNT_LOG_PATH, SUMMARY_LOG_PATH

    SESSION_START_TIME = datetime.datetime.now()
    timestamp  = SESSION_START_TIME.strftime("%Y%m%d_%H%M%S")
    SESSION_DIR = os.path.join(SESSION_BASE_DIR, f"session_{timestamp}")
    os.makedirs(SESSION_DIR, exist_ok=True)

    RECORDING_FILE_PATH = os.path.join(SESSION_DIR, "recording.mp4")
    TIMING_LOG_PATH      = os.path.join(SESSION_DIR, "timing_log.csv")
    DETECTION_LOG_PATH   = os.path.join(SESSION_DIR, "detection_log.csv")
    COUNT_LOG_PATH        = os.path.join(SESSION_DIR, "count.txt")
    SUMMARY_LOG_PATH      = os.path.join(SESSION_DIR, "summary.txt")

    print(f"✓ セッションフォルダ: {SESSION_DIR}")

# ============================================================
# 7.5 処理時間計測用変数
# ============================================================
timing_det      = deque(maxlen=300)   # 検出 (ms)
timing_trk      = deque(maxlen=300)   # 追跡+Re-ID (ms)
timing_zone     = deque(maxlen=300)   # ゾーン判定+描画 (ms)
timing_total    = deque(maxlen=300)   # フレーム全体 (ms)
timing_file     = None                # CSVファイルハンドル

def init_timing_log():
    """処理時間ログCSVを初期化"""
    global timing_file
    timing_file = open(TIMING_LOG_PATH, "w", encoding="utf-8")
    timing_file.write("frame,det_ms,trk_reid_ms,zone_ms,total_ms,num_det,num_track,fps\n")
    timing_file.flush()
    print(f"✓ 処理時間ログ: {TIMING_LOG_PATH}")

def close_timing_log():
    """処理時間ログを閉じてサマリーを出力"""
    global timing_file
    if timing_file:
        timing_file.close()
        timing_file = None
    print_timing_summary()

def print_timing_summary():
    """処理時間の統計サマリーを出力"""
    print("\n" + "=" * 60)
    print("処理時間サマリー")
    print("=" * 60)
    for name, buf in [("検出 (YOLO11n)", timing_det),
                      ("追跡+Re-ID (BoT-SORT)", timing_trk),
                      ("ゾーン判定+描画", timing_zone),
                      ("フレーム全体", timing_total)]:
        if len(buf) < 2:
            print(f"  {name}: データ不足")
            continue
        vals = list(buf)
        avg  = statistics.mean(vals)
        sd   = statistics.stdev(vals)
        mn   = min(vals)
        mx   = max(vals)
        med  = statistics.median(vals)
        p95  = sorted(vals)[int(len(vals) * 0.95)]
        print(f"  {name}:")
        print(f"    平均: {avg:.2f} ms  標準偏差: {sd:.2f} ms")
        print(f"    中央値: {med:.2f} ms  95%ile: {p95:.2f} ms")
        print(f"    最小: {mn:.2f} ms  最大: {mx:.2f} ms")
        print(f"    サンプル数: {len(vals)}")
    print("=" * 60)

# ============================================================
# 7.6 非同期録画：書き込みスレッド・検出ログ
# ============================================================
def init_recording():
    """録画用キュー・VideoWriter・書き込みスレッドを初期化"""
    global record_queue, record_thread, video_writer

    if not ENABLE_RECORDING:
        return

    fourcc = cv2.VideoWriter_fourcc(*RECORD_FOURCC)
    video_writer = cv2.VideoWriter(
        RECORDING_FILE_PATH, fourcc, RECORD_FPS, TARGET_RESOLUTION
    )
    if not video_writer.isOpened():
        print(f"✖ VideoWriterのオープンに失敗しました: {RECORDING_FILE_PATH}")
        video_writer = None
        return

    record_queue = queue.Queue(maxsize=RECORD_QUEUE_MAXSIZE)
    record_thread = threading.Thread(
        target=recording_writer_loop, daemon=True, name="RecordingWriter"
    )
    record_thread.start()

    print(f"✓ 録画開始: {RECORDING_FILE_PATH}")
    print(f"  解像度: {TARGET_RESOLUTION[0]}x{TARGET_RESOLUTION[1]} / "
          f"FourCC: {RECORD_FOURCC} / 記録fps: {RECORD_FPS}")


def recording_writer_loop():
    """録画専用スレッド：キューから生フレームを取り出しディスクへ書き込む"""
    global recording_frame_count
    while True:
        item = record_queue.get()
        if item is None:   # 終了シグナル
            break
        raw_frame, _frame_count, _timestamp = item
        try:
            video_writer.write(raw_frame)
            recording_frame_count += 1
        except Exception as e:
            print(f"✖ 録画書き込みエラー: {e}")
        finally:
            record_queue.task_done()


def enqueue_recording_frame(raw_frame, f_count, ts):
    """検出処理から呼ばれる：生フレームを録画キューへ投入（非ブロッキング）"""
    global recording_dropped
    if not ENABLE_RECORDING or record_queue is None:
        return
    try:
        record_queue.put_nowait((raw_frame, f_count, ts))
    except queue.Full:
        # 古いフレームを1枚捨ててから最新を投入（検出処理を優先し録画側の遅延を防ぐ）
        try:
            record_queue.get_nowait()
            record_queue.task_done()
        except queue.Empty:
            pass
        try:
            record_queue.put_nowait((raw_frame, f_count, ts))
        except queue.Full:
            pass
        recording_dropped += 1


def close_recording():
    """録画スレッドを安全に停止し、VideoWriterを解放"""
    global video_writer
    if not ENABLE_RECORDING or record_queue is None:
        return
    print("録画を終了しています（キューの残りを書き込み中）...")
    record_queue.put(None)   # 終了シグナル
    if record_thread is not None:
        record_thread.join(timeout=30)
    if video_writer is not None:
        video_writer.release()
        video_writer = None
    print(f"✓ 録画終了: 書き込み {recording_frame_count} フレーム / "
          f"破棄 {recording_dropped} フレーム")


def init_detection_log():
    """オフライン再構成用の検出ログCSVを初期化"""
    global detection_log_file
    if not ENABLE_RECORDING:
        return
    detection_log_file = open(DETECTION_LOG_PATH, "w", encoding="utf-8")
    detection_log_file.write(
        "frame,timestamp,track_id,x1,y1,x2,y2,zone,stable\n"
    )
    detection_log_file.flush()
    print(f"✓ 検出ログ（オフライン再構成用）: {DETECTION_LOG_PATH}")


def log_detection(f_count, ts, track_id, x1, y1, x2, y2, zone, stable):
    """1トラック分の検出結果をログへ1行追加"""
    if detection_log_file is None:
        return
    detection_log_file.write(
        f"{f_count},{ts:.3f},{track_id},{x1},{y1},{x2},{y2},{zone},{int(stable)}\n"
    )
    if f_count % 30 == 0:
        detection_log_file.flush()


def close_detection_log():
    global detection_log_file
    if detection_log_file:
        detection_log_file.close()
        detection_log_file = None


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
# 12. FPS 計測
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
# 13. ログ記録
# ============================================================
def log_entry_exit(action, track_id, cnt):
    now     = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    log_str = f"{now_str} - {action} ID:{track_id:03d} (Current: {cnt}人)"
    icon    = "🚪" if action == "Entry" else "👋"
    print(f"{icon} {log_str}")
    try:
        with open(COUNT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(log_str + "\n")
    except Exception as e:
        print(f"ログ保存エラー: {e}")

# ============================================================
# 14. メイン処理 (処理時間計測付き)
# ============================================================
def process_frame(frame):
    """frame: BGR フレーム"""
    global frame_count, entry_count, exit_count, current_count
    global track_states, track_history, display_buffer
    global stable_tracks, detection_history, track_count_history, display_count_history

    frame_count        += 1
    h, w = frame.shape[:2]
    current_time = time.time()
    calculate_fps()

    # 描画前の生フレームを確保（録画用：ラベル・bbox等は一切焼き込まない）
    raw_frame_for_recording = frame.copy() if ENABLE_RECORDING else None

    # ゾーン定義
    boundary_x    = int(w * ZONE_BOUNDARY)  # 境界線X座標
    inside_zone   = (0, boundary_x)         # 左側：inside
    door_zone     = (boundary_x, w)         # 右側：door

    # ── [計測] フレーム全体の開始 ───────────────────────────
    t_frame_start = time.perf_counter()

    # ── [計測] YOLO 推論（Hailo） ──────────────────────────────
    t0 = time.perf_counter()
    boxes, scores = model.predict(frame)
    t1 = time.perf_counter()

    if len(boxes) > 0:
        detections = np.column_stack((boxes, scores))
    else:
        detections = np.empty((0, 5), dtype=np.float32)

    num_det = len(detections)
    detection_history.append(num_det)

    # ── Re-ID 適応制御 ──────────────────────────────────────
    tracker.with_reid = ENABLE_REID

    # ── [計測] トラッキング実行 (Re-ID含む) ─────────────────
    t2 = time.perf_counter()
    try:
        online_targets = tracker.update(detections, frame)
    except Exception as e:
        print(f"Tracker error: {e}")
        online_targets = []
    t3 = time.perf_counter()

    num_track = len(online_targets)
    track_count_history.append(num_track)

    # ── [計測] ゾーン判定+描画 開始 ─────────────────────────
    t4 = time.perf_counter()

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

        # オフライン再構成用の検出ログ（bbox・zone・trackごとに1行）
        log_detection(
            frame_count, now, track_id, x1, y1, x2, y2,
            current_zone, track_id in stable_tracks
        )

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
    # Inside：境界線の左側に大きく表示
    cv2.putText(frame, "Inside", (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 5)
    # Door：境界線の右側に大きく表示
    cv2.putText(frame, "Door", (boundary_x + 30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 5)

    # Re-ID 状態・FPS 表示
    reid_status = "ON" if (ENABLE_REID and tracker.with_reid) else "OFF"
    reid_color  = (0, 255, 0) if tracker.with_reid else (128, 128, 128)
    cv2.putText(frame, f"ReID: {reid_status}", (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, reid_color, 2)
    cv2.putText(frame, f"FPS: {current_fps:.1f}", (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    # ── [計測] ゾーン判定+描画 終了 ─────────────────────────
    t5 = time.perf_counter()

    # ── [計測] 処理時間を記録 ───────────────────────────────
    ms_det   = (t1 - t0) * 1000
    ms_trk   = (t3 - t2) * 1000
    ms_zone  = (t5 - t4) * 1000
    ms_total = (t5 - t_frame_start) * 1000

    timing_det.append(ms_det)
    timing_trk.append(ms_trk)
    timing_zone.append(ms_zone)
    timing_total.append(ms_total)

    # CSVに書き込み
    if timing_file:
        timing_file.write(
            f"{frame_count},{ms_det:.2f},{ms_trk:.2f},{ms_zone:.2f},"
            f"{ms_total:.2f},{num_det},{num_track},{current_fps:.1f}\n"
        )
        if frame_count % 30 == 0:
            timing_file.flush()

    # コンソールに定期出力（30フレームごと）
    if frame_count % 30 == 0:
        avg_det   = sum(timing_det) / len(timing_det)
        avg_trk   = sum(timing_trk) / len(timing_trk)
        avg_zone  = sum(timing_zone) / len(timing_zone)
        avg_total = sum(timing_total) / len(timing_total)
        print(f"[Frame {frame_count}] "
              f"検出: {avg_det:.1f}ms | "
              f"追跡+ReID: {avg_trk:.1f}ms | "
              f"判定: {avg_zone:.1f}ms | "
              f"合計: {avg_total:.1f}ms | "
              f"FPS: {current_fps:.1f}")
        print(f"  📊 乗車:{entry_count} 降車:{exit_count} 現在:{current_count}人")

    # ── 処理時間をフレームにオーバーレイ表示 ─────────────────
    y_timing = h - 80
    cv2.putText(frame, f"Det: {ms_det:.1f}ms", (10, y_timing),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
    cv2.putText(frame, f"Trk+ReID: {ms_trk:.1f}ms", (10, y_timing + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
    cv2.putText(frame, f"Zone: {ms_zone:.1f}ms", (10, y_timing + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
    cv2.putText(frame, f"Total: {ms_total:.1f}ms", (10, y_timing + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # 録画キューへ投入（描画前の生フレーム。非ブロッキング）
    if ENABLE_RECORDING:
        enqueue_recording_frame(raw_frame_for_recording, frame_count, now)


# ============================================================
# 15. 最終サマリー出力
# ============================================================
def write_summary():
    """終了時の最終統計を summary.txt に保存し、コンソールにも表示する"""
    end_time = datetime.datetime.now()
    duration = end_time - SESSION_START_TIME if SESSION_START_TIME else None

    lines = []
    lines.append("=" * 60)
    lines.append("セッションサマリー")
    lines.append("=" * 60)
    lines.append(f"セッションフォルダ : {SESSION_DIR}")
    if SESSION_START_TIME:
        lines.append(f"開始時刻           : {SESSION_START_TIME.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"終了時刻           : {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    if duration:
        lines.append(f"稼働時間           : {str(duration).split('.')[0]}")
    lines.append(f"総入室数           : {entry_count}人")
    lines.append(f"総退室数           : {exit_count}人")
    lines.append(f"最終在室数         : {current_count}人")
    lines.append(f"総処理フレーム数   : {frame_count}")
    if ENABLE_RECORDING:
        lines.append(f"録画ファイル       : {RECORDING_FILE_PATH}")
        lines.append(f"  書き込み {recording_frame_count} フレーム / 破棄 {recording_dropped} フレーム")
    if timing_total:
        lines.append(f"平均フレーム処理時間: {sum(timing_total)/len(timing_total):.1f} ms（直近{len(timing_total)}件）")
    lines.append("=" * 60)

    text = "\n".join(lines)
    print(text)

    if SUMMARY_LOG_PATH:
        try:
            with open(SUMMARY_LOG_PATH, "w", encoding="utf-8") as f:
                f.write(text + "\n")
            print(f"✓ サマリー保存: {SUMMARY_LOG_PATH}")
        except Exception as e:
            print(f"✖ サマリー保存エラー: {e}")


# ============================================================
# 16. メイン実行
# ============================================================
def main():
    print("=" * 60)
    print("入退室カウンター [計測版]")
    print("(YOLO11 + BoT-SORT on Raspberry Pi 5 + Hailo-8)")
    print("=" * 60)
    print(f"Re-ID機能: {'ON (常時)' if ENABLE_REID else 'OFF'}")
    print(f"解像度: {TARGET_RESOLUTION[0]}x{TARGET_RESOLUTION[1]}")
    print("=" * 60)

    # セッションフォルダ作成（録画・各種ログの保存先）
    init_session()

    # 処理時間ログ初期化
    init_timing_log()

    # 非同期録画・検出ログ初期化
    print(f"録画機能: {'ON（生映像＋検出ログのみ、描画なし）' if ENABLE_RECORDING else 'OFF'}")
    init_recording()
    init_detection_log()

    # カメラ起動
    picam2 = init_camera()

    # Hailo YOLO パイプラインを開く
    model.open()

    print("=" * 60)
    print(f"✓ システム起動完了（Web UIなし・コンソール出力のみ）")
    print(f"✓ セッションフォルダ: {SESSION_DIR}")
    print("=" * 60)
    print("操作方法:")
    print("  Ctrl+C : プログラム終了 (サマリー表示)")
    print("  ※ 30フレームごとに処理時間・入退室数をコンソールに出力します")
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

        # 処理時間の詳細サマリーを出力
        close_timing_log()
        # 録画・検出ログの終了処理
        close_recording()
        close_detection_log()
        # 最終統計サマリーを保存・表示
        write_summary()

    except Exception as e:
        print(f"✖ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        close_timing_log()
        close_recording()
        close_detection_log()
        write_summary()

    finally:
        model.close()
        picam2.stop()
        print("✓ クリーンアップ完了")


if __name__ == "__main__":
    main()
