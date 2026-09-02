#!/usr/bin/env python3
"""
テスト②：repvgg_a0_person_reid_2048 による Re-ID 特徴抽出テスト
Hailo-8 上で Re-ID モデルを動かし、人物クロップ画像から
2048次元の特徴ベクトルが正しく出力されるか確認します。

実行方法:
    python3 test_reid.py

確認ポイント:
    - 特徴ベクトルの shape が (2048,) であること
    - 同一人物のコサイン類似度が高いこと（0.8以上が理想）
    - 別人物のコサイン類似度が低いこと（0.5以下が理想）
"""

import numpy as np
import cv2
import time
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
HEF_PATH     = "/home/yaslab/models/repvgg_a0_person_reid_2048.hef"
REID_SIZE    = (128, 256)   # Re-IDモデルの入力サイズ (W, H)
CAMERA_SIZE  = (640, 480)   # カメラ解像度

# カメラの向きを調整
FLIP_HORIZONTAL = True
FLIP_VERTICAL   = True
# ────────────────────────────────────────────────────────────────────────


# ── カメラ初期化 ──────────────────────────────────────────────────────────
def init_camera():
    print("[1/4] カメラを初期化中...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": CAMERA_SIZE, "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)
    print("✓ カメラ初期化完了")
    return picam2


# ── Hailo Re-IDモデルロード ───────────────────────────────────────────────
def load_reid_model(hef_path: str):
    print(f"[2/4] Re-IDモデルをロード中: {hef_path}")
    hef    = HEF(hef_path)
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
    print("✓ Re-IDモデルロード完了")
    return target, hef, network_group, network_group_params, input_params, output_params


# ── Re-ID 前処理 ──────────────────────────────────────────────────────────
def preprocess_crop(crop: np.ndarray) -> np.ndarray:
    """
    人物クロップ画像を Re-ID モデル入力形式に変換
    入力: BGR画像（任意サイズ）
    出力: RGB画像 UINT8 (256, 128, 3)
    """
    # Re-IDモデルの入力サイズにリサイズ (W=128, H=256)
    resized = cv2.resize(crop, REID_SIZE)
    # BGR → RGB 変換
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.uint8)


# ── Re-ID 推論 ────────────────────────────────────────────────────────────
def extract_feature(pipeline, input_params, crop: np.ndarray) -> np.ndarray:
    """
    人物クロップ画像から特徴ベクトルを抽出
    戻り値: L2正規化済み特徴ベクトル shape=(2048,)
    """
    input_data  = preprocess_crop(crop)
    input_batch = {
        list(input_params.keys())[0]: np.expand_dims(input_data, axis=0)
    }
    raw_output = pipeline.infer(input_batch)

    # 出力を取得
    feature = None
    for key, val in raw_output.items():
        if isinstance(val, np.ndarray):
            feature = val.flatten()
        elif isinstance(val, list):
            feature = np.array(val).flatten()
        break

    if feature is None:
        raise ValueError("Re-ID モデルから出力を取得できませんでした")

    # L2 正規化（コサイン類似度のため）
    norm    = np.linalg.norm(feature)
    feature = feature / (norm + 1e-8)
    return feature


# ── コサイン類似度 ────────────────────────────────────────────────────────
def cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """L2正規化済みベクトル同士のコサイン類似度（-1〜1）"""
    return float(np.dot(feat1, feat2))


# ── 反転処理 ─────────────────────────────────────────────────────────────
def apply_flip(frame: np.ndarray) -> np.ndarray:
    if FLIP_HORIZONTAL and FLIP_VERTICAL:
        return cv2.flip(frame, -1)
    elif FLIP_HORIZONTAL:
        return cv2.flip(frame, 1)
    elif FLIP_VERTICAL:
        return cv2.flip(frame, 0)
    return frame


# ── フレーム中央から人物クロップを切り出す ────────────────────────────────
def crop_center_person(frame: np.ndarray, ratio_w=0.3, ratio_h=0.8) -> np.ndarray:
    """
    フレーム中央から人物サイズ相当の領域を切り出す
    （YOLOなしで Re-ID 単体テストをするための簡易クロップ）
    """
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    bw = int(w * ratio_w)
    bh = int(h * ratio_h)
    x1 = max(cx - bw // 2, 0)
    y1 = max(cy - bh // 2, 0)
    x2 = min(cx + bw // 2, w)
    y2 = min(cy + bh // 2, h)
    return frame[y1:y2, x1:x2].copy()


# ── メイン ───────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("テスト②：Re-ID 特徴抽出テスト")
    print("=" * 55)

    # カメラ・モデル初期化
    picam2 = init_camera()
    target, hef, network_group, ng_params, input_params, output_params = \
        load_reid_model(HEF_PATH)

    print("\n[3/4] 特徴ベクトル出力確認...")

    with InferVStreams(network_group, input_params, output_params) as pipeline:
        with network_group.activate(ng_params):

            # ── テスト① 出力フォーマット確認 ──────────────────────────
            print("\n▶ テスト①：出力フォーマット確認")
            frame = picam2.capture_array()
            frame = apply_flip(frame)

            # BGRに変換してからクロップ
            bgr   = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            crop  = crop_center_person(bgr)

            t_start = time.time()
            feature = extract_feature(pipeline, input_params, crop)
            elapsed = (time.time() - t_start) * 1000

            print(f"  特徴ベクトル shape : {feature.shape}")
            print(f"  特徴ベクトル dtype : {feature.dtype}")
            print(f"  L2ノルム           : {np.linalg.norm(feature):.4f}（1.0に近いほど正常）")
            print(f"  推論時間           : {elapsed:.1f} ms")

            if feature.shape == (2048,):
                print("  ✓ 出力フォーマット正常")
            else:
                print(f"  ⚠ 期待: (2048,)  実際: {feature.shape}")

            # ── テスト② 同一人物の類似度確認 ──────────────────────────
            print("\n▶ テスト②：同一人物のコサイン類似度確認")
            print("  カメラの前に立って静止してください...")
            time.sleep(2)

            features_same = []
            for i in range(5):
                frame   = picam2.capture_array()
                frame   = apply_flip(frame)
                bgr     = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                crop    = crop_center_person(bgr)
                feat    = extract_feature(pipeline, input_params, crop)
                features_same.append(feat)
                print(f"  サンプル {i+1}/5 取得完了")
                time.sleep(0.3)

            # 5サンプル間の類似度を計算
            sims = []
            for i in range(len(features_same)):
                for j in range(i + 1, len(features_same)):
                    sim = cosine_similarity(features_same[i], features_same[j])
                    sims.append(sim)

            avg_sim = np.mean(sims)
            print(f"\n  同一人物の平均コサイン類似度: {avg_sim:.4f}")
            if avg_sim >= 0.8:
                print("  ✓ 優秀（0.8以上）")
            elif avg_sim >= 0.6:
                print("  △ 許容範囲（0.6〜0.8）")
            else:
                print("  ⚠ 低い（0.6未満）：照明や角度を確認してください")

            # ── テスト③ 速度テスト ────────────────────────────────────
            print("\n▶ テスト③：推論速度テスト（30フレーム）")
            times = []
            for i in range(30):
                frame   = picam2.capture_array()
                frame   = apply_flip(frame)
                bgr     = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                crop    = crop_center_person(bgr)

                t = time.time()
                extract_feature(pipeline, input_params, crop)
                times.append((time.time() - t) * 1000)

            print(f"  平均推論時間 : {np.mean(times):.1f} ms")
            print(f"  最小推論時間 : {np.min(times):.1f} ms")
            print(f"  最大推論時間 : {np.max(times):.1f} ms")
            print(f"  推論FPS      : {1000/np.mean(times):.1f}")

    # ── 結果サマリー ──────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("テスト結果サマリー")
    print("=" * 55)
    print(f"  特徴ベクトルサイズ    : {feature.shape}")
    print(f"  同一人物コサイン類似度 : {avg_sim:.4f}")
    print(f"  平均推論時間          : {np.mean(times):.1f} ms  "
          f"（{1000/np.mean(times):.1f} FPS）")

    if feature.shape == (2048,) and avg_sim >= 0.6:
        print("\n✓ Re-ID モデル正常動作確認！")
        print("  次のステップ：Step 3 BoT-SORT のインストールへ進んでください")
    else:
        print("\n⚠ 問題が検出されました。上記の結果を確認してください。")

    picam2.stop()


if __name__ == "__main__":
    main()