#!/usr/bin/env python3
"""
hailo_reid.py
FastReIDInterface の代替として Hailo-8 上で Re-ID 特徴抽出を行うクラス。

BoT-SORT の bot_sort.py から以下のように呼ばれる:
    self.encoder = HailoReIDEncoder(hef_path)
    features = self.encoder.inference(img, dets)

インターフェースは FastReIDInterface と同一に保つ。
"""

import numpy as np
import cv2
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


class HailoReIDEncoder:
    """
    Hailo-8 上で repvgg_a0_person_reid_2048.hef を使い
    人物クロップ画像から特徴ベクトルを抽出するクラス。

    FastReIDInterface と同じ inference(img, dets) インターフェースを持つ。
    """

    # Re-ID モデルの入力サイズ (W, H)
    REID_INPUT_W = 128
    REID_INPUT_H = 256

    def __init__(self, hef_path: str, vdevice=None):
        """
        Args:
            hef_path: repvgg_a0_person_reid_2048.hef のパス
            vdevice : 共有VDevice（Noneの場合は新規作成）
        """
        print(f"[HailoReID] モデルをロード中: {hef_path}")
        self.hef    = HEF(hef_path)
        self.target = vdevice if vdevice is not None else VDevice()

        configure_params = ConfigureParams.create_from_hef(
            hef=self.hef,
            interface=HailoStreamInterface.PCIe
        )
        self.network_group        = self.target.configure(self.hef, configure_params)[0]
        self.network_group_params = self.network_group.create_params()

        self.input_params = InputVStreamParams.make(
            self.network_group,
            quantized=False,
            format_type=FormatType.UINT8
        )
        self.output_params = OutputVStreamParams.make(
            self.network_group,
            quantized=False,
            format_type=FormatType.FLOAT32
        )

        # 入力キー名をキャッシュ
        self.input_key = list(self.input_params.keys())[0]

        print("[HailoReID] ✓ モデルロード完了")

    def _crop_person(self, img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        フレームから人物クロップを切り出す
        Args:
            img  : BGR フレーム (H, W, 3)
            bbox : [x1, y1, x2, y2] xyxy 形式
        Returns:
            クロップ画像 (REID_INPUT_H, REID_INPUT_W, 3) RGB
        """
        h, w = img.shape[:2]
        x1 = max(int(bbox[0]), 0)
        y1 = max(int(bbox[1]), 0)
        x2 = min(int(bbox[2]), w)
        y2 = min(int(bbox[3]), h)

        # クロップ領域が有効か確認
        if x2 <= x1 or y2 <= y1:
            # 無効な場合はゼロ画像を返す
            return np.zeros((self.REID_INPUT_H, self.REID_INPUT_W, 3), dtype=np.uint8)

        crop = img[y1:y2, x1:x2]

        # Re-ID モデルの入力サイズにリサイズ
        resized = cv2.resize(crop, (self.REID_INPUT_W, self.REID_INPUT_H))

        # BGR → RGB 変換
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.uint8)

    def _extract_features_batch(self, crops: list) -> np.ndarray:
        """
        人物クロップ画像のリストから特徴ベクトルをバッチ推論で抽出
        Args:
            crops: クロップ画像のリスト（各 RGB UINT8）
        Returns:
            特徴ベクトル配列 shape=(N, 2048) L2正規化済み
        """
        if not crops:
            return np.empty((0, 2048), dtype=np.float32)

        # バッチ作成
        batch = np.stack(crops, axis=0)  # (N, H, W, 3)

        features_list = []

        with InferVStreams(
            self.network_group,
            self.input_params,
            self.output_params
        ) as pipeline:
            for crop in batch:
                input_batch = {
                    self.input_key: np.expand_dims(crop, axis=0)
                }
                raw_output = pipeline.infer(input_batch)

                # 出力を取得
                feature = None
                for _, val in raw_output.items():
                    if isinstance(val, np.ndarray):
                        feature = val.flatten()
                    elif isinstance(val, list):
                        feature = np.array(val).flatten()
                    break

                if feature is None:
                    feature = np.zeros(2048, dtype=np.float32)

                # L2 正規化
                norm    = np.linalg.norm(feature)
                feature = feature / (norm + 1e-8)
                features_list.append(feature)

        return np.stack(features_list, axis=0)  # (N, 2048)

    def inference(self, img: np.ndarray, det: np.ndarray) -> np.ndarray:
        """
        FastReIDInterface.inference() と同じインターフェース

        Args:
            img : BGR フレーム全体 (H, W, 3)
            det : 検出ボックス配列 shape=(N, 4) xyxy 形式
                  ※ BoT-SORT から渡される形式

        Returns:
            特徴ベクトル配列 shape=(N, 2048) L2正規化済み
        """
        if det is None or len(det) == 0:
            return np.empty((0, 2048), dtype=np.float32)

        # 各検出ボックスから人物クロップを切り出す
        crops = [self._crop_person(img, bbox) for bbox in det]

        # Hailo で特徴抽出
        features = self._extract_features_batch(crops)

        return features