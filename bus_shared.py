#!/usr/bin/env python3
"""
bus_shared.py
バス乗降客カウンターシステムの共有状態管理モジュール

検出プロセス（nbotcount.py / imxbotcount.py）と
Flask配信プロセス（flask_server.py）間で
フレームと統計情報を共有する。

仕組み:
    - フレーム: multiprocessing.shared_memory（バイナリ）
    - 統計   : /dev/shm 上のJSONファイル（atomic write）
"""

import os
import json
import numpy as np
from multiprocessing import shared_memory

# ============================================================
# 共有メモリ設定
# ============================================================
FRAME_SHM_NAME = "bus_frame"
STATS_FILE     = "/dev/shm/bus_stats.json"

# サポートする最大解像度（バッファサイズ計算用）
MAX_WIDTH  = 1920
MAX_HEIGHT = 1080
CHANNELS   = 3

# ヘッダ構造: width(4) + height(4) + channels(4) = 12 bytes
HEADER_SIZE = 12
FRAME_SIZE  = MAX_WIDTH * MAX_HEIGHT * CHANNELS
TOTAL_SIZE  = HEADER_SIZE + FRAME_SIZE


# ============================================================
# FrameBuffer: フレーム共有メモリ管理
# ============================================================
class FrameBuffer:
    """
    検出プロセス: FrameBuffer.create() で作成
    Flaskプロセス: FrameBuffer.attach() で接続
    """

    def __init__(self, shm, is_owner: bool):
        self.shm      = shm
        self.is_owner = is_owner

    @classmethod
    def create(cls):
        """検出プロセス側：共有メモリを新規作成"""
        # 既存のものがあれば削除
        try:
            existing = shared_memory.SharedMemory(name=FRAME_SHM_NAME)
            existing.close()
            existing.unlink()
        except FileNotFoundError:
            pass

        shm = shared_memory.SharedMemory(
            name=FRAME_SHM_NAME,
            create=True,
            size=TOTAL_SIZE
        )
        print(f"✓ FrameBuffer 作成: {TOTAL_SIZE/1024/1024:.1f} MB")
        return cls(shm, is_owner=True)

    @classmethod
    def attach(cls):
        """Flaskプロセス側：既存の共有メモリに接続"""
        shm = shared_memory.SharedMemory(name=FRAME_SHM_NAME)
        return cls(shm, is_owner=False)

    def write(self, frame: np.ndarray):
        """フレームを書き込む（BGR形式のnumpy配列）"""
        h, w, c = frame.shape

        if w > MAX_WIDTH or h > MAX_HEIGHT:
            raise ValueError(f"フレームサイズが大きすぎます: {w}x{h}")

        # ヘッダ書き込み（int32 × 3）
        header = np.array([w, h, c], dtype=np.int32)
        self.shm.buf[0:HEADER_SIZE] = header.tobytes()

        # フレームデータ書き込み
        frame_bytes = frame.tobytes()
        self.shm.buf[HEADER_SIZE:HEADER_SIZE + len(frame_bytes)] = frame_bytes

    def read(self):
        """フレームを読み込む（BGR形式のnumpy配列を返す）"""
        # ヘッダ読み込み
        header = np.frombuffer(
            bytes(self.shm.buf[0:HEADER_SIZE]),
            dtype=np.int32
        )
        w, h, c = int(header[0]), int(header[1]), int(header[2])

        if w == 0 or h == 0:
            return None

        # フレームデータ読み込み
        frame_size = w * h * c
        frame_bytes = bytes(self.shm.buf[HEADER_SIZE:HEADER_SIZE + frame_size])
        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(h, w, c)
        return frame.copy()

    def close(self):
        """共有メモリを閉じる"""
        try:
            self.shm.close()
        except:
            pass
        # 作成者のみが unlink できる
        if self.is_owner:
            try:
                self.shm.unlink()
                print("✓ FrameBuffer 解放完了")
            except:
                pass


# ============================================================
# StatsFile: 統計情報のJSON共有
# ============================================================
class StatsFile:
    """統計情報を /dev/shm 上のJSONで共有"""

    DEFAULT_STATS = {
        "entry_count":   0,
        "exit_count":    0,
        "current_count": 0,
        "photo_count":   0,
        "cam_fps":       0.0,
        "imx_fps":       0.0,
        "person_count":  0,
        "system_type":   "unknown",   # "hailo" or "imx500"
    }

    @classmethod
    def write(cls, stats: dict):
        """統計情報を書き込み（atomic write）"""
        merged = {**cls.DEFAULT_STATS, **stats}
        tmp = STATS_FILE + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(merged, f)
            os.rename(tmp, STATS_FILE)   # atomic
        except Exception as e:
            print(f"Stats write error: {e}")

    @classmethod
    def read(cls) -> dict:
        """統計情報を読み込み（存在しなければデフォルト）"""
        try:
            with open(STATS_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return cls.DEFAULT_STATS.copy()

    @classmethod
    def cleanup(cls):
        """終了時にファイル削除"""
        try:
            os.remove(STATS_FILE)
        except:
            pass


# ============================================================
# テスト
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("bus_shared.py 動作テスト")
    print("=" * 50)

    # FrameBuffer テスト
    fb = FrameBuffer.create()
    test_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    fb.write(test_frame)
    read_frame = fb.read()
    print(f"フレームサイズ: {read_frame.shape}")
    print(f"データ一致: {np.array_equal(test_frame, read_frame)}")
    fb.close()

    # StatsFile テスト
    StatsFile.write({"entry_count": 5, "exit_count": 3, "system_type": "test"})
    stats = StatsFile.read()
    print(f"統計情報: {stats}")
    StatsFile.cleanup()
    print("テスト完了")
