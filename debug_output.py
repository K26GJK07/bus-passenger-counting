#!/usr/bin/env python3
"""Hailo NMS出力フォーマットの確認"""
import numpy as np
from picamera2 import Picamera2
from hailo_platform import (
    HEF, VDevice, HailoStreamInterface, InferVStreams,
    ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType,
)
import cv2, time

HEF_PATH = "/home/yaslab/models/yolov11s.hef"

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    controls={"FrameDurationLimits": (33333, 33333)}
)
picam2.configure(config)
picam2.start()
time.sleep(1)

hef = HEF(HEF_PATH)
target = VDevice()
configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
network_group = target.configure(hef, configure_params)[0]
ng_params = network_group.create_params()
input_params  = InputVStreamParams.make(network_group, quantized=False, format_type=FormatType.UINT8)
output_params = OutputVStreamParams.make(network_group, quantized=False, format_type=FormatType.FLOAT32)

with InferVStreams(network_group, input_params, output_params) as pipeline:
    with network_group.activate(ng_params):
        frame = picam2.capture_array()
        resized = cv2.resize(frame, (640, 640)).astype(np.uint8)
        input_batch = {list(input_params.keys())[0]: np.expand_dims(resized, axis=0)}
        raw_output = pipeline.infer(input_batch)

        print("\n=== 出力フォーマット確認 ===")
        for key, val in raw_output.items():
            print(f"\nキー: {key}")
            print(f"  type: {type(val)}")
            if isinstance(val, np.ndarray):
                print(f"  shape: {val.shape}")
                print(f"  dtype: {val.dtype}")
            elif isinstance(val, list):
                print(f"  list len: {len(val)}")
                for i, item in enumerate(val[:3]):  # 最初の3クラスのみ表示
                    print(f"  [class {i}] type={type(item)}", end="")
                    if item is not None and hasattr(item, '__len__') and len(item) > 0:
                        print(f" len={len(item)}", end="")
                        first = item[0] if isinstance(item, list) else item
                        print(f" first_item={first}")
                    else:
                        print(" (empty)")

picam2.stop()