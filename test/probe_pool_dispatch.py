#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C1+C2 自检：直接实例化 MultiProcessOCRPool，跑 single + batch，验证 dispatch/demux 通路。

用法（在项目根目录下）:
    python test/probe_pool_dispatch.py [--device 1] [--instances 2]

不依赖 FastAPI 启动。如果 ElasticOCRPool 服务正在跑，请先停掉避免抢 NPU。
"""
import argparse
import base64
import sys
import time

import cv2
import numpy as np

# 让 import 找到项目根
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr_server import MultiProcessOCRPool


def _encode_dummy() -> str:
    img = np.full((512, 512, 3), 255, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError("encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--instances", type=int, default=2,
                        help="min_instances and max_instances both set to this")
    parser.add_argument("--per-card-max", type=int, default=2)
    args = parser.parse_args()

    print(f"=== C1+C2 self-test: MultiProcessOCRPool ===")
    print(f"device={args.device}, instances={args.instances}, per_card_max={args.per_card_max}")
    print()

    print("--- creating pool (this triggers spawn + warmup, ~30s) ---", flush=True)
    t0 = time.perf_counter()
    pool = MultiProcessOCRPool(
        npu_device_ids=[args.device],
        min_instances=args.instances,
        max_instances=args.instances,
        per_card_max=args.per_card_max,
        worker_init_timeout=300.0,
        det_model_path="./models/ptocr_v5_server_det.pth",
        rec_model_path="./models/ptocr_v5_server_rec.pth",
        cls_model_path="./models/ch_ptocr_mobile_v2.0_cls_infer.pth",
        rec_char_dict_path="./pytorchocr/utils/dict/ppocrv5_dict.txt",
        det_yaml_path="configs/det/PP-OCRv5/PP-OCRv5_server_det.yml",
        rec_yaml_path="configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml",
    )
    init_t = time.perf_counter() - t0
    print(f"pool ready in {init_t:.2f}s, device_info={pool.device_info}")
    print(f"stats: {pool.get_pool_stats()}", flush=True)

    try:
        print("\n--- single OCR ---", flush=True)
        b64 = _encode_dummy()
        t0 = time.perf_counter()
        result = pool.process_single_image(b64, format_output=True)
        single_t = time.perf_counter() - t0
        print(f"single OK={result.get('success')} text_count={result.get('text_count')} elapsed={single_t:.2f}s")

        print("\n--- batch OCR (4 images) ---", flush=True)
        images = [b64] * 4
        t0 = time.perf_counter()
        bresult = pool.process_batch_images(images, format_output=True, use_optimized=True)
        batch_t = time.perf_counter() - t0
        print(f"batch OK={bresult.get('success')} image_count={bresult.get('image_count')} elapsed={batch_t:.2f}s")

        print("\n--- final pool stats ---", flush=True)
        print(pool.get_pool_stats())

        verdict_pass = bool(result.get('success')) and bool(bresult.get('success'))
        print()
        if verdict_pass:
            print("PASS: dispatch/demux works end-to-end")
        else:
            print("FAIL: at least one call returned success=False")
            print(f"  single error: {result.get('error')}")
            print(f"  batch error:  {bresult.get('error')}")
        return 0 if verdict_pass else 1
    finally:
        print("\n--- shutting down pool ---", flush=True)
        pool.shutdown(wait_timeout=10.0)
        print("done.")


if __name__ == "__main__":
    sys.exit(main())
