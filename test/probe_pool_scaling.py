#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C3 自检：验证 monitor 线程能正确触发扩缩容。

用法（在项目根目录下）:
    python test/probe_pool_scaling.py [--devices 1] [--min 1] [--max 4] [--per-card-max 4]

测试流程:
  1. 用 min=1 max=4 起池
  2. 并发提交 8 个 single 请求，观察 monitor 是否扩到 max
  3. 等 idle_timeout（这里调小到 15 秒）+ scale_cooldown 后看是否缩回 min

不依赖 FastAPI 启动；如果 ocr_service 在跑请先停。
"""
import argparse
import base64
import os
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ocr_server import MultiProcessOCRPool


def _encode_dummy() -> str:
    img = np.full((512, 512, 3), 255, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError("encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _print_stats(pool, label):
    s = pool.get_pool_stats()
    print(f"[{label}] ready={s['ready_instance_count']} busy={s['busy_instance_count']} "
          f"idle={s['idle_instance_count']} pending={s['pending_instance_count']} "
          f"per_device={s['instances_per_device']} qsize={s['task_queue_size']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=str, default="1",
                        help="comma-separated NPU device ids, e.g. '1' or '1,2'")
    parser.add_argument("--min", type=int, default=1)
    parser.add_argument("--max", type=int, default=4)
    parser.add_argument("--per-card-max", type=int, default=4)
    parser.add_argument("--idle-timeout", type=int, default=15,
                        help="缩容等待秒数（测试用，调小便于观察）")
    parser.add_argument("--scale-cooldown", type=int, default=3)
    parser.add_argument("--monitor-interval", type=float, default=2.0)
    parser.add_argument("--n-requests", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    devices = [int(x) for x in args.devices.split(",") if x.strip()]
    print(f"=== C3 self-test: scaling ===")
    print(f"devices={devices}, min={args.min}, max={args.max}, per_card_max={args.per_card_max}")
    print(f"idle_timeout={args.idle_timeout}s, scale_cooldown={args.scale_cooldown}s, "
          f"monitor_interval={args.monitor_interval}s")
    print()

    print("--- creating pool ---", flush=True)
    t0 = time.perf_counter()
    pool = MultiProcessOCRPool(
        npu_device_ids=devices,
        min_instances=args.min,
        max_instances=args.max,
        per_card_max=args.per_card_max,
        idle_timeout=args.idle_timeout,
        scale_cooldown=args.scale_cooldown,
        monitor_interval=args.monitor_interval,
        worker_init_timeout=300.0,
        det_model_path="./models/ptocr_v5_server_det.pth",
        rec_model_path="./models/ptocr_v5_server_rec.pth",
        cls_model_path="./models/ch_ptocr_mobile_v2.0_cls_infer.pth",
        rec_char_dict_path="./pytorchocr/utils/dict/ppocrv5_dict.txt",
        det_yaml_path="configs/det/PP-OCRv5/PP-OCRv5_server_det.yml",
        rec_yaml_path="configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml",
    )
    print(f"pool ready in {time.perf_counter() - t0:.2f}s", flush=True)
    _print_stats(pool, "init")

    try:
        # === Phase A: 制造负载，看 monitor 是否扩容 ===
        print(f"\n--- phase A: firing {args.n_requests} concurrent requests ---", flush=True)
        b64 = _encode_dummy()

        def _worker():
            r = pool.process_single_image(b64, format_output=True)
            if not r.get('success'):
                print(f"  request failed: {r.get('error')}", flush=True)

        # 边发请求边打印 stats，看扩容动态
        threads = []
        t0 = time.perf_counter()
        for i in range(args.n_requests):
            t = threading.Thread(target=_worker, name=f"caller-{i}", daemon=True)
            t.start()
            threads.append(t)

        # 每秒采一次 stats，最多 60 秒
        deadline = time.time() + 60
        while time.time() < deadline:
            if all(not t.is_alive() for t in threads):
                break
            _print_stats(pool, f"+{time.perf_counter() - t0:5.1f}s")
            time.sleep(1.0)

        for t in threads:
            t.join(timeout=120)
        elapsed = time.perf_counter() - t0
        print(f"\nphase A complete in {elapsed:.2f}s", flush=True)
        _print_stats(pool, "post-A")

        # === Phase B: 静默等待，看是否缩回 min ===
        print(f"\n--- phase B: idle wait for scale-down "
              f"(idle_timeout={args.idle_timeout}s + grace) ---", flush=True)
        deadline = time.time() + args.idle_timeout + 30
        while time.time() < deadline:
            _print_stats(pool, f"idle+{int(deadline - time.time())}s")
            stats = pool.get_pool_stats()
            if stats['ready_instance_count'] <= args.min:
                print(f"reached min_instances={args.min}, stopping wait", flush=True)
                break
            time.sleep(2.0)
        _print_stats(pool, "post-B")

        # === Verdict ===
        post_a_stats = pool.get_pool_stats()  # may already be back at min
        # 我们主要看 phase A 期间 pending 是否 > 0、ready 最终 > min
        print("\n=== Verdict ===")
        print("  inspect log above:")
        print("  - phase A: should see 'pending' > 0 and 'ready' growing toward max")
        print("  - phase B: should see ready dropping to min")
        return 0
    finally:
        print("\n--- shutting down pool ---", flush=True)
        pool.shutdown(wait_timeout=15.0)
        print("done.")


if __name__ == "__main__":
    sys.exit(main())
