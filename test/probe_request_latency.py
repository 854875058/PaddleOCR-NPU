#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断 mp.Queue 把 base64 jpeg 来回传 vs 服务端单次推理的耗时占比。
直接 POST /ocr/single 量端到端延迟，对比服务端日志里的 OCR 核心耗时。
"""
import argparse
import base64
import statistics
import time
import urllib.request
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:6663/ocr/single")
    parser.add_argument("--image", default="doc/imgs/11.jpg")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--mode", choices=["serial", "concurrent"], default="serial")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"image not found: {args.image}")
        sys.exit(1)

    with open(args.image, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    payload = json.dumps({"image": b64}).encode("utf-8")
    print(f"image: {args.image} ({len(payload)/1024:.1f} KB payload)")

    def _one():
        t0 = time.perf_counter()
        req = urllib.request.Request(args.url, data=payload,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read()
        return time.perf_counter() - t0, len(body)

    print(f"\n--- mode={args.mode} n={args.n} ---")
    if args.mode == "serial":
        timings = []
        for i in range(args.n):
            t, sz = _one()
            timings.append(t)
            print(f"  [{i+1}/{args.n}] {t:.3f}s (resp {sz/1024:.1f}KB)")
    else:
        from concurrent.futures import ThreadPoolExecutor
        timings = []
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(_one) for _ in range(args.n)]
            for f in futs:
                t, sz = f.result()
                timings.append(t)
        wall = time.perf_counter() - t0
        print(f"  wall={wall:.2f}s, throughput={args.n / wall:.2f} req/s")

    print(f"\n--- stats ---")
    print(f"  count: {len(timings)}")
    print(f"  mean:  {statistics.mean(timings):.3f}s")
    print(f"  p50:   {statistics.median(timings):.3f}s")
    print(f"  p95:   {sorted(timings)[int(len(timings) * 0.95) - 1]:.3f}s")
    print(f"  min:   {min(timings):.3f}s")
    print(f"  max:   {max(timings):.3f}s")


if __name__ == "__main__":
    main()
