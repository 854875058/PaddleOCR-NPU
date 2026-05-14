#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C0 探针：验证同一张 NPU 卡上能否并发跑两个独立进程的 PytorchPaddleOCR。

用法（在 PaddleOCR-NPU-main 项目根目录下执行）：
    python zhn_test/算子重启/probe_same_card_multi_instance.py [--device 1] [--rounds 3]

判定阈值：两进程并发耗时 < 单进程耗时 × 1.8 → 通过
        通过 = 单卡多实例方向可行 → 走 MultiProcessOCRPool 重构
        失败 = NPU device busy / 推理崩溃 → 退到 1 实例/卡
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np


def _child_run(device_id: int, n_iters: int, ready_q, done_q):
    """子进程入口：加载 OCR 实例，等同伴 ready 后并发跑 n 轮，回报耗时。"""
    try:
        from pytorch_paddle import PytorchPaddleOCR
        ocr = PytorchPaddleOCR(use_npu=True, npu_device_id=device_id, use_angle_cls=True)
        ready_q.put(("ready", os.getpid(), device_id))
        # 等所有进程都 ready 后再开始计时（同伴的 'go' 信号在主进程统一发出）
        signal = ready_q.get()
        if signal != ("go",):
            done_q.put(("err", os.getpid(), f"unexpected signal {signal}"))
            return

        dummy = np.full((512, 512, 3), 255, dtype=np.uint8)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            ocr.ocr(dummy, format_output=False)
        elapsed = time.perf_counter() - t0
        done_q.put(("ok", os.getpid(), elapsed))
    except Exception as exc:
        done_q.put(("err", os.getpid(), repr(exc)))


def _run_processes(n_workers: int, device_id: int, n_iters: int):
    """启动 n_workers 个子进程，等全部 ready 再统一发 go，返回每个进程耗时。"""
    ctx = mp.get_context("spawn")  # NPU 安全：避免 fork 后子进程 NPU 上下文损坏
    ready_qs = [ctx.Queue() for _ in range(n_workers)]
    done_qs = [ctx.Queue() for _ in range(n_workers)]
    procs = [
        ctx.Process(target=_child_run, args=(device_id, n_iters, ready_qs[i], done_qs[i]))
        for i in range(n_workers)
    ]
    for p in procs:
        p.start()

    # 等所有 worker ready
    for i in range(n_workers):
        msg = ready_qs[i].get(timeout=180)
        if msg[0] != "ready":
            raise RuntimeError(f"worker {i} ready failed: {msg}")
        print(f"  [worker-{i}] ready (pid={msg[1]}, device=npu:{msg[2]})")

    # 同步发 go
    for q in ready_qs:
        q.put(("go",))

    timings = []
    for i in range(n_workers):
        kind, pid, payload = done_qs[i].get(timeout=300)
        if kind != "ok":
            raise RuntimeError(f"worker {i} (pid={pid}) failed: {payload}")
        timings.append(payload)
        print(f"  [worker-{i}] elapsed={payload:.2f}s (pid={pid})")

    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.kill()
    return timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1, help="NPU device id")
    parser.add_argument("--iters", type=int, default=10, help="iterations per worker")
    args = parser.parse_args()

    print(f"=== C0 same-card multi-instance probe ===")
    print(f"device=npu:{args.device}, iters={args.iters}")
    print()

    print(">>> Phase 1: single instance baseline")
    [single_t] = _run_processes(n_workers=1, device_id=args.device, n_iters=args.iters)
    print(f"single_total={single_t:.2f}s, per_iter={single_t / args.iters:.3f}s")
    print()

    print(">>> Phase 2: 2 concurrent instances on the same card")
    timings = _run_processes(n_workers=2, device_id=args.device, n_iters=args.iters)
    parallel_total = max(timings)
    avg_each = sum(timings) / len(timings)
    print(f"parallel_max={parallel_total:.2f}s, avg_each={avg_each:.2f}s")
    print()

    ratio = parallel_total / single_t if single_t > 0 else float("inf")
    threshold = 1.8
    print(f"=== Verdict ===")
    print(f"ratio (parallel_max / single) = {ratio:.2f}, threshold < {threshold}")
    if ratio < threshold:
        print("PASS: same-card multi-instance is viable")
        sys.exit(0)
    else:
        print("FAIL: parallelism is poor; fall back to 1 instance per card")
        sys.exit(1)


if __name__ == "__main__":
    main()
