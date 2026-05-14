#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C0 探针 v2：诊断 worker-1 在哪一步死掉。

v1 看到 worker-0 ready 后 worker-1 卡死，无法判断是 mp.Queue IPC 问题、
PaddleOCR init 死锁，还是 NPU 资源争用。v2 在每一步打 marker，并发出
两次 ready 信号（init_done 和 first_inference_done），让父进程精确定位。

用法：
    python test/probe_same_card_multi_instance_v2.py [--device 1] [--iters 10]
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np


def _emit(prefix, *parts):
    line = f"[{prefix} pid={os.getpid()}] " + " ".join(str(p) for p in parts)
    print(line, flush=True)


def _child_run(worker_idx: int, device_id: int, n_iters: int, ev_q, done_q):
    """每一步显式回报，方便父进程定位 worker 卡在哪一步。"""
    try:
        _emit(f"w{worker_idx}", "M0_started")
        ev_q.put(("M0_started", worker_idx, os.getpid()))

        _emit(f"w{worker_idx}", "M1_before_import")
        ev_q.put(("M1_before_import", worker_idx, os.getpid()))
        from pytorch_paddle import PytorchPaddleOCR

        _emit(f"w{worker_idx}", "M2_before_init")
        ev_q.put(("M2_before_init", worker_idx, os.getpid()))
        ocr = PytorchPaddleOCR(use_npu=True, npu_device_id=device_id, use_angle_cls=True)

        _emit(f"w{worker_idx}", "M3_init_done")
        ev_q.put(("M3_init_done", worker_idx, os.getpid()))

        # warmup（首次推理）
        dummy = np.full((512, 512, 3), 255, dtype=np.uint8)
        _emit(f"w{worker_idx}", "M4_before_warmup")
        ev_q.put(("M4_before_warmup", worker_idx, os.getpid()))
        t0 = time.perf_counter()
        ocr.ocr(dummy, format_output=False)
        warmup_t = time.perf_counter() - t0
        _emit(f"w{worker_idx}", f"M5_warmup_done {warmup_t:.2f}s")
        ev_q.put(("M5_warmup_done", worker_idx, os.getpid(), warmup_t))

        # 等父进程 go
        _emit(f"w{worker_idx}", "M6_waiting_go")
        ev_q.put(("M6_waiting_go", worker_idx, os.getpid()))
        signal = ev_q.get()
        if signal != ("go",):
            done_q.put(("err", worker_idx, os.getpid(), f"unexpected signal {signal}"))
            return

        # 稳态 n_iters 次
        _emit(f"w{worker_idx}", "M7_steady_start")
        t0 = time.perf_counter()
        for _ in range(n_iters):
            ocr.ocr(dummy, format_output=False)
        steady_t = time.perf_counter() - t0
        _emit(f"w{worker_idx}", f"M8_steady_done {steady_t:.2f}s")
        done_q.put(("ok", worker_idx, os.getpid(), warmup_t, steady_t))
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        _emit(f"w{worker_idx}", f"ERROR {exc!r}")
        done_q.put(("err", worker_idx, os.getpid(), repr(exc), tb))


def _drain_events(ev_qs, expected_marker: str, timeout_per_q: float):
    """等待所有 ev_qs 都报到 expected_marker，超时则汇总未达成者并 fail-fast。"""
    states = ["?"] * len(ev_qs)
    deadline_each = time.time() + timeout_per_q
    for i, q in enumerate(ev_qs):
        remaining = deadline_each - time.time()
        if remaining <= 0:
            states[i] = "TIMEOUT"
            continue
        # 取若干事件，直到看到 expected 或超时
        local_deadline = time.time() + remaining
        while True:
            now = time.time()
            if now >= local_deadline:
                break
            try:
                msg = q.get(timeout=min(local_deadline - now, 5.0))
            except Exception:
                continue
            states[i] = msg[0]
            if msg[0] == expected_marker:
                break
    return states


def _run_processes(n_workers: int, device_id: int, n_iters: int):
    ctx = mp.get_context("spawn")
    ev_qs = [ctx.Queue() for _ in range(n_workers)]
    done_qs = [ctx.Queue() for _ in range(n_workers)]
    procs = [
        ctx.Process(
            target=_child_run,
            args=(i, device_id, n_iters, ev_qs[i], done_qs[i]),
        )
        for i in range(n_workers)
    ]
    print(f"--- spawning {n_workers} worker(s) on npu:{device_id} ---", flush=True)
    for p in procs:
        p.start()

    # 关键：分阶段等到 M5_warmup_done。如果某 worker 卡在前面的某个 marker，
    # 我们就能精确知道是 import / init / warmup 哪一步死掉
    print(f"--- waiting for all workers to reach M5_warmup_done (per-worker timeout 240s) ---", flush=True)
    states = _drain_events(ev_qs, "M5_warmup_done", timeout_per_q=240.0)
    for i, s in enumerate(states):
        alive = procs[i].is_alive()
        print(f"  worker-{i} (pid={procs[i].pid}, alive={alive}) last_marker={s}", flush=True)

    if any(s != "M5_warmup_done" for s in states):
        print("\n!!! at least one worker did not reach M5_warmup_done. ABORTING.", flush=True)
        for p in procs:
            if p.is_alive():
                p.kill()
        sys.exit(2)

    print("--- all workers warmup done. sending go for steady-state inference ---", flush=True)
    for q in ev_qs:
        q.put(("go",))

    timings = []
    for i in range(n_workers):
        msg = done_qs[i].get(timeout=300)
        if msg[0] != "ok":
            print(f"!!! worker-{i} failed: {msg}", flush=True)
            sys.exit(2)
        warmup_t = msg[3]
        steady_t = msg[4]
        print(f"  worker-{i}: warmup={warmup_t:.2f}s, steady_total={steady_t:.2f}s ({steady_t / n_iters:.3f}s/iter)", flush=True)
        timings.append(steady_t)

    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.kill()
    return timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    print(f"=== C0 same-card multi-instance probe v2 (with markers) ===")
    print(f"device=npu:{args.device}, iters={args.iters}")
    print()

    print(">>> Phase 1: single instance baseline")
    [single_t] = _run_processes(n_workers=1, device_id=args.device, n_iters=args.iters)
    print(f"Phase 1 single steady_total={single_t:.2f}s\n")

    print(">>> Phase 2: 2 concurrent instances on the same card")
    timings = _run_processes(n_workers=2, device_id=args.device, n_iters=args.iters)
    parallel_max = max(timings)
    avg_each = sum(timings) / len(timings)
    print(f"Phase 2 parallel_max={parallel_max:.2f}s, avg_each={avg_each:.2f}s\n")

    ratio = parallel_max / single_t if single_t > 0 else float("inf")
    print(f"=== Verdict ===")
    print(f"ratio (parallel_max / single) = {ratio:.2f}, threshold < 1.8")
    if ratio < 1.8:
        print("PASS: same-card multi-instance is viable")
        sys.exit(0)
    else:
        print("DEGRADED: parallelism is poor; consider 1-instance-per-card")
        sys.exit(1)


if __name__ == "__main__":
    main()
