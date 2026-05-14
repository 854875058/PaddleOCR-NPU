#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR推理服务 - 基于FastAPI
支持单图推理、多图推理、并发处理
"""

import os
import sys
import time
import traceback
import threading
import multiprocessing as mp
import queue as _queue
import subprocess
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Union
import base64
import io
import uuid

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np
from PIL import Image
import cv2

# 导入OCR模块
from pytorch_paddle import PytorchPaddleOCR, create_ocr


class _AccessLogProbeFilter(logging.Filter):
    """过滤 uvicorn access 日志中的 /health 和 /info 探针请求，避免淹没真错误。"""

    _PATHS = ('/health', '/info')

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return not any(f' {p}' in msg or f' {p}?' in msg for p in self._PATHS)


class OCRRequest(BaseModel):
    """OCR请求模型"""
    image: str = Field(..., description="Base64编码的图像数据")


class BatchOCRRequest(BaseModel):
    """批量OCR请求模型"""
    images: List[str] = Field(..., description="Base64编码的图像数据列表")
    use_optimized: bool = Field(True, description="是否使用优化的批量处理")


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str = Field(..., description="服务状态")
    timestamp: float = Field(..., description="时间戳")
    device_info: str = Field(..., description="设备信息")
    model_loaded: bool = Field(..., description="模型是否已加载")


class InfoResponse(BaseModel):
    """服务信息响应模型"""
    service_name: str = Field(..., description="服务名称")
    version: str = Field(..., description="版本号")
    device_info: str = Field(..., description="设备信息")
    supported_formats: List[str] = Field(..., description="支持的图像格式")
    max_image_size: str = Field(..., description="最大图像尺寸")


class OCRServer:
    """OCR服务类"""
    
    def __init__(
        self,
        use_npu: bool = True,
        **ocr_kwargs
    ):
        """
        初始化OCR服务
        
        Args:
            use_npu: 是否使用NPU
            **ocr_kwargs: OCR初始化参数
        """
        self.ocr_kwargs = {
            'use_npu': use_npu,
            **ocr_kwargs
        }
        
        # 初始化OCR实例
        self.ocr_instance = None
        self.device_info = "unknown"
        self._init_ocr()
        
        # 统计信息
        self.request_count = 0
        self.error_count = 0
        self.lock = threading.Lock()
        
    def _init_ocr(self):
        """初始化OCR实例"""
        try:
            print("🚀 正在初始化OCR实例...")
            self.ocr_instance = create_ocr(**self.ocr_kwargs)
            
            # 获取设备信息
            if hasattr(self.ocr_instance.text_system.text_detector, 'device_type'):
                self.device_info = self.ocr_instance.text_system.text_detector.device_type
            elif self.ocr_kwargs.get('use_npu'):
                self.device_info = f"NPU-{self.ocr_kwargs.get('npu_device_id', 0)}"
            else:
                self.device_info = "CPU"
                
            print(f"OCR实例初始化成功 - 设备: {self.device_info}")
            
        except Exception as e:
            print(f"OCR实例初始化失败: {e}")
            raise e
    

    
    def _decode_base64_image(self, image_base64: str) -> np.ndarray:
        """
        解码Base64图像
        
        Args:
            image_base64: Base64编码的图像
            
        Returns:
            np.ndarray: 图像数组
        """
        try:
            # 移除data URL前缀（如果存在）
            if ',' in image_base64:
                image_base64 = image_base64.split(',')[1]
            
            # 解码Base64
            image_data = base64.b64decode(image_base64)
            
            # 转换为PIL图像
            image_pil = Image.open(io.BytesIO(image_data))
            
            # 转换为numpy数组（RGB格式）
            image_array = np.array(image_pil)
            
            # 如果是RGBA，转换为RGB
            if len(image_array.shape) == 3 and image_array.shape[2] == 4:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            
            return image_array
            
        except Exception as e:
            raise ValueError(f"无法解码图像数据: {e}")
    
    def _validate_image(self, image: np.ndarray) -> bool:
        """
        验证图像是否有效
        
        Args:
            image: 图像数组
            
        Returns:
            bool: 是否有效
        """
        if image is None or image.size == 0:
            return False
            
        # 检查图像尺寸
        if len(image.shape) < 2:
            return False
            
        height, width = image.shape[:2]
        
        # 图像尺寸限制
        if width < 10 or height < 10:
            return False
            
        if width > 10000 or height > 10000:
            return False
            
        return True
    
    def process_single_image(
        self,
        image_base64: str,
        format_output: bool = True,
        slice_params: Optional[Dict] = None
    ) -> Dict:
        """
        处理单张图像
        
        Args:
            image_base64: Base64编码的图像
            format_output: 是否返回格式化结果
            slice_params: 切片参数
            
        Returns:
            Dict: 处理结果
        """
        start_time = time.time()
        
        try:
            # 解码图像
            image = self._decode_base64_image(image_base64)
            
            # 验证图像
            if not self._validate_image(image):
                raise ValueError("图像无效或尺寸不符合要求")
            
            # 直接执行OCR
            results = self.ocr_instance.ocr(
                image,
                slice_params=slice_params,
                format_output=format_output
            )
            
            processing_time = time.time() - start_time
            
            if format_output:
                text_count = len(results.get('raw_results', []))
            else:
                text_count = len(results)
            
            return {
                'success': True,
                'results': results,
                'processing_time': processing_time,
                'text_count': text_count,
                'format_output': format_output
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"处理失败: {str(e)}"
            print(f"单图OCR失败: {error_msg}")
            
            return {
                'success': False,
                'error': error_msg,
                'processing_time': processing_time,
                'text_count': 0,
                'format_output': format_output
            }
    
    def process_batch_images(
        self,
        images: List[str],
        format_output: bool = True,
        use_optimized: bool = True
    ) -> Dict:
        """
        批量处理图像
        
        Args:
            images: Base64编码的图像列表
            format_output: 是否返回格式化结果
            use_optimized: 是否使用优化方法
            
        Returns:
            Dict: 处理结果
        """
        start_time = time.time()
        
        try:
            # 解码所有图像
            decoded_images = []
            for i, image_base64 in enumerate(images):
                try:
                    image = self._decode_base64_image(image_base64)
                    if self._validate_image(image):
                        decoded_images.append(image)
                    else:
                        print(f"图像 {i+1} 无效，将跳过")
                        decoded_images.append(None)
                except Exception as e:
                    print(f"图像 {i+1} 解码失败: {e}")
                    decoded_images.append(None)
            
            if not any(img is not None for img in decoded_images):
                raise ValueError("没有有效的图像数据")
            
            # 直接执行批量OCR
            if use_optimized:
                results = self.ocr_instance.batch_ocr_optimized(
                    decoded_images,
                    show_progress=True,
                    format_output=format_output
                )
                method_used = "optimized"
            else:
                results = self.ocr_instance.batch_ocr(
                    decoded_images,
                    show_progress=True,
                    format_output=format_output
                )
                method_used = "traditional"
            
            processing_time = time.time() - start_time
            
            if format_output:
                total_text_count = sum(len(img_result.get('raw_results', [])) for img_result in results)
            else:
                total_text_count = sum(len(img_results) for img_results in results)
            
            return {
                'success': True,
                'results': results,
                'processing_time': processing_time,
                'image_count': len(images),
                'total_text_count': total_text_count,
                'method_used': method_used,
                'format_output': format_output
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"批量处理失败: {str(e)}"
            print(f"批量OCR失败: {error_msg}")
            
            return {
                'success': False,
                'error': error_msg,
                'processing_time': processing_time,
                'image_count': len(images),
                'total_text_count': 0,
                'method_used': 'none',
                'format_output': format_output
            }



class ElasticOCRPool:
    """弹性 OCR 实例池：默认单实例，优先单卡多实例，必要时再扩到多卡。"""

    def __init__(
        self,
        npu_device_ids: List[int],
        min_instances: int = 1,
        max_instances: int = 3,
        idle_timeout: int = 120,
        scale_cooldown: int = 15,
        batch_acquire_wait: float = 8.0,
        instance_hbm_mb: int = 20000,
        hbm_safety_margin_mb: int = 4096,
        **ocr_kwargs
    ):
        self.npu_device_ids = list(dict.fromkeys(npu_device_ids or [ocr_kwargs.get('npu_device_id', 0)]))
        self.min_instances = max(1, min_instances)
        self.max_instances = max(self.min_instances, max_instances)
        self.idle_timeout = max(10, idle_timeout)
        self.scale_cooldown = max(0, scale_cooldown)
        self.batch_acquire_wait = max(0.0, batch_acquire_wait)
        self.instance_hbm_mb = max(1024, instance_hbm_mb)
        self.hbm_safety_margin_mb = max(0, hbm_safety_margin_mb)
        self.ocr_kwargs = dict(ocr_kwargs)
        self.instances = []
        self.pending_instances = 0
        self.instance_seq = 0
        self.last_scale_up_time = 0.0
        self.request_count = 0
        self.error_count = 0
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._shutdown = False

        for _ in range(self.min_instances):
            self._create_instance_sync(self._pick_scale_device())

        self._reaper_thread = threading.Thread(
            target=self._idle_reaper_loop,
            name="ocr-idle-reaper",
            daemon=True
        )
        self._reaper_thread.start()

    @property
    def device_info(self) -> str:
        with self._lock:
            ready_devices = [f"NPU-{item['device_id']}" for item in self.instances if item['ready']]
        return ",".join(ready_devices) if ready_devices else "unknown"

    def _query_npu_hbm_status(self) -> Dict[int, Dict[str, int]]:
        """读取 npu-smi info，返回每张卡的 HBM 使用情况。"""
        try:
            completed = subprocess.run(
                ["npu-smi", "info"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False
            )
            output = completed.stdout or ""
            lines = output.splitlines()
            status = {}

            for idx, line in enumerate(lines):
                stripped = line.strip()
                if not stripped.startswith("|"):
                    continue
                match = re.match(r"^\|\s*(\d+)\s+", stripped)
                if not match:
                    continue
                device_id = int(match.group(1))
                if device_id not in self.npu_device_ids:
                    continue
                if idx + 1 >= len(lines):
                    continue
                next_line = lines[idx + 1].strip()
                hbm_match = re.search(r"(\d+)\s*/\s*(\d+)\s*\|$", next_line)
                if not hbm_match:
                    continue
                used_mb = int(hbm_match.group(1))
                total_mb = int(hbm_match.group(2))
                free_mb = max(0, total_mb - used_mb)
                status[device_id] = {
                    'used_mb': used_mb,
                    'total_mb': total_mb,
                    'free_mb': free_mb,
                }

            return status
        except Exception:
            return {}

    def _assigned_instance_count(self, device_id: int) -> int:
        return sum(1 for item in self.instances if item['device_id'] == device_id)

    def _pick_scale_device(self) -> Optional[int]:
        hbm_status = self._query_npu_hbm_status()
        candidates = []

        for order, device_id in enumerate(self.npu_device_ids):
            assigned = self._assigned_instance_count(device_id)
            if device_id in hbm_status:
                free_mb = hbm_status[device_id]['free_mb']
                can_fit = free_mb >= (self.instance_hbm_mb + self.hbm_safety_margin_mb)
                score = (0 if can_fit else 1, assigned, order, -free_mb)
            else:
                # 取不到 HBM 时，按优先级兜底，但优先少实例的卡
                can_fit = True
                free_mb = -1
                score = (2, assigned, order, 0)
            candidates.append((score, device_id, free_mb, can_fit))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        best_score, best_device_id, _, can_fit = candidates[0]
        if best_score[0] == 1 and not can_fit:
            return None
        return best_device_id

    def _create_instance_sync(self, device_id: int):
        server_kwargs = dict(self.ocr_kwargs)
        server_kwargs['npu_device_id'] = device_id
        server = OCRServer(**server_kwargs)
        now = time.time()
        with self._lock:
            instance = {
                'instance_id': self.instance_seq,
                'device_id': device_id,
                'server': server,
                'busy': False,
                'ready': True,
                'created_at': now,
                'last_used': now,
            }
            self.instance_seq += 1
            self.instances.append(instance)
            self._cond.notify_all()
        return instance

    def _create_instance_async(self, device_id: int):
        def _target():
            try:
                self._create_instance_sync(device_id)
            finally:
                with self._lock:
                    self.pending_instances = max(0, self.pending_instances - 1)
                    self._cond.notify_all()
        threading.Thread(
            target=_target,
            name=f"ocr-scaleup-npu-{device_id}",
            daemon=True
        ).start()

    def _idle_instances(self) -> List[Dict]:
        return [item for item in self.instances if item['ready'] and not item['busy']]

    def _total_instance_count(self) -> int:
        return len(self.instances) + self.pending_instances

    def _maybe_scale_up_locked(self):
        now = time.time()
        if self._total_instance_count() >= self.max_instances:
            return
        if self.pending_instances > 0:
            return
        if (now - self.last_scale_up_time) < self.scale_cooldown:
            return
        device_id = self._pick_scale_device()
        if device_id is None:
            return
        self.pending_instances += 1
        self.last_scale_up_time = now
        self._create_instance_async(device_id)

    def _acquire_instances(self, target_count: int, wait_for_scale: float = 0.0) -> List[Dict]:
        deadline = time.time() + wait_for_scale
        with self._cond:
            while True:
                idle_instances = sorted(
                    self._idle_instances(),
                    key=lambda item: (item['last_used'], item['instance_id'])
                )
                if len(idle_instances) >= target_count:
                    acquired = idle_instances[:target_count]
                    now = time.time()
                    for item in acquired:
                        item['busy'] = True
                        item['last_used'] = now
                    return acquired

                self._maybe_scale_up_locked()
                remaining = deadline - time.time()
                if remaining <= 0:
                    acquired = idle_instances[:target_count]
                    if acquired:
                        now = time.time()
                        for item in acquired:
                            item['busy'] = True
                            item['last_used'] = now
                    return acquired
                self._cond.wait(timeout=min(remaining, 0.5))

    def _acquire_at_least_one(self) -> Dict:
        while True:
            instances = self._acquire_instances(1, wait_for_scale=0.5)
            if instances:
                return instances[0]

    def _release_instances(self, instances: List[Dict]):
        with self._cond:
            now = time.time()
            for item in instances:
                item['busy'] = False
                item['last_used'] = now
            self._cond.notify_all()

    def _run_on_instance(self, instance: Dict, func_name: str, *args, **kwargs) -> Dict:
        server = instance['server']
        try:
            with server.lock:
                result = getattr(server, func_name)(*args, **kwargs)
            result['device_id'] = instance['device_id']
            result['instance_id'] = instance['instance_id']
            return result
        finally:
            self._release_instances([instance])

    def _idle_reaper_loop(self):
        while True:
            with self._cond:
                if self._shutdown:
                    return
                self._cond.wait(timeout=5)
                if self._shutdown:
                    return
                if len(self.instances) <= self.min_instances:
                    continue
                now = time.time()
                removable = []
                for item in sorted(self.instances, key=lambda x: x['last_used']):
                    if len(self.instances) - len(removable) <= self.min_instances:
                        break
                    if item['busy']:
                        continue
                    if (now - item['last_used']) < self.idle_timeout:
                        continue
                    removable.append(item)
                for item in removable:
                    self.instances.remove(item)

    def process_single_image(
        self,
        image_base64: str,
        format_output: bool = True,
        slice_params: Optional[Dict] = None
    ) -> Dict:
        instance = self._acquire_at_least_one()
        return self._run_on_instance(
            instance,
            'process_single_image',
            image_base64,
            format_output=format_output,
            slice_params=slice_params
        )

    def process_batch_images(
        self,
        images: List[str],
        format_output: bool = True,
        use_optimized: bool = True
    ) -> Dict:
        if not images:
            return {
                'success': True,
                'results': [],
                'processing_time': 0.0,
                'image_count': 0,
                'total_text_count': 0,
                'method_used': 'optimized' if use_optimized else 'traditional',
                'format_output': format_output,
                'device_ids': [],
            }

        start_time = time.time()
        desired_parallelism = min(len(images), self.max_instances)
        instances = self._acquire_instances(desired_parallelism, wait_for_scale=self.batch_acquire_wait)
        if not instances:
            instances = [self._acquire_at_least_one()]

        if len(instances) == 1:
            result = self._run_on_instance(
                instances[0],
                'process_batch_images',
                images,
                format_output=format_output,
                use_optimized=use_optimized
            )
            result['device_ids'] = [instances[0]['device_id']]
            return result

        method_used = 'optimized' if use_optimized else 'traditional'
        buckets = [[] for _ in instances]
        for index, image in enumerate(images):
            buckets[index % len(instances)].append((index, image))

        merged_results = [None] * len(images)
        total_text_count = 0
        used_device_ids = []

        with ThreadPoolExecutor(max_workers=len(instances)) as executor:
            future_map = {}
            for instance, bucket in zip(instances, buckets):
                if not bucket:
                    self._release_instances([instance])
                    continue
                bucket_images = [image for _, image in bucket]
                future = executor.submit(
                    self._run_on_instance,
                    instance,
                    'process_batch_images',
                    bucket_images,
                    format_output=format_output,
                    use_optimized=use_optimized
                )
                future_map[future] = (instance, bucket)

            for future in as_completed(future_map):
                instance, bucket = future_map[future]
                sub_result = future.result()
                if not sub_result['success']:
                    return {
                        'success': False,
                        'error': sub_result.get('error', f"instance {instance['instance_id']} failed"),
                        'processing_time': time.time() - start_time,
                        'image_count': len(images),
                        'total_text_count': 0,
                        'method_used': method_used,
                        'format_output': format_output,
                    }

                used_device_ids.append(instance['device_id'])
                total_text_count += sub_result.get('total_text_count', 0)
                for (original_index, _), item_result in zip(bucket, sub_result['results']):
                    merged_results[original_index] = item_result

        return {
            'success': True,
            'results': merged_results,
            'processing_time': time.time() - start_time,
            'image_count': len(images),
            'total_text_count': total_text_count,
            'method_used': f"{method_used}_elastic",
            'format_output': format_output,
            'device_ids': used_device_ids,
        }

    def shutdown(self, wait_timeout: float = 5.0):
        """优雅停止：通知 reaper 退出 + 等线程清理 + 标记池关闭。"""
        with self._cond:
            if self._shutdown:
                return
            self._shutdown = True
            self._cond.notify_all()

        reaper = getattr(self, '_reaper_thread', None)
        if reaper is not None and reaper.is_alive():
            reaper.join(timeout=max(0.0, wait_timeout))

    def get_pool_stats(self) -> Dict:
        with self._lock:
            ready_instances = [item for item in self.instances if item['ready']]
            busy_instances = [item for item in ready_instances if item['busy']]
            idle_instances = [item for item in ready_instances if not item['busy']]
            per_device = {}
            for item in ready_instances:
                key = str(item['device_id'])
                per_device[key] = per_device.get(key, 0) + 1
            return {
                'min_instances': self.min_instances,
                'max_instances': self.max_instances,
                'configured_device_ids': self.npu_device_ids,
                'instance_hbm_mb': self.instance_hbm_mb,
                'hbm_safety_margin_mb': self.hbm_safety_margin_mb,
                'ready_instance_count': len(ready_instances),
                'busy_instance_count': len(busy_instances),
                'idle_instance_count': len(idle_instances),
                'pending_instance_count': self.pending_instances,
                'instances_per_device': per_device,
                'npu_hbm_status': self._query_npu_hbm_status(),
            }


# ============================================================
#  MultiProcessOCRPool —— 多进程 worker 池（支持单卡多实例 + 动态扩缩容）
# ============================================================
#
# 设计要点见 plan: silly-greeting-whale.md
# - 每个 OCR 实例 = 一个独立 OS 进程，绕开同进程多线程下的 OCRServer.lock 串行
# - 父进程不 import torch_npu / 不加载模型，所有 NPU 触碰只发生在 worker 子进程
# - task_q/result_q 各一根，请求按 req_id demux 路由回 dispatcher
# - 当前实现：dispatch + demux 主体（C1+C2）；扩缩容 monitor 线程见 C3


def ocr_worker_main(
    worker_id: str,
    device_id: int,
    task_q,
    result_q,
    ready_event,
    busy_state,
    last_used_state,
    ocr_kwargs: dict,
):
    """子进程入口。所有 torch_npu / PaddleOCR 触碰都发生在这里。

    参数（顶层函数 + 仅 picklable 类型，可被 mp.spawn 启动）:
      worker_id          字符串 id（父进程指派，便于日志）
      device_id          NPU 设备号
      task_q / result_q  共享 mp.Queue，单向通信
      ready_event        mp.Event，模型加载完毕后由 worker 自己 set
      busy_state         mp.Value('b')，0=idle 1=busy，父进程读用于负载判断
      last_used_state    mp.Value('d')，最近一次完成任务的 wall-clock，父进程读用于缩容
      ocr_kwargs         传给 OCRServer 的 dict（不含 npu_device_id，函数内拼）
    """
    try:
        kwargs = dict(ocr_kwargs)
        kwargs['npu_device_id'] = device_id
        # 关键：OCRServer 实例化挪到子进程，父进程不再创建
        server = OCRServer(use_npu=True, **kwargs)
        ready_event.set()
    except Exception as exc:
        # init 失败时通过 result_q 报一次，父进程能感知到 worker 死掉
        try:
            result_q.put({
                'id': '__init_failed__',
                'ok': False,
                'error': f'worker {worker_id} init failed: {exc!r}',
                'worker_id': worker_id,
                'device_id': device_id,
            })
        except Exception:
            pass
        return

    while True:
        try:
            task = task_q.get()
        except (KeyboardInterrupt, SystemExit):
            return
        if task is None:
            # 优雅退出 sentinel
            return

        req_id = task['id']
        method = task['method']
        args = task['args']

        with busy_state.get_lock():
            busy_state.value = 1
        try:
            if method == 'single':
                payload = server.process_single_image(**args)
            elif method == 'batch':
                payload = server.process_batch_images(**args)
            else:
                payload = {'success': False, 'error': f'unknown method {method!r}'}
            result_q.put({
                'id': req_id,
                'ok': True,
                'payload': payload,
                'worker_id': worker_id,
                'device_id': device_id,
            })
        except Exception as exc:
            result_q.put({
                'id': req_id,
                'ok': False,
                'error': repr(exc),
                'worker_id': worker_id,
                'device_id': device_id,
            })
        finally:
            with busy_state.get_lock():
                busy_state.value = 0
            with last_used_state.get_lock():
                last_used_state.value = time.time()


class MultiProcessOCRPool:
    """多进程 OCR 实例池：每实例独占 OS 进程，支持同卡多实例 + 动态扩缩容。

    本类的接口契约与 ElasticOCRPool 保持一致，可在 startup_event 里平替：
      - process_single_image(image_base64, format_output, slice_params) -> Dict
      - process_batch_images(images, format_output, use_optimized) -> Dict
      - device_info: str  (property)
      - request_count / error_count: int 属性
      - get_pool_stats() -> Dict
      - shutdown(wait_timeout: float = 5.0)

    扩缩容 monitor 线程见后续 C3 实现，当前版本仅启动 min_instances 个 worker。
    """

    def __init__(
        self,
        npu_device_ids: List[int],
        min_instances: int = 1,
        max_instances: int = 32,
        per_card_max: int = 0,
        idle_timeout: int = 120,
        scale_cooldown: int = 15,
        batch_acquire_wait: float = 8.0,
        instance_hbm_mb: int = 8000,
        hbm_safety_margin_mb: int = 4096,
        result_timeout: float = 300.0,
        monitor_interval: float = 3.0,
        worker_init_timeout: float = 240.0,
        **ocr_kwargs,
    ):
        self.npu_device_ids = list(dict.fromkeys(npu_device_ids or [ocr_kwargs.get('npu_device_id', 0)]))
        self.min_instances = max(1, min_instances)
        self.max_instances = max(self.min_instances, max_instances)
        # per_card_max <= 0 表示不限制（仅靠 hbm 自然限制）
        self.per_card_max = per_card_max if per_card_max > 0 else 0
        self.idle_timeout = max(10, idle_timeout)
        self.scale_cooldown = max(0, scale_cooldown)
        self.batch_acquire_wait = max(0.0, batch_acquire_wait)
        self.instance_hbm_mb = max(1024, instance_hbm_mb)
        self.hbm_safety_margin_mb = max(0, hbm_safety_margin_mb)
        self.result_timeout = max(10.0, result_timeout)
        self.monitor_interval = max(1.0, monitor_interval)
        self.worker_init_timeout = max(30.0, worker_init_timeout)
        self.ocr_kwargs = dict(ocr_kwargs)
        # 父进程不能持有 npu_device_id，避免误用
        self.ocr_kwargs.pop('npu_device_id', None)

        self.request_count = 0
        self.error_count = 0
        self._mp_ctx = mp.get_context('spawn')
        self.task_q = self._mp_ctx.Queue()
        self.result_q = self._mp_ctx.Queue()
        self.workers: List[Dict] = []
        self.pending: Dict[str, Dict] = {}
        self._workers_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._shutdown = False
        self._worker_seq = 0
        self._last_scale_up = 0.0

        # 启动 demux 线程（必须在 spawn worker 之前，避免 worker init 失败信号丢失）
        self._demux_thread = threading.Thread(
            target=self._result_demux_loop,
            name='ocr-result-demux',
            daemon=True,
        )
        self._demux_thread.start()

        # 同步起 min_instances 个 worker
        primary_devices = self._initial_device_round_robin(self.min_instances)
        for device_id in primary_devices:
            self._spawn_worker(device_id)
        self._wait_for_workers_ready(self.workers, self.worker_init_timeout)

        # 启动 monitor 线程（C3）：周期性检查负载，扩/缩容
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name='ocr-pool-monitor',
            daemon=True,
        )
        self._monitor_thread.start()

    # --------- 公共接口（契约保持） ---------

    @property
    def device_info(self) -> str:
        with self._workers_lock:
            ready = [w for w in self.workers if w['ready'] and not w.get('retiring', False)]
            if not ready:
                return "unknown"
            counts: Dict[int, int] = {}
            for w in ready:
                counts[w['device_id']] = counts.get(w['device_id'], 0) + 1
            parts = [f"NPU-{d}x{n}" for d, n in sorted(counts.items())]
            return ",".join(parts)

    def process_single_image(self, image_base64: str, format_output: bool = True,
                             slice_params: Optional[Dict] = None) -> Dict:
        return self._dispatch('single', dict(
            image_base64=image_base64,
            format_output=format_output,
            slice_params=slice_params,
        ))

    def process_batch_images(self, images: List[str], format_output: bool = True,
                             use_optimized: bool = True) -> Dict:
        if not images:
            return {
                'success': True,
                'results': [],
                'processing_time': 0.0,
                'image_count': 0,
                'total_text_count': 0,
                'method_used': 'optimized' if use_optimized else 'traditional',
                'format_output': format_output,
                'device_ids': [],
            }
        return self._dispatch('batch', dict(
            images=images,
            format_output=format_output,
            use_optimized=use_optimized,
        ))

    def get_pool_stats(self) -> Dict:
        with self._workers_lock:
            ready_workers = [w for w in self.workers if w['ready']]
            busy_workers = [w for w in ready_workers if w['busy_state'].value != 0]
            idle_workers = [w for w in ready_workers if w['busy_state'].value == 0]
            per_device: Dict[str, int] = {}
            for w in ready_workers:
                key = str(w['device_id'])
                per_device[key] = per_device.get(key, 0) + 1
            return {
                'min_instances': self.min_instances,
                'max_instances': self.max_instances,
                'per_card_max': self.per_card_max,
                'configured_device_ids': self.npu_device_ids,
                'instance_hbm_mb': self.instance_hbm_mb,
                'hbm_safety_margin_mb': self.hbm_safety_margin_mb,
                'ready_instance_count': len(ready_workers),
                'busy_instance_count': len(busy_workers),
                'idle_instance_count': len(idle_workers),
                'pending_instance_count': sum(1 for w in self.workers if not w['ready']),
                'instances_per_device': per_device,
                'task_queue_size': self._safe_qsize(self.task_q),
            }

    def shutdown(self, wait_timeout: float = 5.0):
        if self._shutdown:
            return
        self._shutdown = True
        # 给每个 worker 投 sentinel
        with self._workers_lock:
            n_workers = len(self.workers)
        for _ in range(n_workers):
            try:
                self.task_q.put(None, timeout=1.0)
            except Exception:
                break
        # 等 worker 退出
        deadline = time.time() + wait_timeout
        with self._workers_lock:
            workers_snapshot = list(self.workers)
        for w in workers_snapshot:
            remaining = max(0.1, deadline - time.time())
            try:
                w['process'].join(timeout=remaining)
            except Exception:
                pass
            if w['process'].is_alive():
                try:
                    w['process'].kill()
                except Exception:
                    pass

    # --------- 内部 ---------

    def _dispatch(self, method: str, args: Dict) -> Dict:
        """投任务进 task_q，按 req_id 等结果回来。"""
        self.request_count += 1
        if self._shutdown:
            self.error_count += 1
            return {'success': False, 'error': 'pool is shutting down'}

        # 至少要有一个 ready worker，否则原地等一小段时间
        if not self._has_ready_worker(wait_seconds=self.batch_acquire_wait):
            self.error_count += 1
            return {'success': False, 'error': 'no ready OCR worker available'}

        req_id = uuid.uuid4().hex
        ev = threading.Event()
        slot = {'event': ev, 'msg': None}
        with self._pending_lock:
            self.pending[req_id] = slot

        try:
            self.task_q.put({'id': req_id, 'method': method, 'args': args})
        except Exception as exc:
            with self._pending_lock:
                self.pending.pop(req_id, None)
            self.error_count += 1
            return {'success': False, 'error': f'failed to enqueue task: {exc!r}'}

        if not ev.wait(timeout=self.result_timeout):
            with self._pending_lock:
                self.pending.pop(req_id, None)
            self.error_count += 1
            return {'success': False, 'error': f'timed out after {self.result_timeout}s'}

        msg = slot['msg']
        if msg is None or not msg.get('ok'):
            self.error_count += 1
            err = msg.get('error') if msg else 'no result'
            return {'success': False, 'error': err}
        return msg['payload']

    def _result_demux_loop(self):
        """单线程读 result_q，按 req_id 路由到 pending 等待者。"""
        while not self._shutdown:
            try:
                msg = self.result_q.get(timeout=0.5)
            except _queue.Empty:
                continue
            except Exception:
                continue

            req_id = msg.get('id')
            if req_id == '__init_failed__':
                # worker init 失败，标记并打印；后续 monitor 再决定是否重启
                print(f"[pool] worker init failed: {msg.get('error')}", flush=True)
                continue

            with self._pending_lock:
                slot = self.pending.pop(req_id, None)
            if slot is None:
                continue  # 调用方已经超时放弃
            slot['msg'] = msg
            slot['event'].set()

    def _spawn_worker(self, device_id: int) -> Dict:
        """spawn 一个新 worker 进程。返回 worker dict（未必 ready）。"""
        with self._workers_lock:
            self._worker_seq += 1
            worker_id = f"ocr-w{self._worker_seq}-npu{device_id}"

        ready_event = self._mp_ctx.Event()
        busy_state = self._mp_ctx.Value('b', 0)
        last_used_state = self._mp_ctx.Value('d', time.time())

        process = self._mp_ctx.Process(
            target=ocr_worker_main,
            name=worker_id,
            args=(worker_id, device_id, self.task_q, self.result_q,
                  ready_event, busy_state, last_used_state, self.ocr_kwargs),
            daemon=True,
        )
        process.start()
        worker = {
            'worker_id': worker_id,
            'device_id': device_id,
            'process': process,
            'ready_event': ready_event,
            'busy_state': busy_state,
            'last_used_state': last_used_state,
            'ready': False,
            'retiring': False,
            'spawned_at': time.time(),
        }
        with self._workers_lock:
            self.workers.append(worker)
        return worker

    def _wait_for_workers_ready(self, workers: List[Dict], timeout: float) -> List[Dict]:
        deadline = time.time() + timeout
        ready: List[Dict] = []
        for w in workers:
            remaining = max(0.0, deadline - time.time())
            if w['ready_event'].wait(timeout=remaining):
                w['ready'] = True
                ready.append(w)
                # ready 状态变化打印一行，便于观察扩容是否生效
                ready_count = sum(1 for x in self.workers if x['ready'])
                print(f"[pool] worker {w['worker_id']} ready (now ready_count={ready_count})", flush=True)
            else:
                print(f"[pool] worker {w['worker_id']} init timeout after {timeout}s", flush=True)
        return ready

    def _has_ready_worker(self, wait_seconds: float = 0.0) -> bool:
        deadline = time.time() + wait_seconds
        while True:
            with self._workers_lock:
                if any(w['ready'] and not w.get('retiring', False) for w in self.workers):
                    return True
            if time.time() >= deadline:
                return False
            time.sleep(0.2)

    def _initial_device_round_robin(self, n: int) -> List[int]:
        """min_instances 初始分布：每张卡轮转，避免初始全堆在卡 0。"""
        if not self.npu_device_ids:
            return []
        result = []
        for i in range(n):
            result.append(self.npu_device_ids[i % len(self.npu_device_ids)])
        return result

    @staticmethod
    def _safe_qsize(q) -> int:
        try:
            return q.qsize()
        except (NotImplementedError, OSError):
            return -1

    # --------- 扩缩容（C3） ---------

    def _query_npu_hbm_status(self) -> Dict[int, Dict[str, int]]:
        """读取 npu-smi info，返回每张卡的 HBM 使用情况。结构与 ElasticOCRPool 一致。"""
        try:
            completed = subprocess.run(
                ["npu-smi", "info"],
                capture_output=True, text=True, timeout=10, check=False,
            )
            output = completed.stdout or ""
            lines = output.splitlines()
            status: Dict[int, Dict[str, int]] = {}
            for idx, line in enumerate(lines):
                stripped = line.strip()
                if not stripped.startswith("|"):
                    continue
                m = re.match(r"^\|\s*(\d+)\s+", stripped)
                if not m:
                    continue
                device_id = int(m.group(1))
                if device_id not in self.npu_device_ids:
                    continue
                if idx + 1 >= len(lines):
                    continue
                next_line = lines[idx + 1].strip()
                hbm_m = re.search(r"(\d+)\s*/\s*(\d+)\s*\|$", next_line)
                if not hbm_m:
                    continue
                used_mb = int(hbm_m.group(1))
                total_mb = int(hbm_m.group(2))
                status[device_id] = {
                    'used_mb': used_mb,
                    'total_mb': total_mb,
                    'free_mb': max(0, total_mb - used_mb),
                }
            return status
        except Exception:
            return {}

    def _assigned_count(self, device_id: int) -> int:
        with self._workers_lock:
            return sum(1 for w in self.workers
                       if w['device_id'] == device_id and not w.get('retiring', False))

    def _pick_device_for_scale_up(self) -> Optional[int]:
        """填卡优先：从 npu_device_ids 中挑一张能再塞一个实例的卡。

        排序键 (能容纳, -已分配, order, -free_mb)：
          - 已分配多的优先（填满当前卡，避免分散）
          - per_card_max > 0 时必须 assigned < per_card_max；per_card_max <= 0 表示不限制
          - 必须 npu-smi 取得到 hbm 且 free_mb >= instance_hbm_mb + safety_margin
            （取不到 hbm 视为"不确定是否被外部占用"，保守拒绝扩容，避免 OOM）
        """
        hbm_status = self._query_npu_hbm_status()
        if not hbm_status:
            print(f"[pool] scale-up declined: npu-smi gave no HBM info; refusing to risk OOM",
                  flush=True)
            return None

        candidates = []
        rejected_reasons = []
        for order, device_id in enumerate(self.npu_device_ids):
            assigned = self._assigned_count(device_id)
            if self.per_card_max > 0 and assigned >= self.per_card_max:
                rejected_reasons.append(f"npu:{device_id}(per_card_max={self.per_card_max} reached)")
                continue
            if device_id not in hbm_status:
                rejected_reasons.append(f"npu:{device_id}(no hbm info)")
                continue
            free_mb = hbm_status[device_id]['free_mb']
            need_mb = self.instance_hbm_mb + self.hbm_safety_margin_mb
            if free_mb < need_mb:
                rejected_reasons.append(f"npu:{device_id}(free={free_mb}MB < need={need_mb}MB)")
                continue
            score = (-assigned, order, -free_mb)
            candidates.append((score, device_id))
        if not candidates:
            if rejected_reasons:
                print(f"[pool] scale-up declined: {'; '.join(rejected_reasons)}", flush=True)
            return None
        candidates.sort()
        return candidates[0][1]

    def _busy_ready_idle_counts(self):
        with self._workers_lock:
            ready = [w for w in self.workers if w['ready'] and not w.get('retiring', False)]
            busy = sum(1 for w in ready if w['busy_state'].value != 0)
            idle = len(ready) - busy
            total_alive = sum(1 for w in self.workers if not w.get('retiring', False))
            pending = total_alive - len(ready)
        return len(ready), busy, idle, pending, total_alive

    def _should_scale_up(self) -> bool:
        if self._shutdown:
            return False
        ready, busy, idle, pending, total = self._busy_ready_idle_counts()
        if total + pending >= self.max_instances:
            return False
        # 扩容冷却：避免 monitor loop 每个 tick 都疯狂 spawn。
        # 注意不要因 pending>0 一票否决——同卡多实例可并行加载，应允许在 cooldown
        # 间隔内连续触发，否则单 worker init 慢时其它请求会被卡住。
        if (time.time() - self._last_scale_up) < self.scale_cooldown:
            return False
        # 触发条件：有积压 或 所有就绪 worker 都在忙
        qsize = self._safe_qsize(self.task_q)
        has_pressure = (qsize and qsize > 0) or (ready > 0 and busy == ready)
        if not has_pressure:
            return False
        return self._pick_device_for_scale_up() is not None

    def _scale_up_one(self):
        device_id = self._pick_device_for_scale_up()
        if device_id is None:
            return
        print(f"[pool] scale up: spawning new worker on npu:{device_id}", flush=True)
        worker = self._spawn_worker(device_id)
        self._last_scale_up = time.time()
        # 异步等 ready，不阻塞 monitor loop
        threading.Thread(
            target=self._wait_for_workers_ready,
            args=([worker], self.worker_init_timeout),
            name=f"ocr-scaleup-wait-{worker['worker_id']}",
            daemon=True,
        ).start()

    def _pick_worker_to_retire(self) -> Optional[Dict]:
        """缩容选 worker：必须 idle、ready、未 retiring；优先选同卡上有兄弟的（即 device_id assigned > 1），
        且 last_used 最早的。保证每张卡至少留 1 个实例（直到 total <= min_instances）。"""
        now = time.time()
        with self._workers_lock:
            candidates = []
            for w in self.workers:
                if not w['ready'] or w.get('retiring', False):
                    continue
                if w['busy_state'].value != 0:
                    continue
                if (now - w['last_used_state'].value) < self.idle_timeout:
                    continue
                same_card = sum(1 for x in self.workers
                                if x['device_id'] == w['device_id']
                                and not x.get('retiring', False))
                # 优先缩同卡多实例
                priority = 0 if same_card > 1 else 1
                candidates.append((priority, w['last_used_state'].value, w))
        if not candidates:
            return None
        candidates.sort(key=lambda x: (x[0], x[1]))
        return candidates[0][2]

    def _maybe_scale_down(self):
        if self._shutdown:
            return
        ready, busy, idle, pending, total = self._busy_ready_idle_counts()
        if total <= self.min_instances:
            return
        target = self._pick_worker_to_retire()
        if target is None:
            return
        print(f"[pool] scale down: retiring worker {target['worker_id']} (npu:{target['device_id']})", flush=True)
        with self._workers_lock:
            target['retiring'] = True
        # 投一个 None sentinel，target worker 会接住它退出
        try:
            self.task_q.put(None, timeout=1.0)
        except Exception:
            pass
        # 异步等它真的死掉再从 workers 列表里清除
        threading.Thread(
            target=self._reap_retired_worker,
            args=(target,),
            name=f"ocr-reap-{target['worker_id']}",
            daemon=True,
        ).start()

    def _reap_retired_worker(self, worker: Dict, kill_after: float = 30.0):
        worker['process'].join(timeout=kill_after)
        if worker['process'].is_alive():
            print(f"[pool] worker {worker['worker_id']} did not exit in {kill_after}s, killing", flush=True)
            try:
                worker['process'].kill()
            except Exception:
                pass
        with self._workers_lock:
            try:
                self.workers.remove(worker)
            except ValueError:
                pass

    def _monitor_loop(self):
        """每 monitor_interval 秒检查一次负载，触发扩/缩容/死进程清理。"""
        while not self._shutdown:
            try:
                self._reap_dead_workers()    # C5: 清理已死的 worker
                self._respawn_to_min()       # C5: 死掉后实例数若低于 min 自动补
                if self._should_scale_up():
                    self._scale_up_one()
                else:
                    self._maybe_scale_down()
            except Exception as exc:
                print(f"[pool] monitor loop error: {exc!r}", flush=True)
            # 短睡一段，再检查 shutdown
            for _ in range(int(self.monitor_interval * 10)):
                if self._shutdown:
                    return
                time.sleep(0.1)

    # --------- C5 watchdog ---------

    def _reap_dead_workers(self):
        """检测进程已退出但还在 self.workers 列表里的项，清理。"""
        with self._workers_lock:
            dead = [w for w in self.workers
                    if not w.get('retiring', False) and not w['process'].is_alive()]
        for w in dead:
            print(f"[pool] watchdog: worker {w['worker_id']} (pid={w['process'].pid}) "
                  f"died unexpectedly, cleaning up", flush=True)
            with self._workers_lock:
                try:
                    self.workers.remove(w)
                except ValueError:
                    pass

    def _respawn_to_min(self):
        """死掉后 ready+pending 总数低于 min_instances 时补回去。"""
        if self._shutdown:
            return
        ready, busy, idle, pending, total = self._busy_ready_idle_counts()
        deficit = self.min_instances - (total + pending)
        if deficit <= 0:
            return
        for _ in range(deficit):
            device_id = self._pick_device_for_scale_up()
            if device_id is None:
                return
            print(f"[pool] watchdog: respawning to reach min_instances={self.min_instances}, "
                  f"target=npu:{device_id}", flush=True)
            worker = self._spawn_worker(device_id)
            threading.Thread(
                target=self._wait_for_workers_ready,
                args=([worker], self.worker_init_timeout),
                name=f"ocr-respawn-wait-{worker['worker_id']}",
                daemon=True,
            ).start()


ocr_server = None

# 创建FastAPI应用
app = FastAPI(
    title="PytorchPaddleOCR 推理服务",
    description="基于PytorchPaddleOCR的高性能OCR推理服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """服务启动事件"""
    global ocr_server
    
    print("🚀 正在启动OCR推理服务...")
    
    try:
        # 从环境变量读取配置
        use_angle_cls = os.getenv('OCR_USE_ANGLE_CLS', 'True').lower() == 'true'
        npu_device_ids_env = os.getenv('OCR_NPU_DEVICE_IDS', '').strip()
        if npu_device_ids_env:
            npu_device_ids = [
                int(device_id.strip())
                for device_id in npu_device_ids_env.split(',')
                if device_id.strip()
            ]
        else:
            npu_device_ids = [int(os.getenv('OCR_NPU_DEVICE_ID', '1'))]
        
        # 初始化OCR服务配置 - 从环境变量读取参数
        ocr_config = {
            'use_angle_cls': use_angle_cls,
            'npu_device_id': npu_device_ids[0],
            
            # 模型路径配置
            'det_model_path': os.getenv('OCR_DET_MODEL_PATH', './models/ptocr_v5_server_det.pth'),
            'rec_model_path': os.getenv('OCR_REC_MODEL_PATH', './models/ptocr_v5_server_rec.pth'),
            'cls_model_path': os.getenv('OCR_CLS_MODEL_PATH', './models/ch_ptocr_mobile_v2.0_cls_infer.pth'),
            'rec_char_dict_path': os.getenv('OCR_REC_CHAR_DICT_PATH', './pytorchocr/utils/dict/ppocrv5_dict.txt'),
            
            # 模型配置文件路径
            'det_yaml_path': os.getenv('OCR_DET_YAML_PATH', 'configs/det/PP-OCRv5/PP-OCRv5_server_det.yml'),
            'rec_yaml_path': os.getenv('OCR_REC_YAML_PATH', 'configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml'),
            
            # 模型输入形状配置
            'rec_image_shape': os.getenv('OCR_REC_IMAGE_SHAPE', '3,48,320'),
            'cls_image_shape': os.getenv('OCR_CLS_IMAGE_SHAPE', '3,48,192'),
            
            # 检测模型参数
            'det_db_thresh': float(os.getenv('OCR_DET_DB_THRESH', '0.12')),
            'det_db_box_thresh': float(os.getenv('OCR_DET_DB_BOX_THRESH', '0.15')),
            'det_limit_side_len': int(os.getenv('OCR_DET_LIMIT_SIDE_LEN', '960')),
            'det_db_unclip_ratio': float(os.getenv('OCR_DET_DB_UNCLIP_RATIO', '1.8')),
            'drop_score': float(os.getenv('OCR_DROP_SCORE', '0.5')),
            
            # 识别模型参数
            'max_text_length': int(os.getenv('OCR_MAX_TEXT_LENGTH', '25')),
            'use_space_char': os.getenv('OCR_USE_SPACE_CHAR', 'True').lower() == 'true',
            
            # 分类模型参数
            'cls_thresh': float(os.getenv('OCR_CLS_THRESH', '0.9')),
            
            # 性能优化参数
            'cls_batch_num': int(os.getenv('OCR_CLS_BATCH_NUM', '24')),
            'rec_batch_num': int(os.getenv('OCR_REC_BATCH_NUM', '12')),
            
            # 图像处理参数
            'original_size_threshold': int(os.getenv('OCR_ORIGINAL_SIZE_THRESHOLD', '4000000')),
            'max_progressive_attempts': int(os.getenv('OCR_MAX_PROGRESSIVE_ATTEMPTS', '5'))
        }
        
        elastic_config = {
            'npu_device_ids': npu_device_ids,
            'min_instances': int(os.getenv('OCR_MIN_INSTANCES', '1')),
            'max_instances': int(os.getenv('OCR_MAX_INSTANCES', '32')),
            'per_card_max': int(os.getenv('OCR_PER_CARD_MAX', '0')),
            'idle_timeout': int(os.getenv('OCR_IDLE_TIMEOUT', '120')),
            'scale_cooldown': int(os.getenv('OCR_SCALE_COOLDOWN', '15')),
            'batch_acquire_wait': float(os.getenv('OCR_BATCH_ACQUIRE_WAIT', '8')),
            'instance_hbm_mb': int(os.getenv('OCR_INSTANCE_HBM_MB', '8000')),
            'hbm_safety_margin_mb': int(os.getenv('OCR_HBM_SAFETY_MARGIN_MB', '4096')),
            'monitor_interval': float(os.getenv('OCR_MONITOR_INTERVAL', '3.0')),
            'worker_init_timeout': float(os.getenv('OCR_WORKER_INIT_TIMEOUT', '300.0')),
        }

        # 使用多进程池：每实例一个独立 OS 进程，支持同卡多实例 + 动态扩缩容
        ocr_server = MultiProcessOCRPool(**elastic_config, **ocr_config)

        cls_status = "启用" if use_angle_cls else "禁用"
        per_card_str = elastic_config['per_card_max'] if elastic_config['per_card_max'] > 0 else "unlimited"
        print(f"OCR推理服务启动成功")
        print(f"  - 设备: {ocr_server.device_info}")
        print(f"  - 文本方向分类: {cls_status}")
        print(f"  - Batch配置: 分类=24, 识别=12 (优化模式)")
        print(f"  - Pool: min={elastic_config['min_instances']}, max={elastic_config['max_instances']}, "
              f"per_card_max={per_card_str}, "
              f"instance_hbm_mb={elastic_config['instance_hbm_mb']}, "
              f"safety_margin_mb={elastic_config['hbm_safety_margin_mb']}")
        
    except Exception as e:
        print(f"OCR推理服务启动失败: {e}")
        print(f"详细错误: {traceback.format_exc()}")
        sys.exit(1)


@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭事件 - 优雅释放 NPU 资源。"""
    global ocr_server

    print("正在关闭OCR推理服务...")
    pool = ocr_server
    ocr_server = None
    if pool is not None and hasattr(pool, 'shutdown'):
        try:
            pool.shutdown()
        except Exception as exc:
            print(f"OCR pool shutdown error: {exc}")
    elif pool is not None and hasattr(pool, '_shutdown'):
        pool._shutdown = True
    print("OCR推理服务已关闭")


@app.get("/", summary="首页", description="服务首页")
async def root():
    """服务首页"""
    return {
        "message": "PytorchPaddleOCR 推理服务",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "info": "/info"
    }


_HEALTH_PROBE_CACHE = {"ts": 0.0, "ok": False, "ttl": 30.0}
_HEALTH_PROBE_LOCK = threading.Lock()


def _run_health_inference_probe() -> bool:
    """对 OCR pool 跑一次轻量真实推理；30s 缓存避免每次探针都打满 NPU。"""
    global _HEALTH_PROBE_CACHE
    now = time.time()
    cached = _HEALTH_PROBE_CACHE
    if (now - cached["ts"]) < cached["ttl"]:
        return cached["ok"]

    with _HEALTH_PROBE_LOCK:
        # double-check after acquiring lock
        cached = _HEALTH_PROBE_CACHE
        if (time.time() - cached["ts"]) < cached["ttl"]:
            return cached["ok"]

        ok = False
        try:
            if ocr_server is not None:
                tiny = np.full((32, 64, 3), 255, dtype=np.uint8)
                ok_enc, buf = cv2.imencode(".jpg", tiny)
                if ok_enc:
                    image_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                    result = ocr_server.process_single_image(
                        image_b64,
                        format_output=True,
                        slice_params=None,
                    )
                    ok = bool(result and result.get("success"))
        except Exception as exc:
            print(f"health probe failed: {exc}")
            ok = False

        _HEALTH_PROBE_CACHE = {"ts": time.time(), "ok": ok, "ttl": cached["ttl"]}
        return ok


@app.get("/health", response_model=HealthResponse, summary="健康检查")
async def health_check(probe: int = 0):
    """
    健康检查接口。

    - 默认 (`/health`)：仅检查 pool 是否就绪，轻量、可用于高频探针。
    - `/health?probe=1`：跑一次真实 OCR 推理（带 30s 缓存），用于深度健康检测。
    """
    global ocr_server

    try:
        if ocr_server is None:
            return HealthResponse(
                status="unhealthy",
                timestamp=time.time(),
                device_info="unknown",
                model_loaded=False,
            )

        if hasattr(ocr_server, 'get_pool_stats'):
            pool_stats = ocr_server.get_pool_stats()
            ready_count = pool_stats.get('ready_instance_count', 0)
        else:
            ready_count = 1 if getattr(ocr_server, 'ocr_instance', None) is not None else 0

        if probe and ready_count > 0:
            ready_count = ready_count if _run_health_inference_probe() else 0

        return HealthResponse(
            status="healthy" if ready_count > 0 else "unhealthy",
            timestamp=time.time(),
            device_info=ocr_server.device_info if ocr_server else "unknown",
            model_loaded=ready_count > 0,
        )

    except Exception as exc:
        print(f"health_check error: {exc}")
        return HealthResponse(
            status="error",
            timestamp=time.time(),
            device_info="unknown",
            model_loaded=False,
        )


@app.get("/info", response_model=InfoResponse, summary="服务信息")
async def get_info():
    """获取服务信息"""
    global ocr_server
    
    return InfoResponse(
        service_name="PytorchPaddleOCR 推理服务",
        version="1.0.0",
        device_info=ocr_server.device_info if ocr_server else "unknown",
        supported_formats=["JPG", "JPEG", "PNG", "BMP", "TIFF", "WEBP"],
        max_image_size="10000x10000",
    )


@app.post("/ocr/single", summary="单图OCR推理")
def single_ocr(request: OCRRequest):
    """
    单图OCR推理接口
    
    支持的图像格式: JPG, JPEG, PNG, 
    默认返回格式化结果（包含分行信息和标准的words_block格式）
    
    返回格式:
    {
        "result": {
            "direction": 0,
            "words_block_list": [
                {
                    "words": "识别的文本",
                    "confidence": 0.95,
                    "location": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                }
            ],
            "markdown_result": "格式化文本",
            "words_block_count": 10
        }
    }
    """
    global ocr_server
    
    if not ocr_server:
        raise HTTPException(status_code=503, detail="OCR服务未初始化")
    
    # 更新请求计数
    ocr_server.request_count += 1
    
    try:
        # 处理图像
        result = ocr_server.process_single_image(
            request.image,
            format_output=True,
            slice_params=None
        )
        
        if result['success']:
            # 直接返回字典格式，确保与您期望的访问方式兼容
            return {
                "result": {
                    "markdown_result": result['results']['formatted_text'],
                    "words_block_count": len(result['results']['raw_results']),
                    "direction": 0,
                    "words_block_list": [
                        {
                            "words": res['text'],
                            "confidence": res['confidence'],
                            "location": res['bbox']
                        } for res in result['results']['raw_results']
                    ],

                }
            }
        else:
            ocr_server.error_count += 1
            raise HTTPException(status_code=400, detail=result.get('error', '未知错误'))
            
    except HTTPException:
        raise
    except Exception as e:
        ocr_server.error_count += 1
        error_msg = f"服务内部错误: {str(e)}"
        print(f"单图OCR异常: {error_msg}")
        print(f"详细错误: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/ocr/batch", summary="批量OCR推理")
def batch_ocr(request: BatchOCRRequest):
    """
    批量OCR推理接口
    
    支持的图像格式: JPG, JPEG, PNG, 
    推荐使用优化模式以获得更好的性能
    默认返回格式化结果（包含分行信息和标准的words_block格式）
    
    返回格式:
    {
        "results": [
            {
                "direction": 0,
                "words_block_list": [
                    {
                        "words": "识别的文本",
                        "confidence": 0.95,
                        "location": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    }
                ],
                "markdown_result": "格式化文本",
                "words_block_count": 10
            }
        ],
        "processing_time": 1.23,
        "image_count": 8,
        "total_text_count": 80,
        "method_used": "optimized"
    }
    """
    global ocr_server
    
    if not ocr_server:
        raise HTTPException(status_code=503, detail="OCR服务未初始化")
    
    # 检查图像数量限制
    if len(request.images) > 100:
        raise HTTPException(status_code=400, detail="单次批量处理图像数量不能超过100张")
    
    # 更新请求计数
    ocr_server.request_count += 1
    
    try:
        # 批量处理图像
        result = ocr_server.process_batch_images(
            request.images,
            format_output=True,
            use_optimized=request.use_optimized
        )
        
        if result['success']:
            # 直接返回字典格式，确保与您期望的访问方式兼容
            return {
                "results": [
                    {
                        "direction": 0,
                        "words_block_list": [
                            {
                                "words": res['text'],
                                "confidence": res['confidence'],
                                "location": res['bbox']
                            } for res in img_results['raw_results']
                        ],
                        "markdown_result": img_results['formatted_text'],
                        "words_block_count": len(img_results['raw_results'])
                    } for img_results in result['results']
                ],
                "processing_time": result['processing_time'],
                "image_count": result['image_count'],
                "total_text_count": result['total_text_count'],
                "method_used": result['method_used']
            }
        else:
            ocr_server.error_count += 1
            raise HTTPException(status_code=400, detail=result.get('error', '未知错误'))
            
    except HTTPException:
        raise
    except Exception as e:
        ocr_server.error_count += 1
        error_msg = f"服务内部错误: {str(e)}"
        print(f"批量OCR异常: {error_msg}")
        print(f"详细错误: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/ocr/upload", summary="文件上传OCR推理")
async def upload_ocr(
    file: UploadFile = File(...)
):
    """
    文件上传OCR推理接口
    
    支持直接上传图像文件进行OCR识别
    
    返回格式与/ocr/single接口相同:
    {
        "result": {
            "direction": 0,
            "words_block_list": [
                {
                    "words": "识别的文本",
                    "confidence": 0.95,
                    "location": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                }
            ],
            "markdown_result": "格式化文本",
            "words_block_count": 10
        }
    }
    """
    global ocr_server
    
    if not ocr_server:
        raise HTTPException(status_code=503, detail="OCR服务未初始化")
    
    # 检查文件类型
    allowed_types = ['image/jpeg', 'image/jpg', 'image/png']
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"不支持的文件类型: {file.content_type}. 支持的类型: {allowed_types}"
        )
    
    # 检查文件大小（10MB限制）
    # if file.size and file.size > 10 * 1024 * 1024:
    #     raise HTTPException(status_code=400, detail="文件大小不能超过10MB")
    
    try:
        # 读取文件内容
        file_content = await file.read()
        
        # 转换为Base64
        image_base64 = base64.b64encode(file_content).decode('utf-8')
        
        # 创建OCR请求
        ocr_request = OCRRequest(
            image=image_base64
        )
        
        # 执行OCR
        return single_ocr(ocr_request)
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"文件处理失败: {str(e)}"
        print(f"文件上传OCR异常: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/stats", summary="服务统计")
async def get_stats():
    """获取服务统计信息"""
    global ocr_server
    
    if not ocr_server:
        raise HTTPException(status_code=503, detail="OCR服务未初始化")
    
    pool_stats = ocr_server.get_pool_stats() if hasattr(ocr_server, 'get_pool_stats') else {}

    return {
        "request_count": ocr_server.request_count,
        "error_count": ocr_server.error_count,
        "success_rate": (
            (ocr_server.request_count - ocr_server.error_count) / ocr_server.request_count 
            if ocr_server.request_count > 0 else 0
        ),
        "device_info": ocr_server.device_info,
        "pool": pool_stats
    }


if __name__ == "__main__":
    # 抑制 /health /info 探针在 access log 中的噪音
    logging.getLogger("uvicorn.access").addFilter(_AccessLogProbeFilter())

    # 启动服务
    uvicorn.run(
        "ocr_server:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # 由于OCR模型不支持多进程，使用单进程
        reload=False,
        access_log=True
    ) 
