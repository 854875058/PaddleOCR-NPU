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
import subprocess
import re
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
        while not self._shutdown:
            time.sleep(5)
            with self._cond:
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
            'max_instances': int(os.getenv('OCR_MAX_INSTANCES', '3')),
            'idle_timeout': int(os.getenv('OCR_IDLE_TIMEOUT', '120')),
            'scale_cooldown': int(os.getenv('OCR_SCALE_COOLDOWN', '15')),
            'batch_acquire_wait': float(os.getenv('OCR_BATCH_ACQUIRE_WAIT', '8')),
            'instance_hbm_mb': int(os.getenv('OCR_INSTANCE_HBM_MB', '20000')),
            'hbm_safety_margin_mb': int(os.getenv('OCR_HBM_SAFETY_MARGIN_MB', '4096')),
        }

        ocr_server = ElasticOCRPool(**elastic_config, **ocr_config)
        
        cls_status = "启用" if use_angle_cls else "禁用"
        print(f"OCR推理服务启动成功")
        print(f"  - 设备: {ocr_server.device_info}")
        print(f"  - 文本方向分类: {cls_status}")
        print(f"  - Batch配置: 分类=24, 识别=12 (优化模式)")
        print(f"  - Elastic: min={elastic_config['min_instances']}, max={elastic_config['max_instances']}, batch_wait={elastic_config['batch_acquire_wait']}s")
        
    except Exception as e:
        print(f"OCR推理服务启动失败: {e}")
        print(f"详细错误: {traceback.format_exc()}")
        sys.exit(1)


@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭事件"""
    global ocr_server
    
    print("正在关闭OCR推理服务...")
    if ocr_server is not None and hasattr(ocr_server, '_shutdown'):
        ocr_server._shutdown = True
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


@app.get("/health", response_model=HealthResponse, summary="健康检查")
async def health_check():
    """健康检查接口"""
    global ocr_server
    
    try:
        if ocr_server is None:
            model_loaded = False
        elif hasattr(ocr_server, 'ocr_instance'):
            model_loaded = ocr_server.ocr_instance is not None
        else:
            pool_stats = ocr_server.get_pool_stats()
            model_loaded = pool_stats.get('ready_instance_count', 0) > 0
        
        return HealthResponse(
            status="healthy" if model_loaded else "unhealthy",
            timestamp=time.time(),
            device_info=ocr_server.device_info if ocr_server else "unknown",
            model_loaded=model_loaded
        )
        
    except Exception as e:
        return HealthResponse(
            status="error",
            timestamp=time.time(),
            device_info="unknown",
            model_loaded=False
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
    # 启动服务
    uvicorn.run(
        "ocr_server:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # 由于OCR模型不支持多进程，使用单进程
        reload=False,
        access_log=True
    ) 
