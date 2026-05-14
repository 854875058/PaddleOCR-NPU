#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR推理服务启动脚本
支持多种启动模式和配置选项
"""

import os
import sys
import argparse
import uvicorn


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="OCR推理服务启动脚本")

    # 服务配置
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务绑定地址")
    parser.add_argument("--port", type=int, default=8011, help="服务端口")

    # OCR基础配置
    parser.add_argument("--disable_angle_cls", action="store_true", help="禁用文本方向分类（默认启用）")
    parser.add_argument("--npu_device_id", type=int, default=1, help="NPU设备ID")
    parser.add_argument("--npu_device_ids", type=str, default="1,2,3", help="NPU设备优先级列表，逗号分隔，如 1,2,3")
    parser.add_argument("--min_instances", type=int, default=1, help="最小常驻实例数")
    parser.add_argument("--max_instances", type=int, default=3, help="实例池最大实例数")
    parser.add_argument("--idle_timeout", type=int, default=120, help="实例空闲多久后自动缩容，单位秒")
    parser.add_argument("--scale_cooldown", type=int, default=15, help="扩容冷却时间，单位秒")
    parser.add_argument("--batch_acquire_wait", type=float, default=8.0, help="批量请求为等待新实例就绪而额外等待的秒数")
    parser.add_argument("--instance_hbm_mb", type=int, default=20000, help="估算单个OCR实例占用的HBM，单位MB")
    parser.add_argument("--hbm_safety_margin_mb", type=int, default=4096, help="每张卡保留的HBM安全余量，单位MB")
    
    # 模型路径配置
    parser.add_argument("--det_model_path", type=str, default="./models/ptocr_v5_server_det.pth", help="检测模型路径")
    parser.add_argument("--rec_model_path", type=str, default="./models/ptocr_v5_server_rec.pth", help="识别模型路径")
    parser.add_argument("--cls_model_path", type=str, default="./models/ch_ptocr_mobile_v2.0_cls_infer.pth", help="分类模型路径")
    parser.add_argument("--rec_char_dict_path", type=str, default="./pytorchocr/utils/dict/ppocrv5_dict.txt", help="识别字典路径")
    
    # 模型配置文件路径
    parser.add_argument("--det_yaml_path", type=str, default="configs/det/PP-OCRv5/PP-OCRv5_server_det.yml", help="检测模型配置文件路径")
    parser.add_argument("--rec_yaml_path", type=str, default="configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml", help="识别模型配置文件路径")
    
    # 模型输入形状配置
    parser.add_argument("--rec_image_shape", type=str, default="3,48,320", help="识别模型输入图像形状")
    parser.add_argument("--cls_image_shape", type=str, default="3,48,192", help="分类模型输入图像形状")
    
    # 检测模型参数
    parser.add_argument("--det_db_thresh", type=float, default=0.12, help="检测阈值，越小检测越敏感")
    parser.add_argument("--det_db_box_thresh", type=float, default=0.15, help="边界框阈值")
    parser.add_argument("--det_limit_side_len", type=int, default=960, help="检测图像边长限制")
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.8, help="文本框扩展比例")
    parser.add_argument("--drop_score", type=float, default=0.5, help="置信度过滤阈值")
    
    # 识别模型参数
    parser.add_argument("--max_text_length", type=int, default=25, help="最大文本长度")
    parser.add_argument("--use_space_char", action="store_true", default=True, help="是否使用空格字符")
    
    # 分类模型参数
    parser.add_argument("--cls_thresh", type=float, default=0.9, help="分类置信度阈值")
    
    # 性能优化参数
    parser.add_argument("--cls_batch_num", type=int, default=24, help="分类批处理大小")
    parser.add_argument("--rec_batch_num", type=int, default=12, help="识别批处理大小")
    
    # 图像处理参数
    parser.add_argument("--original_size_threshold", type=int, default=4000000, help="原始图像处理阈值（像素）")
    parser.add_argument("--max_progressive_attempts", type=int, default=5, help="渐进式缩放最大尝试次数")

    return parser.parse_args()


def check_model_files(args):
    """检查模型文件"""
    # 基础模型文件
    required_files = [
        args.det_model_path,
        args.rec_model_path,
        args.rec_char_dict_path
    ]

    # 如果启用cls，添加cls模型文件
    # 默认启用CLS，除非明确禁用
    use_angle_cls = not args.disable_angle_cls
    if use_angle_cls:
        required_files.append(args.cls_model_path)

    missing_files = []

    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print(f"缺少模型文件:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\n请确保模型文件在正确路径下")
        if use_angle_cls and args.cls_model_path in missing_files:
            print(f"提示: 如果不需要文本方向分类，可以使用 --disable_angle_cls 参数")
        sys.exit(1)


def print_startup_info(args):
    """打印启动信息"""
    use_angle_cls = not args.disable_angle_cls

    print("🚀 PytorchPaddleOCR 推理服务")
    print("=" * 60)
    print(f"服务地址: http://{args.host}:{args.port}")
    print(f"API文档: http://{args.host}:{args.port}/docs")
    print(f"配置信息:")
    print(f"  - 处理模式: 同步处理 (简化架构)")
    print(f"  - 计算设备: NPU 弹性实例池")
    print(f"  - 设备优先级: {args.npu_device_ids}")
    print(f"  - Elastic: min={args.min_instances}, max={args.max_instances}, batch_wait={args.batch_acquire_wait}s")
    print(f"  - 文本方向分类: {'启用' if use_angle_cls else '禁用'}")
    if use_angle_cls:
        print(f"  - 分类模型: {args.cls_model_path}")
    print("=" * 60)


def main():
    """主函数"""
    args = parse_args()

    # 检查模型文件
    check_model_files(args)

    # 打印启动信息
    print_startup_info(args)

    # 设置环境变量传递配置给OCR服务
    use_angle_cls = not args.disable_angle_cls
    os.environ['OCR_USE_ANGLE_CLS'] = str(use_angle_cls)
    os.environ['OCR_NPU_DEVICE_ID'] = str(args.npu_device_id)
    os.environ['OCR_NPU_DEVICE_IDS'] = str(args.npu_device_ids)
    os.environ['OCR_MIN_INSTANCES'] = str(args.min_instances)
    os.environ['OCR_MAX_INSTANCES'] = str(args.max_instances)
    os.environ['OCR_IDLE_TIMEOUT'] = str(args.idle_timeout)
    os.environ['OCR_SCALE_COOLDOWN'] = str(args.scale_cooldown)
    os.environ['OCR_BATCH_ACQUIRE_WAIT'] = str(args.batch_acquire_wait)
    os.environ['OCR_INSTANCE_HBM_MB'] = str(args.instance_hbm_mb)
    os.environ['OCR_HBM_SAFETY_MARGIN_MB'] = str(args.hbm_safety_margin_mb)
    
    # 模型路径配置
    os.environ['OCR_DET_MODEL_PATH'] = args.det_model_path
    os.environ['OCR_REC_MODEL_PATH'] = args.rec_model_path
    os.environ['OCR_CLS_MODEL_PATH'] = args.cls_model_path
    os.environ['OCR_REC_CHAR_DICT_PATH'] = args.rec_char_dict_path
    
    # 模型配置文件路径
    os.environ['OCR_DET_YAML_PATH'] = args.det_yaml_path
    os.environ['OCR_REC_YAML_PATH'] = args.rec_yaml_path
    
    # 模型输入形状配置
    os.environ['OCR_REC_IMAGE_SHAPE'] = args.rec_image_shape
    os.environ['OCR_CLS_IMAGE_SHAPE'] = args.cls_image_shape
    
    # 检测模型参数
    os.environ['OCR_DET_DB_THRESH'] = str(args.det_db_thresh)
    os.environ['OCR_DET_DB_BOX_THRESH'] = str(args.det_db_box_thresh)
    os.environ['OCR_DET_LIMIT_SIDE_LEN'] = str(args.det_limit_side_len)
    os.environ['OCR_DET_DB_UNCLIP_RATIO'] = str(args.det_db_unclip_ratio)
    os.environ['OCR_DROP_SCORE'] = str(args.drop_score)
    
    # 识别模型参数
    os.environ['OCR_MAX_TEXT_LENGTH'] = str(args.max_text_length)
    os.environ['OCR_USE_SPACE_CHAR'] = str(args.use_space_char)
    
    # 分类模型参数
    os.environ['OCR_CLS_THRESH'] = str(args.cls_thresh)
    
    # 性能优化参数
    os.environ['OCR_CLS_BATCH_NUM'] = str(args.cls_batch_num)
    os.environ['OCR_REC_BATCH_NUM'] = str(args.rec_batch_num)
    
    # 图像处理参数
    os.environ['OCR_ORIGINAL_SIZE_THRESHOLD'] = str(args.original_size_threshold)
    os.environ['OCR_MAX_PROGRESSIVE_ATTEMPTS'] = str(args.max_progressive_attempts)

    # 启动服务
    try:
        # 抑制 /health /info 探针在 access log 中的噪音
        import logging
        from ocr_server import _AccessLogProbeFilter
        logging.getLogger("uvicorn.access").addFilter(_AccessLogProbeFilter())

        uvicorn.run(
            "ocr_server:app",
            host=args.host,
            port=args.port,
            workers=1,  # OCR模型不支持多进程，固定为1
            reload=False,  # 生产环境不需要热重载
            log_level="info",  # 固定日志级别
            access_log=True  # 启用访问日志
        )
    except KeyboardInterrupt:
        print("\n服务已停止")
    except Exception as e:
        print(f"服务启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
