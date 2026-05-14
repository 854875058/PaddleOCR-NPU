#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick OCR - 简化的OCR命令行工具
基于 pytorch_paddle.py 封装，提供简洁的命令行接口和可视化输出
"""

import os
import sys
import cv2
import time
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw
from typing import List

# 添加项目路径
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)

# 导入我们的OCR类
from pytorch_paddle import PytorchPaddleOCR

# 导入工具函数
sys.path.append(os.path.abspath(os.path.join(__dir__, 'tools/infer')))
sys.path.append(os.path.abspath(os.path.join(__dir__, 'pytorchocr/utils')))

from utility import get_image_file_list, check_and_read
from pytorchocr_utility import draw_ocr_box_txt


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Quick OCR - 简化的OCR命令行工具")
    
    # 输入输出参数
    parser.add_argument('--image_dir', type=str, required=True, help='图像文件或目录路径')
    parser.add_argument('--output_dir', type=str, default='./inference_results', 
                       help='输出目录，默认: ./inference_results')
    
    # 设备参数
    parser.add_argument('--use_npu', type=str, default='true', 
                       help='是否使用NPU，默认: true')
    parser.add_argument('--npu_device_id', type=int, default=0, 
                       help='NPU设备ID，默认: 0')
    
    # 模型参数
    parser.add_argument('--det_model_path', type=str, default='./models/ptocr_v5_server_det.pth',
                       help='检测模型路径')
    parser.add_argument('--rec_model_path', type=str, default='./models/ptocr_v5_server_rec.pth',
                       help='识别模型路径')
    parser.add_argument('--rec_char_dict_path', type=str, default='./pytorchocr/utils/dict/ppocrv5_dict.txt',
                       help='字符字典路径')
    parser.add_argument('--rec_image_shape', type=str, default='3,48,320',
                       help='识别图像尺寸')
    
    # 角度分类参数
    parser.add_argument('--use_angle_cls', type=str, default='true',
                       help='是否使用角度分类，默认: true')
    parser.add_argument('--cls_model_path', type=str, default='./models/ch_ptocr_mobile_v2.0_cls_infer.pth',
                       help='角度分类模型路径')
    
    # 检测参数
    parser.add_argument('--det_limit_type', type=str, default='min',
                       help='检测限制类型，默认: min')
    parser.add_argument('--det_db_thresh', type=float, default=0.12,
                       help='DB检测阈值，默认: 0.12')
    parser.add_argument('--det_db_box_thresh', type=float, default=0.15,
                       help='DB边界框阈值，默认: 0.15')
    parser.add_argument('--det_db_unclip_ratio', type=float, default=1.8,
                       help='DB反裁剪比例，默认: 1.8')
    parser.add_argument('--use_dilation', type=str, default='true',
                       help='是否使用膨胀操作，默认: true')
    
    # 其他参数
    parser.add_argument('--drop_score', type=float, default=0.0,
                       help='置信度阈值，默认: 0.0')
    parser.add_argument('--vis_font_path', type=str, default='./doc/fonts/simfang.ttf',
                       help='可视化字体路径')
    parser.add_argument('--save_txt_results', type=str, default='true',
                       help='是否保存文本结果，默认: true')
    parser.add_argument('--save_visual_results', type=str, default='true',
                       help='是否保存可视化结果，默认: true')
    
    return parser.parse_args()


def str_to_bool(v):
    """字符串转布尔值"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_ocr_instance(args):
    """创建OCR实例"""
    print("🚀 初始化OCR实例...")
    
    # 转换字符串参数为布尔值
    use_npu = str_to_bool(args.use_npu)
    use_angle_cls = str_to_bool(args.use_angle_cls)
    use_dilation = str_to_bool(args.use_dilation)
    
    try:
        ocr = PytorchPaddleOCR(
            use_npu=use_npu,
            npu_device_id=args.npu_device_id,
            det_model_path=args.det_model_path,
            rec_model_path=args.rec_model_path,
            rec_char_dict_path=args.rec_char_dict_path,
            rec_image_shape=args.rec_image_shape,
            use_angle_cls=use_angle_cls,
            cls_model_path=args.cls_model_path,
            drop_score=args.drop_score,
            # 检测参数
            det_limit_type=args.det_limit_type,
            det_db_thresh=args.det_db_thresh,
            det_db_box_thresh=args.det_db_box_thresh,
            det_db_unclip_ratio=args.det_db_unclip_ratio,
            use_dilation=use_dilation,
        )
        
        print("✅ OCR实例初始化成功")
        print(f"   设备: {'NPU' if use_npu else 'CPU'}")
        print(f"   角度分类: {'启用' if use_angle_cls else '禁用'}")
        print(f"   置信度阈值: {args.drop_score}")
        return ocr
        
    except Exception as e:
        print(f"❌ OCR实例初始化失败: {e}")
        sys.exit(1)


def save_visualization(image_path, dt_boxes, rec_res, output_dir, font_path, drop_score):
    """保存可视化结果"""
    try:
        # 读取原图
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️  无法读取图像: {image_path}")
            return None
            
        # 转换为PIL图像
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # 检查是否有OCR结果
        has_results = (dt_boxes is not None and 
                      len(dt_boxes) > 0 and 
                      rec_res is not None and 
                      len(rec_res) > 0)
        
        if has_results:
            # 准备数据
            boxes = dt_boxes
            texts = [res['text'] for res in rec_res]
            scores = [res['confidence'] for res in rec_res]
            
            # 绘制OCR结果
            draw_img = draw_ocr_box_txt(
                image, boxes, texts, scores, 
                drop_score=drop_score, font_path=font_path
            )
        else:
            # 如果没有检测结果，返回原图
            draw_img = np.array(image)
        
        # 保存图像
        filename = os.path.basename(image_path)
        save_path = os.path.join(output_dir, f"vis_{filename}")
        cv2.imwrite(save_path, draw_img[:, :, ::-1])  # RGB转BGR
        
        return save_path
        
    except Exception as e:
        print(f"⚠️  保存可视化结果失败: {e}")
        return None


def process_single_image(ocr, image_path, output_dir, args):
    """处理单张图像"""
    print(f"\n📸 处理图像: {os.path.basename(image_path)}")
    
    start_time = time.time()
    
    try:
        # OCR识别
        results = ocr.ocr(image_path)
        
        # 转换为predict_system.py格式的结果
        dt_boxes = []
        rec_res = []
        
        if results:
            for result in results:
                # 转换bbox格式
                bbox = np.array(result['bbox'])
                dt_boxes.append(bbox)
                rec_res.append(result)
        
        # 只有当有结果时才转换为numpy数组
        if dt_boxes:
            dt_boxes = np.array(dt_boxes)
        else:
            dt_boxes = None
        
        elapsed_time = time.time() - start_time
        
        # 输出识别结果
        if rec_res:
            print(f"✅ 识别完成，用时: {elapsed_time:.3f}s，识别到 {len(rec_res)} 个文本区域")
            for i, result in enumerate(rec_res):
                print(f"   {i+1}. {result['text']} (置信度: {result['confidence']:.3f})")
        else:
            print(f"⚠️  未检测到文本内容，用时: {elapsed_time:.3f}s")
        
        # 保存可视化结果
        vis_path = None
        if str_to_bool(args.save_visual_results):
            vis_path = save_visualization(
                image_path, dt_boxes, rec_res, output_dir, 
                args.vis_font_path, args.drop_score
            )
            if vis_path:
                print(f"🖼️  可视化结果已保存: {vis_path}")
        
        # 准备保存的结果
        save_result = {
            'image_path': image_path,
            'results': []
        }
        
        if rec_res:
            for result in rec_res:
                save_result['results'].append({
                    "transcription": result['text'],
                    "points": result['bbox'],
                    "confidence": result['confidence']
                })
        
        return save_result, elapsed_time
        
    except Exception as e:
        print(f"❌ 处理图像失败: {e}")
        return None, 0


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 60)
    print("🔥 Quick OCR - 简化的OCR命令行工具")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建OCR实例
    ocr = create_ocr_instance(args)
    
    # 获取图像文件列表
    image_file_list = get_image_file_list(args.image_dir)
    
    if not image_file_list:
        print(f"❌ 在 {args.image_dir} 中未找到图像文件")
        sys.exit(1)
    
    print(f"\n📂 发现 {len(image_file_list)} 个图像文件")
    
    # 处理图像
    all_results = []
    total_time = 0
    total_time_exclude_first = 0  # 排除首张图片的总时间
    success_count = 0
    first_image_time = 0  # 首张图片的处理时间
    
    start_batch_time = time.time()
    
    for idx, image_file in enumerate(image_file_list):
        print(f"\n[{idx+1}/{len(image_file_list)}]", end="")
        
        result, elapsed = process_single_image(ocr, image_file, args.output_dir, args)
        
        if result:
            all_results.append(result)
            success_count += 1
        
        total_time += elapsed
        
        # 记录首张图片时间，从第二张开始累计排除首张的时间
        if idx == 0:
            first_image_time = elapsed
        else:
            total_time_exclude_first += elapsed
    
    batch_time = time.time() - start_batch_time
    
    # 保存文本结果
    if str_to_bool(args.save_txt_results) and all_results:
        results_file = os.path.join(args.output_dir, "ocr_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 文本结果已保存: {results_file}")
        
        # 同时保存简化的文本文件
        txt_file = os.path.join(args.output_dir, "ocr_results.txt")
        with open(txt_file, 'w', encoding='utf-8') as f:
            for result in all_results:
                filename = os.path.basename(result['image_path'])
                f.write(f"{filename}\n")
                for item in result['results']:
                    f.write(f"  {item['transcription']} (置信度: {item['confidence']:.3f})\n")
                f.write("\n")
        print(f"📄 简化文本结果已保存: {txt_file}")
    
    # 输出统计信息
    print("\n" + "=" * 60)
    print("📊 处理统计")
    print("=" * 60)
    print(f"总图像数: {len(image_file_list)}")
    print(f"成功处理: {success_count}")
    print(f"失败数量: {len(image_file_list) - success_count}")
    print(f"总耗时: {batch_time:.3f}s")
    
    # 计算平均耗时（排除首张图片的预热时间）
    if len(image_file_list) > 1:
        avg_time_exclude_first = total_time_exclude_first / (len(image_file_list) - 1)
        print(f"首张图片耗时: {first_image_time:.3f}s (包含模型预热)")
        print(f"平均耗时 (排除首张): {avg_time_exclude_first:.3f}s/图")
    else:
        print(f"平均耗时: {total_time/len(image_file_list):.3f}s/图")
    
    print(f"处理速度: {len(image_file_list)/batch_time:.2f}图/s")
    
    if str_to_bool(args.save_visual_results):
        print(f"🖼️  可视化结果目录: {args.output_dir}")
    
    if str_to_bool(args.save_txt_results):
        print(f"📄 文本结果目录: {args.output_dir}")
    
    print("\n✅ 处理完成!")


if __name__ == "__main__":
    main() 