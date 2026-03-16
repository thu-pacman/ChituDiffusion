#!/usr/bin/env python3
# Run this script to initialize VBench models.
# Use origin download: python download_vbench_models.py
# Use hf-mirror: python download_vbench_models.py --mirror


import os
import subprocess
from pathlib import Path
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置缓存目录
CACHE_DIR = os.path.expanduser('~/.cache/vbench')

def get_models(use_mirror=False):
    """根据是否使用镜像返回模型配置"""
    hf_base = "hf-mirror.com" if use_mirror else "huggingface.co"
    
    return {
        'clip': {
            'ViT-B-32': {
                'url': 'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt',
                'path': 'clip_model/ViT-B-32.pt'
            },
            'ViT-L-14': {
                'url': 'https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt',
                'path': 'clip_model/ViT-L-14.pt'
            }
        },
        'umt': {
            'action_recognition': {
                'url': f'https://{hf_base}/OpenGVLab/VBench_Used_Models/resolve/main/l16_ptk710_ftk710_ftk400_f16_res224.pth',
                'path': 'umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth'
            }
        },
        'amt': {
            'motion_smoothness': {
                'url': f'https://{hf_base}/lalala125/AMT/resolve/main/amt-s.pth',
                'path': 'amt_model/amt-s.pth'
            }
        },
        'raft': {
            'optical_flow': {
                'url': 'https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip',
                'path': 'raft_model/models.zip',
                'is_zip': True
            }
        },
        'dino': {
            'model': {
                'url': 'https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth',
                'path': 'dino_model/dino_vitbase16_pretrain.pth'
            },
            'repo': {
                'url': 'https://github.com/facebookresearch/dino',
                'path': 'dino_model/facebookresearch_dino_main',
                'is_git': True
            }
        },
        'musiq': {
            'spaq': {
                'url': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth',
                'path': 'pyiqa_model/musiq_spaq_ckpt-358bb6af.pth'
            }
        },
        'grit': {
            'object_detection': {
                'url': f'https://{hf_base}/OpenGVLab/VBench_Used_Models/resolve/main/grit_b_densecap_objectdet.pth',
                'path': 'grit_model/grit_b_densecap_objectdet.pth'
            }
        },
        'tag2text': {
            'swin': {
                'url': f'https://{hf_base}/spaces/xinyu1205/recognize-anything/resolve/main/tag2text_swin_14m.pth',
                'path': 'caption_model/tag2text_swin_14m.pth'
            }
        },
        'viclip': {
            'internvid': {
                'url': f'https://{hf_base}/OpenGVLab/VBench_Used_Models/resolve/main/ViClip-InternVid-10M-FLT.pth',
                'path': 'ViCLIP/ViClip-InternVid-10M-FLT.pth'
            }
        }
    }

def download_file(url, path):
    """下载文件到指定路径"""
    try:
        subprocess.run(['wget', url, '-O', path], check=True)
        logger.info(f"Successfully downloaded {url}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download {url}: {e}")
        return False

def clone_repository(url, path):
    """克隆git仓库"""
    try:
        subprocess.run(['git', 'clone', url, path], check=True)
        logger.info(f"Successfully cloned {url}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone {url}: {e}")
        return False

def handle_zip(zip_path, extract_dir):
    """处理zip文件"""
    try:
        subprocess.run(['unzip', '-d', extract_dir, zip_path], check=True)
        os.remove(zip_path)
        logger.info(f"Successfully extracted {zip_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to extract {zip_path}: {e}")
        return False

def download_models(use_mirror=False):
    """下载所有模型"""
    # 创建主缓存目录
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    MODELS = get_models(use_mirror)
    
    for model_type, model_info in MODELS.items():
        logger.info(f"Processing {model_type} models...")
        
        for model_name, details in model_info.items():
            full_path = os.path.join(CACHE_DIR, details['path'])
            
            # 如果文件已存在，跳过
            if os.path.exists(full_path) and not details.get('is_zip', False):
                logger.info(f"Skipping {model_name}, already exists at {full_path}")
                continue
                
            # 创建目标目录
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            if details.get('is_git', False):
                if not os.path.exists(full_path):
                    clone_repository(details['url'], full_path)
            elif details.get('is_zip', False):
                if not os.path.exists(os.path.dirname(full_path)):
                    download_file(details['url'], full_path)
                    handle_zip(full_path, os.path.dirname(full_path))
            else:
                download_file(details['url'], full_path)

def main():
    parser = argparse.ArgumentParser(description='Download VBench models')
    parser.add_argument('--mirror', action='store_true', 
                      help='Use hf-mirror.com instead of huggingface.co')
    args = parser.parse_args()
    
    if args.mirror:
        logger.info("Using hf-mirror.com as HuggingFace mirror")
    
    download_models(args.mirror)

if __name__ == "__main__":
    main()