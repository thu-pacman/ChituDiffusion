# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

_device_name = None


def get_device_name():
    """
    Get the name of the current device.
    
    Returns:
        str: The CUDA device name if available, otherwise "CPU".
    """
    global _device_name
    if _device_name is None:
        if torch.cuda.is_available():
            _device_name = torch.cuda.get_device_name()
        else:
            _device_name = "CPU"
    return _device_name


def is_nvidia():
    """
    Check if the current device is an NVIDIA GPU.
    
    Returns:
        bool: True if device is NVIDIA, False otherwise.
    """
    return "NVIDIA" in get_device_name()


def is_muxi():
    """
    Check if the current device is a Muxi device (4000, 4001, or MetaX).
    
    Returns:
        bool: True if device is Muxi, False otherwise.
    """
    MUXI_DEVICE_PATTERNS = ["4000", "4001", "MetaX"]
    device_name = get_device_name()
    return any(
        pattern.lower() in device_name.lower() for pattern in MUXI_DEVICE_PATTERNS
    )


def is_ascend():
    """
    Check if the current device is an Ascend NPU.
    
    Returns:
        bool: True if device is Ascend, False otherwise.
    """
    return "Ascend" in get_device_name()


def is_ascend_910b():
    """
    Check if the current device is an Ascend 910B NPU.
    
    Returns:
        bool: True if device is Ascend 910B, False otherwise.
    """
    return "910B" in get_device_name()


def has_native_fp8():
    """
    Check if the current device has native FP8 support.
    
    Native FP8 is supported on NVIDIA GPUs with compute capability >= 8.9.
    
    Returns:
        bool: True if device supports native FP8, False otherwise.
    """
    return is_nvidia() and torch.cuda.get_device_capability() >= (8, 9)


def is_hopper():
    """
    Check if the current device is an NVIDIA Hopper GPU (H20 or H100).
    
    Returns:
        bool: True if device is Hopper architecture, False otherwise.
    """
    HOPPER_DEVICE_PATTERNS = ["H20", "H100"]
    device_name = get_device_name()
    return any(pattern in device_name for pattern in HOPPER_DEVICE_PATTERNS)


def is_blackwell():
    """
    Check if the current device is an NVIDIA Blackwell GPU (5090, B200, or B100).
    
    Returns:
        bool: True if device is Blackwell architecture, False otherwise.
    """
    BLACKWELL_DEVICE_PATTERNS = ["5090", "B200", "B100"]
    device_name = get_device_name()
    return any(pattern in device_name for pattern in BLACKWELL_DEVICE_PATTERNS)


def is_hygon():
    """
    Check if the current device is a Hygon device (BW).
    
    Returns:
        bool: True if device is Hygon, False otherwise.
    """
    HYGON_DEVICE_PATTERNS = ["BW"]
    device_name = get_device_name()
    return any(pattern in device_name for pattern in HYGON_DEVICE_PATTERNS)
