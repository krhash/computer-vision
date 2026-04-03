# src/utils/device_utils.py
# Project 5: Recognition using Deep Networks
# Author: Krushna Sanjay Sharma
# Description: Utility for resolving the best available compute device using
#              the torch.accelerator API (PyTorch >= 2.4). Falls back to CPU
#              if no accelerator is available.
#              Used by all task files and classes that require a torch.device.

import torch


def get_device() -> torch.device:
    """
    Resolves the best available compute device via torch.accelerator.

    Priority order (handled automatically by PyTorch):
        1. CUDA  — NVIDIA GPU
        2. MPS   — Apple Silicon GPU
        3. CPU   — fallback

    Returns:
        torch.device: The selected device (e.g. device('cuda'), device('mps'),
                      or device('cpu')).
    """
    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    print(f"  [Device] Using: {device}")
    return device