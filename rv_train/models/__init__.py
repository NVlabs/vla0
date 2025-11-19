# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].

try:
    from .qwen.model import QwenActor  # noqa F401
except ImportError as e:
    print(f"Qwen not found: {e}")
