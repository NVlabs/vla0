# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].

from typing import List, Optional

from pydantic import BaseModel


class So100Base64DataModel(BaseModel):
    base64_rgb: List[str]
    state: List[float]  # (6)
    instr: Optional[str] = None
