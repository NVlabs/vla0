# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].

# Stores all the constants used in the project

THREE_P1 = "3p1"
THREE_P2 = "3p2"
THREE_P3 = "3p3"
WRIST_LEFT1 = "wrist_left1"
WRIST_RIGHT1 = "wrist_right1"
CAMERA_NAMES = [THREE_P1, THREE_P2, THREE_P3, WRIST_LEFT1, WRIST_RIGHT1]
REQUIRED_KEYS_ALL = [
    "rgb",
    "instr",
]
REQUIRED_KEYS_3D_COMPATIBLE = [
    "depth",
    "cam_int",
    "cam_ext",
]
REQUIRED_KEYS_EE_COMPATIBLE = [
    "ee_pos",
    "ee_rot",
    "ee_gri",
    "out_ee_pos",
    "out_ee_rot",
    "out_ee_gri",
]
REQUIRED_KEYS_ORIGINAL_ACTION = ["ori_act", "out_ori_act"]
REQUIRED_KEYS_PROPRIO = ["proprio"]

# Name of all supported datasets
LEROBOT = "lerobot"

# All supported unifiers
IMAGE = "image"
