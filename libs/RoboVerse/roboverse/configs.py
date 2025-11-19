# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].

import roboverse.constants as c
from yacs.config import CfgNode as CN

_C = CN()

# ----------------------------------------------------------------------------
# Which unifier to select and list of datasets to use in that unifier
# ----------------------------------------------------------------------------
_C.unifier = c.IMAGE
_C.datasets = [c.LEROBOT]

# Following configs are common for all datasets
_C.history = 5
_C.horizon = 5

# Following configs are common among some datasets, but not all
# these might be helpful to avoid loading the entire dataset
_C.load_depth = False
_C.load_original_action = False

# ----------------------------------------------------------------------------
# IMAGE Unifier specific arguments
# ----------------------------------------------------------------------------
_C.IMAGE = CN()
_C.IMAGE.cam_list = [
    c.THREE_P1
]  # list of camera names to use, all datasets must have these cameras
_C.IMAGE.return_ee = False
_C.IMAGE.return_ori_act = False
_C.IMAGE.return_proprio = False
_C.IMAGE.crop_img = 1.0  # 0.0 to 1.0, denotes the fraction of the image to crop. The height and width of the image is cropped to this fraction. Same as pixel shift augmentation in robomimic. if eval is true in image_unifier_transform, we do center crop.
_C.IMAGE.img_size = 0  # if 0, then the image is not resized, otherwise the image is resized to this size. When img_size in 0 and crop_img < 1.0, the image is resized to the original height and width.
_C.IMAGE.brightness_aug = 0.0  # brightness augmentation, 0.0 to 1.0, denotes the amount of brightness augmentation.
_C.IMAGE.contrast_aug = 0.0  # contrast augmentation, 0.0 to 1.0, denotes the amount of contrast augmentation.
_C.IMAGE.saturation_aug = 0.0  # saturation augmentation, 0.0 to 1.0, denotes the amount of saturation augmentation.
_C.IMAGE.hue_aug = (
    0.0  # hue augmentation, 0.0 to 1.0, denotes the amount of hue augmentation.
)

# ----------------------------------------------------------------------------
# LEROBOT Dataset specific arguments
# ----------------------------------------------------------------------------
_C.LEROBOT = CN()
_C.LEROBOT.repo_id = "danaaubakirova/koch_test"
_C.LEROBOT.le_cam_list = None
_C.LEROBOT.rv_cam_list = None
_C.LEROBOT.action_key = "action"
_C.LEROBOT.state_key = "observation.state"
_C.LEROBOT.convert_ori_act_to_delta_act = False
_C.LEROBOT.remove_noop_actions = False
_C.LEROBOT.fps = -1


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
