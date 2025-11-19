# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].

from yacs.config import CfgNode as CN

import rv_train.constants as C

_C = CN()
# ----------------------------------------------------------------------------
# EXPERIMENT
# ----------------------------------------------------------------------------
_C.EXP = CN()
_C.EXP.EXP_ID = ""
_C.EXP.SEED = 0  # no longer used, kept for backward compatibility
_C.EXP.DATASET = "roboverse"  # roboverse
_C.EXP.MODEL = "qwen"
_C.EXP.OPTIMIZER = "adam"  # adam, adamw, adamw_bnb_fp8, adamw_bnb
_C.EXP.LR_SCHED = "none"  # cosine_anneal, none
_C.EXP.AMP = False  # whether to use automatic mixed precision training
# ----------------------------------------------------------------------------
# LOSS
# ----------------------------------------------------------------------------
_C.EXP.LOSS = CN()
# ----------------------------------------------------------------------------
# Extra Experiment Parameters
# ----------------------------------------------------------------------------
_C.EXP_EXTRA = CN()
_C.EXP_EXTRA.no_val = True
_C.EXP_EXTRA.no_test = True
_C.EXP_EXTRA.no_track = True
_C.EXP_EXTRA.val_eval_freq = 1
_C.EXP_EXTRA.test_eval_freq = 1
_C.EXP_EXTRA.save_ckp = 20
_C.EXP_EXTRA.save_last_ckpt = True
# ----------------------------------------------------------------------------
# TRAINING DETAILS (contains things common across the training, optimizer,
# lr_scheduler etc)
# ----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.num_epochs = 100
_C.TRAIN.l2 = 0.0
_C.TRAIN.lr = 1e-4
_C.TRAIN.clip_grad_norm = 1.0  # 0 means no clipping
# ----------------------------------------------------------------------------
# TRAINING SCHEDULER
# ----------------------------------------------------------------------------
_C.LR_SCHED = CN()
_C.LR_SCHED.lr_decay_factor = 0.5
_C.LR_SCHED.lr_patience = 4
_C.LR_SCHED.lr_clip = 1e-8
# ----------------------------------------------------------------------------
# DATALOADER (contains things common across the datasets)
# ----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.batch_size = 16
_C.DATALOADER.num_workers = 1
# ----------------------------------------------------------------------------
# METAWORLD
# ----------------------------------------------------------------------------
_C.DATALOADER.ROBOVERSE = CN()
_C.DATALOADER.ROBOVERSE.cfg_path = "libs/RoboVerse/roboverse/configs/test.yaml"
_C.DATALOADER.ROBOVERSE.cfg_opts = ""
# ----------------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------------
_C.MODEL = CN()
# ----------------------------------------------------------------------------
# QWEN
# ----------------------------------------------------------------------------
_C.MODEL.QWEN = CN()
_C.MODEL.QWEN.action_type = C.ORIGINAL
_C.MODEL.QWEN.original_action_dim = 4
_C.MODEL.QWEN.horizon = 8
_C.MODEL.QWEN.history = 1
_C.MODEL.QWEN.qwen_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
_C.MODEL.QWEN.use_lora = True
_C.MODEL.QWEN.use_qlora = True  # if use_lora is False, then this does not matter
_C.MODEL.QWEN.num_cam = 1
_C.MODEL.QWEN.lora_config = "default"
_C.MODEL.QWEN.lora_rank = 8
_C.MODEL.QWEN.rgb_input = True
_C.MODEL.QWEN.rgb_img_size = (84, 84)
_C.MODEL.QWEN.add_vision_id = False
_C.MODEL.QWEN.tiled_rgb_imgs = False
_C.MODEL.QWEN.num_bins_actions = 1000
_C.MODEL.QWEN.use_flash_attention_2 = False
_C.MODEL.QWEN.action_mask_aug_per = 0.1
_C.MODEL.QWEN.attention_dropout = 0.0


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
