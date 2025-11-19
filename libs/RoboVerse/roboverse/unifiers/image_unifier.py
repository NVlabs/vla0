# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].

import numpy as np
import roboverse.constants as c
import torch
from einops import rearrange
from roboverse.utils.stats_utils import get_unifier_stats
from roboverse.utils.unifier_utils import get_datasets, remove_keys
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms.functional import center_crop, crop

from rv_train.utils.train_utils import ForkedPdb as debug  # noqa


def image_unifier_transform(cfg, sample, sample_cam_list, eval=False):
    """
    All the operations that happen in transforming a sample from indiviusal
    dataset format to Image unifier format.

    Currently, it just loads the required keys. In future, it can be extended
    to do more operations like resizing the image to a fixed size

    :param cfg: Config object
    :param sample: Sample from individual dataset or simulation environment
    :param sample_cam_list: List of cameras in the sample. It defines the order
    of rgb images in the sample
    :param eval: Whether this is called during evaluation. During evaluation,
    we disable augmentations like instead of random crops, we do center crops.
    """
    cam_list = cfg.IMAGE.cam_list
    sample["rgb"] = np.stack(
        [sample["rgb"][:, sample_cam_list.index(cam)] for cam in cam_list],
        axis=1,
    )

    # in case of any augmentation or resizing, we convert the rgb image to tensor first
    if (
        (cfg.IMAGE.img_size != 0)
        or (cfg.IMAGE.crop_img < 1.0)
        or (cfg.IMAGE.brightness_aug > 0.0)
        or (cfg.IMAGE.contrast_aug > 0.0)
        or (cfg.IMAGE.saturation_aug > 0.0)
        or (cfg.IMAGE.hue_aug > 0.0)
    ):
        _rgb = torch.from_numpy(sample["rgb"])
        _rgb = rearrange(_rgb, "... h w c -> ... c h w")

    if cfg.IMAGE.crop_img < 1.0:
        crop_height = int(_rgb.shape[-2] * cfg.IMAGE.crop_img)
        crop_width = int(_rgb.shape[-1] * cfg.IMAGE.crop_img)

        original_shape = _rgb.shape
        # Get the product of all leading dimensions
        leading_dims_size = np.prod(original_shape[:-3]).astype(int)
        # Reshape to flatten all leading dimensions into one
        _rgb = _rgb.reshape(leading_dims_size, *original_shape[-3:])

        if eval:
            # do center crop - works on batches automatically
            _rgb = center_crop(_rgb, (crop_height, crop_width))
        else:
            # Create output tensor for storing results
            cropped_rgb = torch.zeros(
                leading_dims_size, _rgb.shape[1], crop_height, crop_width
            )

            # Apply random crop individually to each item
            for i in range(leading_dims_size):
                top = torch.randint(0, _rgb.shape[-2] - crop_height + 1, (1,)).item()
                left = torch.randint(0, _rgb.shape[-1] - crop_width + 1, (1,)).item()
                cropped_rgb[i] = crop(_rgb[i], top, left, crop_height, crop_width)

            _rgb = cropped_rgb
        _rgb = _rgb.reshape(*original_shape[:-2], crop_height, crop_width)

    if (cfg.IMAGE.img_size != 0) or (
        (cfg.IMAGE.img_size == 0) and (cfg.IMAGE.crop_img < 1.0)
    ):
        current_shape = _rgb.shape
        # Get the product of all leading dimensions
        leading_dims_size = np.prod(current_shape[:-3]).astype(int)
        # Reshape to flatten all leading dimensions into one
        _rgb = _rgb.reshape(leading_dims_size, *current_shape[-3:])

        if cfg.IMAGE.img_size != 0:
            if (_rgb.shape[-2] != cfg.IMAGE.img_size) or (
                _rgb.shape[-1] != cfg.IMAGE.img_size
            ):
                _rgb = F.interpolate(
                    _rgb,
                    size=(cfg.IMAGE.img_size, cfg.IMAGE.img_size),
                    mode="bilinear",
                    align_corners=False,
                )
            # Reshape back to original dimensions
            _rgb = _rgb.reshape(
                *current_shape[:-2], cfg.IMAGE.img_size, cfg.IMAGE.img_size
            )
        elif (cfg.IMAGE.img_size == 0) and (cfg.IMAGE.crop_img < 1.0):
            # resize the _rgb image to original h and w
            if (_rgb.shape[-2] != original_shape[-2]) or (
                _rgb.shape[-1] != original_shape[-1]
            ):
                _rgb = F.interpolate(
                    _rgb,
                    size=(original_shape[-2], original_shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )

            # Reshape back to original dimensions
            _rgb = _rgb.reshape(*original_shape)

    augment_needed = (not eval) and any(
        [
            cfg.IMAGE.brightness_aug > 0.0,
            cfg.IMAGE.contrast_aug > 0.0,
            cfg.IMAGE.saturation_aug > 0.0,
            cfg.IMAGE.hue_aug > 0.0,
        ]
    )
    if augment_needed:
        _rgb = _rgb.float() / 255.0  # for ColorJitter, we need to normalize to [0, 1]
        _rgb = ColorJitter(
            brightness=cfg.IMAGE.brightness_aug,
            contrast=cfg.IMAGE.contrast_aug,
            saturation=cfg.IMAGE.saturation_aug,
            hue=cfg.IMAGE.hue_aug,
        )(_rgb)
        _rgb = (_rgb * 255.0).to(torch.uint8)

    # in case of any augmentation or resizing, we convert back to numpy
    if (
        (cfg.IMAGE.img_size != 0)
        or (cfg.IMAGE.crop_img < 1.0)
        or (cfg.IMAGE.brightness_aug > 0.0)
        or (cfg.IMAGE.contrast_aug > 0.0)
        or (cfg.IMAGE.saturation_aug > 0.0)
        or (cfg.IMAGE.hue_aug > 0.0)
    ):
        _rgb = rearrange(_rgb, "... c h w -> ... h w c")
        sample["rgb"] = _rgb.numpy()

    return sample


class Image_Unifier(Dataset):
    """
    Unifier that resizes image to a fixed size. Deletes unnessary keys.

    Currently, it just loads the required keys. In future, it can be extended
    to do more operations like resizing the image to a fixed size, data augmentation, etc.
    """

    def __init__(self, cfg):
        assert cfg.unifier == c.IMAGE, cfg.unifier
        self.cfg = cfg
        self.datasets = get_datasets(cfg)
        self.cam_list = cfg.IMAGE.cam_list
        for name, dataset in self.datasets.items():
            assert set(self.cam_list) <= set(
                dataset.cam_list
            ), f"{name} does not have all cameras"
            if cfg.IMAGE.return_ee:
                assert dataset.ee_compatible, f"{name} is not EE compatible"
            if cfg.IMAGE.return_ori_act:
                assert (
                    dataset.has_original_action
                ), f"{name} does not have original action"
            if cfg.IMAGE.return_proprio:
                assert dataset.has_proprio, f"{name} does not have proprioception"
        self.keys_to_remove = c.REQUIRED_KEYS_3D_COMPATIBLE
        if not cfg.IMAGE.return_ee:
            self.keys_to_remove.extend(c.REQUIRED_KEYS_EE_COMPATIBLE)
        if not cfg.IMAGE.return_ori_act:
            self.keys_to_remove.extend(c.REQUIRED_KEYS_ORIGINAL_ACTION)
        if not cfg.IMAGE.return_proprio:
            self.keys_to_remove.extend(c.REQUIRED_KEYS_PROPRIO)

        # calculate stats for the output keys
        stat_keys = []
        if cfg.IMAGE.return_ori_act:
            stat_keys.append("out_ori_act")
        if cfg.IMAGE.return_ee:
            stat_keys.extend(["out_ee_pos", "out_ee_rot", "out_ee_gri"])

        self.stats = get_unifier_stats(self, cfg, stat_keys)
        print(f"Stats: {self.stats}")

    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets.values()])

    def __getitem__(self, idx):
        for task, dataset in self.datasets.items():
            if idx < len(dataset):
                sample = dataset[idx]
                break
            else:
                idx -= len(dataset)

        sample = remove_keys(sample, self.keys_to_remove)
        sample = image_unifier_transform(self.cfg, sample, dataset.cam_list)
        return sample


if __name__ == "__main__":
    from roboverse.configs import get_cfg_defaults

    config = get_cfg_defaults()

    config.unifier = c.IMAGE
    config.history = 5
    config.horizon = 10
    config.IMAGE.return_ori_act = True

    print(config)

    unifier = Image_Unifier(config)
    sample = unifier[0]
    print(sample.keys())
    for key, value in sample.items():
        if not isinstance(value, str):
            print(key, value.shape)
        else:
            print(key, value)
