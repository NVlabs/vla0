# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].

import roboverse.constants as c
from roboverse.configs import get_cfg_defaults


def get_cfg(cfg_path, cfg_opts=""):
    cfg = get_cfg_defaults()
    if cfg_path != "":
        cfg.merge_from_file(cfg_path)

    if cfg_opts != "":
        cfg.merge_from_list(cfg_opts.split(":"))

    cfg.freeze()
    return cfg


def get_unified_dataset(cfg_path, cfg_opts=""):
    """
    Get the unified dataset based on the config file.
    """
    cfg = get_cfg(cfg_path, cfg_opts)

    if cfg.unifier == c.IMAGE:
        from roboverse.unifiers.image_unifier import Image_Unifier

        return Image_Unifier(cfg)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="./roboverse/configs/img_libero.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()
    cfg_path = args.cfg_path
    dataset = get_unified_dataset(cfg_path)
    breakpoint()
    print("done")
