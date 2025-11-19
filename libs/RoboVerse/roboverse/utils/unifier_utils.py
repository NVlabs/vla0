# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].

from collections import OrderedDict

import roboverse.constants as c


def get_datasets(cfg):
    """
    Get a dictionary of datasets from the config.
    """
    datasets = OrderedDict()
    for dataset in cfg.datasets:
        if dataset == c.LEROBOT:
            from roboverse.datasets import LeRobotRV

            args = dict(cfg.LEROBOT)
            args.update(
                {
                    "history": cfg.history,
                    "horizon": cfg.horizon,
                }
            )
            curr_dataset = LeRobotRV(**args)
        else:
            raise NotImplementedError(dataset)

        datasets[dataset] = curr_dataset
    return datasets


def remove_keys(sample, keys_to_remove):
    """
    Remove keys from the sample that are not needed.
    """
    for key in keys_to_remove:
        if key in sample:
            del sample[key]
    return sample
