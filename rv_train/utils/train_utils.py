# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].

import os
import pdb
import subprocess
import sys
from collections.abc import MutableMapping
from time import sleep

import tbparse
import tensorboardX
import torch
import torch.distributed as dist


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class TensorboardManager:
    def __init__(self, path):
        self.path = path
        self.writer = tensorboardX.SummaryWriter(path)

    def update(self, split, step, vals):
        for k, v in vals.items():
            self.writer.add_scalar("%s_%s" % (split, k), v, step)

    def get(self, split, step, key):
        """
        Get the value of a specific key from the tensorboard
        Args:
            split (str): The split to get the value from.
            step (int): The step to get the value from.
            key (str): The key to get the value from.
        Returns:
            The value of the key, or None if not found.
        """
        self.writer.flush()  # Ensure data is written to disk
        try:
            reader = tbparse.SummaryReader(self.path)
        except Exception as e:
            print(f"Error reading tensorboard file {self.path}: {e}")
            # sleep for 10 seconds and try again
            sleep(10)
            try:
                reader = tbparse.SummaryReader(self.path)
            except Exception as e:
                print(f"Error reading tensorboard file {self.path}: {e}")
                return None

        scalar_df = reader.scalars

        if scalar_df.empty:
            # print(f"No scalar data found in {self.path}") # Optional: for debugging
            return None

        target_tag = f"{split}_{key}"

        # Filter by tag and step
        # Ensure 'step' in scalar_df is of the same type as the 'step' argument (e.g., int)
        # tbparse typically loads 'step' as int64, which should be fine with int comparison.
        filtered_df = scalar_df[
            (scalar_df["tag"] == target_tag) & (scalar_df["step"] == step)
        ]

        if filtered_df.empty:
            # Optional: for debugging
            # print(f"No data found for tag '{target_tag}' at step {step}")
            # print("Available tags:", scalar_df['tag'].unique())
            # print(f"Available steps for tag '{target_tag}':", scalar_df[scalar_df['tag'] == target_tag]['step'].unique())
            return None

        # Assuming there's only one value for a given tag and step
        # .item() is preferred for extracting a single value from a Series/DataFrame cell
        try:
            value = filtered_df["value"].item()
        except ValueError:
            # This can happen if more than one row matches, which shouldn't be the case
            # if (tag, step) is unique. Or if the filtered_df is unexpectedly shaped.
            # print(f"Warning: Multiple values found for tag '{target_tag}' at step {step}. Returning the first one.") # Optional
            value = filtered_df["value"].iloc[-1]  # Fallback to iloc[-1]

        return value

    def close(self):
        self.writer.flush()
        self.writer.close()


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


# source: https://github.com/pytorch/pytorch/issues/105045#issue-1800357420
def argmax(input, dims):
    with torch.no_grad():
        try:
            assert isinstance(dims, list)
            for dim in dims:
                assert dim > 0
            # add small random noise to break ties
            input = input + (torch.randn_like(input) * 1e-6)
            dims.sort()
            max_vals = torch.amax(input, dims)
            max_val_shape = max_vals.shape
            for dim in dims:
                max_vals = max_vals.unsqueeze(dim)
            non_zero = torch.nonzero(torch.Tensor(input == max_vals))
            non_zero = non_zero[:, dims]
            non_zero = non_zero.view(*max_val_shape, len(dims))
        except RuntimeError as e:
            print(e)
            non_zero = 0

        return non_zero


class PerfTrackVal:
    """
    Records epoch wise performance for validation
    """

    def __init__(self, cfg):
        self.no_ops = cfg.EXP_EXTRA.no_track
        if self.no_ops:
            print("Not tracking performance")
            return

        self.all_pos = []

    def update(self, data_batch, out):
        if self.no_ops:
            return

        with torch.no_grad():
            pass
            # TODO: add code
            # gt_pos = data_batch['act_pos']
            # pred_pos = out['pred_pos']
            # rmse = np.sqrt(((gt_pos - pred_pos) ** 2).sum(-1).float().mean().item())
            # self.all_pos.append(rmse)

    def agg(self):
        if self.no_ops:
            return {}

        perf = {}
        perf["pos_rmse"] = sum(self.all_pos) / len(self.all_pos)
        # resetting
        self.all_pos = []

        return perf

    @staticmethod
    def get_correct_list(logit, label):
        label = label.to(logit.device)
        pred_class = logit.argmax(axis=1)
        return (label == pred_class).to("cpu").tolist()

    @staticmethod
    def get_avg_list(all_list):
        for x in all_list:
            assert isinstance(x, bool)
        return sum(all_list) / len(all_list)


class PerfTrackTrain(PerfTrackVal):
    """
    Records epoch wise performance during training
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # add a list to track loss
        self.all_loss = []

    def update_loss(self, loss):
        self.all_loss.append(loss.item())

    def agg_loss(self):
        # print(self.all_loss)
        return sum(self.all_loss) / len(self.all_loss)

    def update_all(self, data_batch, out, loss):
        self.update(data_batch, out)
        self.update_loss(loss)


# source: https://stackoverflow.com/questions/2363731/append-new-row-to-old-csv-file-python
def flatten_dict(d, parent_key="", sep="_", use_short_name=True):
    items = []
    for k, v in d.items():
        if use_short_name:
            k, v = short_name(k), short_name(v)
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def short_name(cfg_opts):
    SHORT_FORMS = {
        "EXP_EXTRA": "EXE",
        "True": "T",
        "False": "F",
        "DATALOADER": "DL",
        "OPTIMIZER": "OPT",
        "adamw": "AW",
        "METAWORLD": "MW",
        "QWEN": "QW",
        "ROBOVERSE": "RV",
        "RoboVerse": "RV",
        "roboverse": "RV",
        "auto_resume": "AR",
        "img_size": "IM_S",
        "libs/RoboVerse/roboverse/configs/": "RV_CFG",
        "relaunch_on_loss_increase": "RL",
        "relaunch_on_loss_increase_factor": "RL_F",
        "original_action_dim": "OAD",
        "num_epochs": "NE",
        "TRAIN": "TR",
        "EXP_EXTRA": "EXE",
        "MODEL": "MD",
        "LIBERO": "LB",
        "metaworld": "mw",
        "num_bins_actions": "NBA",
        "batch_size": "BS",
        "lora_rank": "LOR",
        "use_lora": "UL",
        "use_qlora": "QL",
        "use_flash_attention_2": "FA2",
        "OPTIMIZER": "OPT",
        "LR_SCHED": "LRS",
        "cosine_anneal": "CA",
        "adam_bnb": "AB",
        "metaworld_XXXX_expert_T_5_40_YYYY_168": "MW_168_40",
        "mw_XXXX_expert_T_5_40_YYYY_168": "MW_168_40",
        "tasks": "T",
        "zarr_path_format": "ZPF",
        "rgb_img_size": "RGB_IM_S",
        "IMAGE": "IM",
        "clip_grad_norm": "CGN",
        "brightness_aug": "BA",
        "contrast_aug": "CA",
        "saturation_aug": "SA",
        "hue_aug": "HA",
        "action_mask_aug": "AM",
        "action_mask_aug_per": "AM_P",
        "attention_dropout": "AD",
        "use_ema": "EM",
        "ema_decay": "EM_D",
        "EMA_DECAY": "EM_D",
        "USE_EMA": "UE",
        "horizon": "H",
        "crop_img": "CI",
        "LEROBOT_LIBERO": "LB",
        "per_goal": "PG",
        "per_object": "PO",
        "per_spatial": "PS",
        "per_long": "PL",
        "action_space_ver": "ASV",
        "Qwen2.5-VL-3B-Instruct": "3B",
        "Qwen2.5-VL-7B-Instruct": "7B",
        "Qwen": "QW",
        "grad_checkpoint": "GC",
        "adamw_bnb_fp8": "ABF8",
        "LEROBOT": "LE",
        "lerobot_libero": "LL",
        "libero": "LB",
        "observation": "OBS",
        "images": "IMGS",
        "right": "R",
        "remove_noop_actions": "RNO",
        "action": "ACT",
        "num_cam": "NC",
        "cam_list": "CL",
        "left": "L",
        "tiled_rgb_imgs": "TR",
        "cfg_path_libs": "CPL",
        "configs": "CO",
        "img_real_aug": "IRA",
    }
    cfg_opts = cfg_opts.replace(" ", "_")
    cfg_opts = cfg_opts.replace("/", "_")
    cfg_opts = cfg_opts.replace("[", "")
    cfg_opts = cfg_opts.replace("]", "")
    cfg_opts = cfg_opts.replace("..", "")
    for a, b in SHORT_FORMS.items():
        cfg_opts = cfg_opts.replace(a, b)

    return cfg_opts


# source:
# https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/3
def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in GB.
    """
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        encoding="utf-8",
    )
    # Convert lines into a dictionary
    gpu_memory = [(int(x) / 1024.0) for x in result.strip().split("\n")]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    gpu_memory_map = {(f"GPU {x}"): f"{y:.2} GB" for x, y in gpu_memory_map.items()}
    return gpu_memory_map


if __name__ == "__main__":
    tb = TensorboardManager(path="./runs/test")
    tb.update("train", 1, {"loss": 1.0, "lr": 0.01})
    tb.update("train", 2, {"loss": 1.1, "lr": 0.02})
    tb.update("train", 3, {"loss": 1.2, "lr": 0.03})
    tb.update("train", 4, {"loss": 1.3, "lr": 0.04})
    tb.update("train", 4, {"loss": 1.3, "lr": 0.045})
    tb.update("train", 5, {"loss": 1.4, "lr": 0.05})
    tb.update("train", 6, {"loss": 1.5, "lr": 0.06})
    tb.update("train", 7, {"loss": 1.6, "lr": 0.07})
    tb.update("train", 8, {"loss": 1.7, "lr": 0.08})
    print(tb.get("train", 1, "loss"))
    print(tb.get("train", 4, "lr"))
    tb.close()
