# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].

import argparse
import gc
import os
import pickle as pkl
import pprint
import random
import shutil
from contextlib import redirect_stdout
from datetime import datetime
from time import time

import roboverse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tqdm
from torch import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from rv_train import models
from rv_train.configs import get_cfg_defaults
from rv_train.utils import train_utils as utils

DEVICE = ""

START_TIME = time()


def save_checkpoint(name, epoch, model, optimizer, lr_sched, cfg, log_dir):
    """
    Saves all information required for resuming training in the experiment
    folder.
    """
    # take care of DDP
    if isinstance(model, DDP):
        model_module = model.module
    else:
        model_module = model

    # take care of model saving for models that have save_pretrained method
    if hasattr(model_module, "save_pretrained"):
        model_state = None
        print("WARNING: model has save_pretrained method, not saving model state")
        model_module.save_pretrained(f"{log_dir}/model_{name}")
    else:
        model_state = model_module.state_dict()

    # Prepare checkpoint data
    checkpoint_data = {
        "cfg": vars(cfg),
        "epoch": epoch,
        "model_state": model_state,
        "optimizer_state": optimizer.state_dict(),
        "lr_sched_state": lr_sched.state_dict() if lr_sched is not None else None,
    }

    pth_path = f"{log_dir}/model_{name}.pth"
    torch.save(checkpoint_data, pth_path)

    # save the dataset stats
    if cfg.EXP.MODEL in ["qwen", "dp", "qwen_dp"]:
        with open(f"{log_dir}/dataset_stats.pkl", "wb") as f:
            pkl.dump(model_module.original_dataset_stats, f)

    print(f"Checkpoint saved to {pth_path}.")


def load_model(model, model_path, cfg):
    """
    Loads a pretrained model from a given path.
    :param model: model to load
    :param model_path: path to the pretrained model
    :param cfg: config object
    """
    print(f"Recovering model and checkpoint from {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # take care of DDP
    if isinstance(model, DDP):
        model_module = model.module
    else:
        model_module = model

    # take care of model loading for models that have load_pretrained method
    if hasattr(model_module, "from_pretrained"):
        print("WARNING: model has from_pretrained method")
        assert model_path[-4:] == ".pth"
        print(f"Loading from {model_path[:-4]}")
        model_module.from_pretrained(model_path[:-4])
    else:
        model_module.load_state_dict(checkpoint["model_state"])

    # load the dataset stats
    if cfg.EXP.MODEL in ["qwen", "dp", "qwen_dp"]:
        log_dir = "/".join(model_path.split("/")[:-1])
        with open(f"{log_dir}/dataset_stats.pkl", "rb") as f:
            original_dataset_stats = pkl.load(f)
            model_module.set_dataset_stats(original_dataset_stats)

    return model, checkpoint


def load_model_opt_sched(
    model,
    optimizer,
    lr_sched,
    model_path,
    cfg,
    to_load_model=True,
    only_load_model=False,
):
    """
    Loads a pretrained model from a given path.
    :param model: model to load
    :param optimizer: optimizer to load
    :param lr_sched: learning rate scheduler to load
    :param model_path: path to the pretrained model
    :param cfg: config object
    :param to_load_model: whether to load the model from the checkpoint or not
    """
    if to_load_model:
        model, checkpoint = load_model(model, model_path, cfg)
    else:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    if not only_load_model:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if lr_sched is not None:
            lr_sched.load_state_dict(checkpoint["lr_sched_state"])

    epoch = checkpoint["epoch"]

    # clean GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    return model, epoch, optimizer, lr_sched


def get_pretrained_model(model_path, device, torch_compile=False):
    """
    Loads a pretrained model from a given path.
    :param model_path: path to the pretrained model
    :param device: device to load the model on, supports only single GPU for now
    :return: model, cfg
    """
    model_folder = "/".join(model_path.split("/")[:-1])
    cfg_path = model_folder + "/config.yaml"
    cfg = get_cfg(cfg_path, cfg_opts="")

    model = get_model(
        cfg, calculate_dataset_stats=False
    )  # don't calculate dataset stats for pretrained model, its loaded from a checkpoint
    model.to(device)
    optimizer, lr_sched = get_optimizer(cfg, model, num_gpus=1)

    model, _, _, _ = load_model_opt_sched(
        model=model,
        optimizer=optimizer,
        lr_sched=lr_sched,
        model_path=model_path,
        cfg=cfg,
        only_load_model=True,
    )

    if torch_compile:
        print(
            "Compiling model with torch.compile, this will put the model in eval mode and may take a while..."
        )
        model.eval()
        model = torch.compile(model)
        if hasattr(model, "model"):
            if hasattr(model.model, "generate"):
                print("Compiling model.model.generate with torch.compile")
                model.model.generate = torch.compile(model.model.generate)

    return model, cfg


def get_cfg(cfg_path, cfg_opts):
    cfg = get_cfg_defaults()
    if cfg_path != "":
        cfg.merge_from_file(cfg_path)

    if cfg_opts != "":
        cfg.merge_from_list(cfg_opts.split(" "))
        cfg.EXP.EXP_ID += f"_{utils.short_name(cfg_opts)}"
    cfg.freeze()

    print(cfg)
    return cfg


def get_inp(cfg, data_batch):
    """
    Constructs the input for the model using the batched data.
    :param cfg: config object
    :param data_batch: contains the batched data provided by the dataloader
    """

    inp = data_batch
    return inp


def get_model(cfg, calculate_dataset_stats=True):
    """
    Returns model based on the config
    """
    if cfg.EXP.MODEL == "qwen":
        model = models.QwenActor(**cfg.MODEL.QWEN)
    else:
        assert False, f"Invalid model: {cfg.EXP.MODEL}"

    if calculate_dataset_stats and cfg.EXP.MODEL in ["qwen"]:
        temp_dataset = get_dataloader(split="train", cfg=cfg, get_dataset=True)
        model.set_dataset_stats(temp_dataset.stats)
        del temp_dataset

    return model


def default_batch_proc(data_batch, device):
    for x in data_batch:
        if isinstance(data_batch[x], dict):
            for y in data_batch[x]:
                data_batch[x][y] = data_batch[x][y].to(device).float()
        else:
            if isinstance(data_batch[x], torch.Tensor):
                data_batch[x] = data_batch[x].to(device).float()
            else:
                data_batch[x] = data_batch[x]
    return data_batch


def get_dataloader(split, cfg, get_dataset=False):
    """
    Returns dataloader based on the config and split
    :param get_dataset: whether to return the dataset or the dataloader
    """
    num_workers = cfg.DATALOADER.num_workers
    batch_size = cfg.DATALOADER.batch_size
    dataset_args = {"split": split}

    if cfg.EXP.DATASET == "roboverse":
        print("WARNING: split is ignored for roboverse dataset.")
        dataset_args = dict(**cfg.DATALOADER.ROBOVERSE)
        dataset = roboverse.get_unified_dataset(**dataset_args)
    else:
        raise NotImplementedError

    if "batch_proc" not in dir(dataset):
        dataset.batch_proc = default_batch_proc

    if get_dataset:
        return dataset
    else:
        return DataLoader(
            dataset,
            batch_size,
            num_workers=num_workers,
            shuffle=(split == "train"),
            drop_last=(split == "train"),
            pin_memory=(torch.cuda.is_available()) and (not num_workers),
            persistent_workers=(num_workers > 0),
        )


def check_grad(model, loss):
    bad_grad = False
    if loss.ne(loss).any():
        bad_grad = True
        print("WARNING: nan in the loss")
    else:
        for x in model.parameters():
            if x.grad is not None:
                if x.grad.ne(x.grad).any():
                    print("WARNING: nan in a gradient")
                    bad_grad = True
                    break
                if ((x.grad == float("inf")) | (x.grad == float("-inf"))).any():
                    print("WARNING: inf in a gradient")
                    bad_grad = True
                    break
    return bad_grad


def get_optimizer(cfg, model, num_gpus=1):
    """
    Returns optimizer and learning rate scheduler based on the config
    :param cfg: config object
    :param model: model to optimize
    :param num_gpus: number of GPUs to optimize the model on, required for scaling the learning rate
    """
    if cfg.EXP.OPTIMIZER == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.TRAIN.lr * num_gpus, weight_decay=cfg.TRAIN.l2
        )
    elif cfg.EXP.OPTIMIZER == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.TRAIN.lr * num_gpus, weight_decay=cfg.TRAIN.l2
        )
    elif cfg.EXP.OPTIMIZER == "adam_bnb":
        import bitsandbytes as bnb

        optimizer = bnb.optim.Adam(
            model.parameters(), lr=cfg.TRAIN.lr * num_gpus, weight_decay=cfg.TRAIN.l2
        )
    elif cfg.EXP.OPTIMIZER == "adamw_bnb":
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW(
            model.parameters(), lr=cfg.TRAIN.lr * num_gpus, weight_decay=cfg.TRAIN.l2
        )
    elif cfg.EXP.OPTIMIZER == "adamw_bnb_fp8":
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(
            model.parameters(), lr=cfg.TRAIN.lr * num_gpus, weight_decay=cfg.TRAIN.l2
        )
    else:
        raise NotImplementedError

    if cfg.EXP.LR_SCHED == "none":
        lr_sched = None
    elif cfg.EXP.LR_SCHED == "cosine_anneal":
        lr_sched = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.TRAIN.num_epochs, eta_min=cfg.LR_SCHED.lr_clip
        )
    else:
        raise NotImplementedError

    return optimizer, lr_sched


def train(
    cfg,
    loader,
    model,
    optimizer,
    device=0,
    check_grad_fn=False,  # Rename this parameter
    fn_check_time_limit_and_relaunch=None,
    rank=0,
):
    """
    Training for one epoch
    """

    model.train()
    perf = utils.PerfTrackTrain(cfg)

    time_for = 0
    time_bac = 0
    time_dl = 0
    time4 = time()
    for i, data_batch in tqdm.tqdm(enumerate(loader), dynamic_ncols=True):
        data_batch = loader.dataset.batch_proc(data_batch, device)
        inp = get_inp(cfg, data_batch)

        time1 = time()
        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.EXP.AMP):
            out = model(**inp, get_loss=True)
        loss = out["loss"]
        perf.update_all(data_batch=data_batch, out=out, loss=loss)

        time2 = time()
        optimizer.zero_grad()
        loss.backward()
        if cfg.TRAIN.clip_grad_norm != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.TRAIN.clip_grad_norm)

        if check_grad_fn and check_grad(model, loss):  # Use the renamed parameter
            print("WARNING: avoiding step as bad gradient")
        else:
            optimizer.step()

        time3 = time()
        time_dl += time1 - time4
        time_for += time2 - time1
        time_bac += time3 - time2
        time4 = time()

        if fn_check_time_limit_and_relaunch is not None:
            # checking every 300 batches ~ 5 minutes
            if (i + 1) % 300 == 0:
                fn_check_time_limit_and_relaunch(perf.agg_loss())

        # uncomment for intermediate printing
        # if i % 10 == 0:
        #     print(f"Iteration {i} time taken: {time_for:.2f}s, {time_bac:.2f}s, {time_dl:.2f}s")

    print(
        f"Avg_loss: {perf.agg_loss():.4f}, "
        f"Forward: {time_for:.2f}s, Backward: {time_bac:.2f}s, "
        f"Data Load: {time_dl:.2f}s, "
        f"Memory Usage: {utils.get_gpu_memory_map()}"
    )

    return perf.agg(), perf.agg_loss()


def print_model_stats(model):
    """Print model statistics including parameter counts."""
    # Get model module if using DDP
    model_module = model.module if isinstance(model, DDP) else model

    # Count total parameters
    total_params = sum(p.numel() for p in model_module.parameters())

    # Count trainable parameters
    trainable_params = sum(
        p.numel() for p in model_module.parameters() if p.requires_grad
    )

    # Count non-trainable parameters
    non_trainable_params = total_params - trainable_params

    print("=" * 50)
    print("Model Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print("=" * 50)


def get_log_dir(cfg, logdir_with_time=False):
    if logdir_with_time:
        log_dir = (
            f"./runs/{cfg.EXP.EXP_ID}/{str(datetime.now())[:-7].replace(' ', '-')}"
        )
    else:
        log_dir = f"./runs/{cfg.EXP.EXP_ID}"
    return log_dir


def entry_train(
    rank,
    cfg,
    logdir_with_time=False,
    resume=False,
    model_path="",
    devices=[0],
    port=12345,
):
    """
    Training and evaluating a network based on the specified config.
    """

    device = devices[rank]
    device = f"cuda:{device}"
    ddp = len(devices) > 1
    utils.setup(rank, world_size=len(devices), port=port)
    torch.cuda.set_device(device)
    if ddp:
        print(f"Running on rank {rank}")

    # random.seed(cfg.EXP.SEED + rank)
    # np.random.seed(cfg.EXP.SEED + rank)
    # torch.manual_seed(cfg.EXP.SEED + rank)

    loader_train = get_dataloader(split="train", cfg=cfg)
    model = get_model(cfg)
    model.to(device)

    to_load_model = True
    if (
        hasattr(model, "load_param_before_ddp")
        and model.load_param_before_ddp
        and resume
    ):
        to_load_model = False
        model, _ = load_model(model, model_path, cfg)
        model.to(device)

    if ddp:
        # Set find_unused_parameters=False when using gradient checkpointing
        # to avoid synchronization issues and deadlocks
        using_grad_checkpoint = False
        if cfg.EXP.MODEL in ["qwen", "qwen_dp"]:
            model_config = (
                cfg.MODEL.QWEN if cfg.EXP.MODEL == "qwen" else cfg.MODEL.QWEN_DP
            )
            using_grad_checkpoint = getattr(model_config, "grad_checkpoint", False)

        find_unused_params = not using_grad_checkpoint
        if rank == 0:
            print(
                f"DDP configuration: grad_checkpoint={using_grad_checkpoint}, find_unused_parameters={find_unused_params}"
            )
        model = DDP(
            model, device_ids=[device], find_unused_parameters=find_unused_params
        )
    if rank == 0:
        print(model)

    optimizer, lr_sched = get_optimizer(cfg, model, num_gpus=len(devices))
    if resume:
        model, old_epoch, optimizer, lr_sched = load_model_opt_sched(
            model=model,
            optimizer=optimizer,
            lr_sched=lr_sched,
            model_path=model_path,
            cfg=cfg,
            to_load_model=to_load_model,
        )
    else:
        assert model_path == "", model_path
        old_epoch = -1

    if rank == 0:
        print_model_stats(model)

    dist.barrier()

    if rank == 0:
        log_dir = get_log_dir(cfg, logdir_with_time)
        print(f"Log directory: {log_dir}")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        with open(f"{log_dir}/config.yaml", "w") as f:
            with redirect_stdout(f):
                print(cfg.dump())

        # tb is initialized only for rank 0
        tb = utils.TensorboardManager(log_dir)
    else:
        # log_dir and tb should not be used for any rank other than rank 0
        log_dir = ""
        tb = None

    for epoch in range(old_epoch + 1, cfg.TRAIN.num_epochs):
        fn_check_time_limit_and_relaunch = None

        # print epoch number
        if rank == 0:
            print(f"Training for epoch {epoch} / {cfg.TRAIN.num_epochs}")

        # train
        train_perf, train_loss = train(
            cfg=cfg,
            loader=loader_train,
            model=model,
            optimizer=optimizer,
            device=device,
            fn_check_time_limit_and_relaunch=fn_check_time_limit_and_relaunch,
            rank=rank,
        )

        # update tensorboard
        if rank == 0:
            _lr = (
                lr_sched.optimizer.param_groups[0]["lr"]
                if lr_sched
                else optimizer.param_groups[0]["lr"]
            )
            pprint.pprint(f"Performance: {train_perf}", width=80)
            tb.update("train", epoch, train_perf)
            tb.update(
                "train",
                epoch,
                {"loss": train_loss, "lr": _lr},
            )

        # save checkpoint
        if rank == 0:
            if not (cfg.EXP_EXTRA.save_ckp == 0) and (
                epoch % cfg.EXP_EXTRA.save_ckp == 0
            ):
                save_checkpoint(
                    f"{epoch}",
                    epoch,
                    model,
                    optimizer,
                    lr_sched,
                    cfg,
                    log_dir,
                )

            if cfg.EXP_EXTRA.save_last_ckpt:
                # change name of last checkpoint to second_last so that it is not overwritten by the new last checkpoint.
                # this second last checkpoint will be used to resume training if the training is relaunched because of loss increase
                if os.path.exists(log_dir + "/model_last.pth"):
                    # remove second last checkpoint if it exists
                    if os.path.exists(log_dir + "/model_second_last.pth"):
                        os.remove(log_dir + "/model_second_last.pth")
                    os.rename(
                        log_dir + "/model_last.pth", log_dir + "/model_second_last.pth"
                    )
                if os.path.exists(log_dir + "/model_last"):
                    # remove second last checkpoint if it exists
                    if os.path.exists(log_dir + "/model_second_last"):
                        shutil.rmtree(log_dir + "/model_second_last")
                    os.rename(log_dir + "/model_last", log_dir + "/model_second_last")
                save_checkpoint(
                    "last",
                    epoch,
                    model,
                    optimizer,
                    lr_sched,
                    cfg,
                    log_dir,
                )

        # update learning rate
        if cfg.EXP.LR_SCHED in ["none"]:
            print(f"Current lr: {optimizer.param_groups[0]['lr']}")
        elif cfg.EXP.LR_SCHED in ["cosine_anneal"]:
            lr_sched.step()
            print(f"Current lr: {lr_sched.optimizer.param_groups[0]['lr']}")
        else:
            raise NotImplementedError

    if rank == 0:
        print("Saving the final model")
        save_checkpoint(
            "final",
            cfg.TRAIN.num_epochs - 1,
            model,
            optimizer,
            lr_sched,
            cfg,
            log_dir,
        )

    if rank == 0:
        # close tensorboard
        tb.close()


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if DEVICE.type == "cpu":
        print("WARNING: Using CPU")

    parser = argparse.ArgumentParser()
    parser.set_defaults(entry=lambda cmd_args: parser.print_help())
    parser.add_argument("--entry", type=str, default="train")
    parser.add_argument("--exp-config", type=str, default="")
    parser.add_argument("--exp-cfg-opts", type=str, default="")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--logdir-with-time", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0")

    cmd_args = parser.parse_args()

    if cmd_args.entry == "train":
        assert (
            not cmd_args.logdir_with_time
        ), "Temporarily disable logdir_with_time as it is not handled properly when autoresuming and auto-relaunching with loss increase. It is fine for one time launch or manual relaunching."
        _cfg = get_cfg(cmd_args.exp_config, cmd_args.exp_cfg_opts)
        if cmd_args.resume:
            if cmd_args.model_path == "":
                print(
                    "WARNING: No model path provided, resuming from latest checkpoint"
                )
                log_dir = get_log_dir(_cfg, cmd_args.logdir_with_time)
                cmd_args.model_path = os.path.join(log_dir, "model_last.pth")
            print(f"Resuming from {cmd_args.model_path}")
        else:
            assert cmd_args.model_path == ""

        devices = cmd_args.devices.split(",")
        devices = [int(x) for x in devices]
        port = (random.randint(0, 3000) % 3000) + 27000
        mp.spawn(
            entry_train,
            args=(
                _cfg,
                cmd_args.logdir_with_time,
                cmd_args.resume,
                cmd_args.model_path,
                devices,
                port,
            ),
            nprocs=len(devices),
            join=True,
        )

    else:
        assert False
