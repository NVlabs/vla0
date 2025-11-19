# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].

import json
import math
import os
import pickle as pkl
from copy import deepcopy

import imageio
import numpy as np
import numpy.core.multiarray
import roboverse.constants as c
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite import load_controller_config
from roboverse.datasets.lerobot.dataloader import le_sample_to_rv_sample
from roboverse.main import get_cfg
from roboverse.unifiers.image_unifier import (image_unifier_transform,
                                              remove_keys)
from tqdm import tqdm

torch.serialization.add_safe_globals(
    [
        numpy.core.multiarray._reconstruct,
        np.ndarray,
    ]
)

LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data (https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py#L18C1-L18C71)
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

LIBERO_TASK_SUITE_MAX_STEPS = {
    "libero_spatial": 220,  # longest training demo has 193 steps
    "libero_object": 280,  # longest training demo has 254 steps
    "libero_goal": 300,  # longest training demo has 270 steps
    "libero_10": 520,  # longest training demo has 505 steps
    "libero_90": 400,  # longest training demo has 373 steps
}

LIBERO_ACT_SPACES = {
    "ee": "abs_ee",
    "original": "original",
    "delta_ee": "rel_ee",
}


# Taken from openpi repo
def _max_task_length_lookup(task_suite_name) -> int:
    if task_suite_name in LIBERO_TASK_SUITE_MAX_STEPS.keys():
        return LIBERO_TASK_SUITE_MAX_STEPS[task_suite_name]
    else:
        raise ValueError(
            f"Unknown task suite: {task_suite_name}. Available task suites: {LIBERO_TASK_SUITE_MAX_STEPS.keys()}"
        )


def _get_libero_env(
    task,
    seed,
    resolution,
    camera_depths: bool = True,
    action_space="original",
):
    """Initializes and returns the LIBERO environment, along with the task description."""
    assert action_space in ["original", "abs_ee", "rel_ee"]

    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_depths": camera_depths,
    }

    if action_space == "abs_ee" or action_space == "rel_ee":
        ctrl_cfgs = load_controller_config(default_controller="OSC_POSE")
        ctrl_cfgs["control_delta"] = False
        ctrl_args = {
            "control_freq": 20,
            "controller_configs": ctrl_cfgs,
        }
        env_args.update(ctrl_args)

    env = OffScreenRenderEnv(**env_args)
    env.seed(
        seed
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env


def _get_libero_task(task_name, task_suite_name=None, num_steps=0):
    """
    Given task name, and task suite names, return the task, init states, max steps, and task description.

    Args:
        task_name (_type_): _description_
        task_suite_name (_type_, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if task_name is None:
        raise ValueError("task_name must be provided")

    if task_suite_name is not None:
        ts = benchmark.get_benchmark_dict()[task_suite_name]()
        for i, task in enumerate(ts.tasks):
            if task_name == task.name:
                init_states = ts.get_task_init_states(i)
                task_description = task.language
                max_steps = (
                    _max_task_length_lookup(task_suite_name)
                    if num_steps == 0
                    else num_steps
                )
                return task, init_states, max_steps, task_description
        raise ValueError(
            f"Task '{task_name}' not found in task suite '{task_suite_name}'"
        )
    else:
        # iterate over task suites, e.g., ['libero_10', 'libero_90', ...]
        for task_suite_name, task_suite in benchmark.get_benchmark_dict().items():
            # libero_100 doesn't work
            if task_suite_name == "libero_100":
                continue

            # iterate over individual tasks, e.g., [LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket, ...]
            ts = task_suite()
            for i, task in enumerate(ts.tasks):
                if task_name == task.name:
                    # Found the right one
                    init_states = ts.get_task_init_states(i)
                    task_description = task.language
                    max_steps = (
                        _max_task_length_lookup(task_suite_name)
                        if num_steps == 0
                        else num_steps
                    )
                    return task, init_states, max_steps, task_description

    return None, None, None, None


def _get_task_suite_name_from_task(task_name):
    for task_suite_name, task_suite in benchmark.get_benchmark_dict().items():
        ts = task_suite()
        for task in ts.tasks:
            # skip if this is not the correct task
            if task_name is not None and task_name != task.name:
                continue

            return task_suite_name
    return None


def get_evaluation_tasks(task_suite_name=None, task_name=None):
    """
    Given task suite name or task name, return the tasks to evaluate on.

    Args:
        task_suite_name (_type_, optional): _description_. Defaults to None.
        task_name (_type_, optional): _description_. Defaults to None.
    """
    if task_suite_name is not None:
        assert task_suite_name != "libero_100", "libero_100 doesn't work"
        assert (
            task_suite_name in benchmark.get_benchmark_dict().keys()
        ), f"Task suite {task_suite_name} is not a valid task suite. Available task suites: {benchmark.get_benchmark_dict().keys()}"

    tasks_to_evaluate = {}  # {task_suite_name: [task_name, ...], ...}
    if task_suite_name is None and task_name is None:
        # Evaluate on all tasks
        print("Evaluating on all tasks")
        for task_suite_name, task_suite in benchmark.get_benchmark_dict().items():
            # libero_100 doesn't work
            if task_suite_name == "libero_100":
                continue
            ts = task_suite()
            tasks = [task.name for task in ts.tasks]
            tasks_to_evaluate[task_suite_name] = tasks
    elif task_name is None and task_suite_name is not None:
        print(f"Evaluating task suite: {task_suite_name}")
        task_suite = benchmark.get_benchmark_dict()[task_suite_name]
        ts = task_suite()
        tasks = [task.name for task in ts.tasks]
        tasks_to_evaluate[task_suite_name] = tasks
    elif task_name is not None and task_suite_name is None:
        print(f"Evaluating task: {task_name}")
        tasks = [task_name]
        task_suite_name = _get_task_suite_name_from_task(task_name)
        tasks_to_evaluate[task_suite_name] = [task_name]
    else:
        tasks_to_evaluate[task_suite_name] = [task_name]

    return tasks_to_evaluate


def init_libero_env(
    verse_config,
    task_name,
    seed,
    act_space="original",
    task_suite_name=None,
    num_steps=0,
):
    """
    Given task name and optionally the task suite name that it is in, initialize the libero environment.

    Args:
        verse_config (_type_): _description_
        task_name (_type_): _description_
        seed (_type_): _description_
        task_suite_name (str, optional): _description_. Defaults to None.
        act_space (str, optional): _description_. Defaults to "original".

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    assert act_space in ["original", "abs_ee", "rel_ee"]

    if task_name is not None:
        task, init_states, max_steps, task_description = _get_libero_task(
            task_name, task_suite_name, num_steps
        )
        env = _get_libero_env(
            task, seed=seed, action_space=act_space, resolution=LIBERO_ENV_RESOLUTION
        )
        return env, init_states, max_steps, task_description

    raise ValueError(
        f"Cannot initialize libero env for task {task_name}, we should not be here"
    )


@torch.no_grad()
def eval(
    model,
    cfg_path,
    cfg_opts,
    action_type="original",
    task_name=None,
    task_suite_name=None,
    log_dir="",
    save_video=True,
    frame_skip=10,  # default value same as as in openpi, https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py#L37
    action_horizon=5,  # default value same as as in openpi, https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py#L29
    seed=7,
    skip_evaluated=False,
    save_all_data=False,
    ensemble_prediction=1,
    ensemble_version=1,
    ensemble_2_weight=0.5,
    task_id_index=0,
    task_id_count=1,
    num_steps=0,
):
    """
    Writes results to results.json file in the following format:
    {"success": X, "failure": X,
     "task_suite_name": {
        "success": X, "failure": X,
        "task_name": {
            "success": X, "failure": X,
        },
        ...
     },
     ...
    }
    To query success rates:
        success_dict["success"]
        success_dict["failure"]
        success_dict["task_suite_name"]["success"]
        success_dict["task_suite_name"]["failure"]
        success_dict["task_suite_name"]["task_name"]["success"]
        success_dict["task_suite_name"]["task_name"]["failure"]

    Args:
        model (_type_): _description_
        cfg_path (_type_): _description_
        cfg_opts (_type_): _description_
        action_type (str, optional): _description_. Defaults to "original".
        task_name (_type_, optional): _description_. Defaults to None.
        task_suite_name (_type_, optional): _description_. Defaults to None.
        log_dir (str, optional): _description_. Defaults to "".
        save_video (bool, optional): _description_. Defaults to True.
        frame_skip (int, optional): _description_. Defaults to 10.
        action_horizon (int, optional): _description_. Defaults to 15.
        seed (int, optional): _description_. Defaults to 7.
        skip_evaluated (bool, optional): skip tasks that have already been evaluated. Defaults to False. This is checked using the runx_<failure/success>_<task_name>.mp4 file.
        save_all_data (bool, optional): save all data to the log directory. Defaults to False.
        ensemble_prediction (int, optional): number of action chunks to ensemble. Defaults to 1.
        ensemble_version (int, optional): version of the ensemble. Defaults to 1.
        ensemble_2_weight (float, optional): the hyperparameter which controls how the weight weight for each chuch is updated in ensember version 2.
        task_id_index (int, optional): index of the task subset to evaluate. Defaults to 0.
        task_id_count (int, optional): total number of task subsets to evaluate. Defaults to 1. We split the total number of tasks into task_id_count subsets and evaluate on task_id_index subset.
    """
    if task_id_count > 1:
        print(f"Evaluating on task subset {task_id_index} of {task_id_count}")
        print(
            "WARNING: results.json only contains the results for the current task subset"
        )
        assert (
            save_video
        ), "save_video must be True. We used the saved video to calculate success rates."

    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Set random seed
    np.random.seed(seed)

    # Get model config
    cfg = get_cfg(cfg_path, cfg_opts)
    assert cfg.unifier in [c.IMAGE], f"Unifier {cfg.unifier} not supported"

    # Get action space
    assert action_type in ["original"], f"Action type {action_type} not supported"

    # Sanitize task suite/task specifications
    tasks_to_evaluate = get_evaluation_tasks(task_suite_name, task_name)

    # Save the final results to a JSON file
    results_path = os.path.join(log_dir, "results.json")

    # Start evaluation
    success_dict = {"success": 0, "failure": 0}
    for task_suite_name, tasks in tasks_to_evaluate.items():
        success_dict[task_suite_name] = {"success": 0, "failure": 0}
        for task_name in tasks:
            success_dict[task_suite_name][task_name] = {"success": 0, "failure": 0}

            # Initialize Libero environment
            env, init_states, max_steps, language_instruction = init_libero_env(
                seed=seed,
                verse_config=cfg,
                task_name=task_name,
                task_suite_name=task_suite_name,
                act_space=LIBERO_ACT_SPACES[action_type],
                num_steps=num_steps,
            )
            total_states = len(init_states)
            assert (
                total_states % task_id_count == 0
            ), f"Total states {total_states} must be divisible by task_id_count {task_id_count}"
            start_index = task_id_index * (total_states // task_id_count)
            end_index = (task_id_index + 1) * (total_states // task_id_count)

            # Run the eval for the task
            for i, init_state in tqdm(enumerate(init_states)):
                # Skip if not in the current task_id_index
                if not (start_index <= i < end_index):
                    continue

                if skip_evaluated:
                    _language_instruction = deepcopy(language_instruction)
                    task_segment = _language_instruction.replace(" ", "_")
                    success_path = os.path.join(
                        log_dir, f"run{i}__success__{task_segment}.mp4"
                    )
                    failure_path = os.path.join(
                        log_dir, f"run{i}__failure__{task_segment}.mp4"
                    )
                    assert not (
                        os.path.exists(success_path) and os.path.exists(failure_path)
                    ), f"Both success and failure videos exist for task {task_name} run {i}, success path: {success_path}, failure path: {failure_path}"
                    if os.path.exists(success_path) or os.path.exists(failure_path):
                        print(
                            f"Skipping task: {task_name} [{i}/{len(init_states)}] ('{task_suite_name}')"
                        )
                        suffix = (
                            "success" if os.path.exists(success_path) else "failure"
                        )
                        success_dict[task_suite_name][task_name][suffix] += 1
                        success_dict[task_suite_name][suffix] += 1
                        success_dict[suffix] += 1
                        print_task_success_dict(
                            task_name, success_dict[task_suite_name][task_name]
                        )
                        # Save the complete success_dict directly
                        with open(results_path, "w") as f:
                            json.dump(success_dict, f, indent=4)
                        continue

                init_state = init_states[i]
                print(
                    f"Running task: {task_name} [{i}/{len(init_states)}] ('{task_suite_name}')"
                )
                success, frames = eval_run(
                    env=env,
                    model=model,
                    cfg=cfg,
                    language_instruction=language_instruction,
                    init_state=init_state,
                    max_steps=max_steps,
                    frame_skip=frame_skip,
                    action_horizon=action_horizon,
                    log_file_name=f"run{i}__{task_name}.pkl",
                    save_all_data=save_all_data,
                    ensemble_prediction=ensemble_prediction,
                    ensemble_version=ensemble_version,
                    ensemble_2_weight=ensemble_2_weight,
                )

                task_segment = language_instruction.replace(" ", "_")
                suffix = "success" if success else "failure"
                if save_video:
                    frames = np.array(frames)
                    imageio.mimwrite(
                        os.path.join(log_dir, f"run{i}__{suffix}__{task_segment}.mp4"),
                        frames,
                        fps=10,
                    )
                    # Write outcomes to a file
                    outcome_dict = {
                        "init_state": init_state.tolist(),
                        "task_name": task_name,
                        "task_suite_name": task_suite_name,
                        "max_steps": max_steps,
                        "frame_skip": frame_skip,
                        "action_horizon": action_horizon,
                        "seed": seed,
                        "action_type": action_type,
                        "cfg_path": cfg_path,
                        "cfg_opts": cfg_opts,
                        "log_dir": log_dir,
                    }
                    outcome_path = os.path.join(
                        log_dir, f"run{i}__{suffix}__{task_segment}.json"
                    )
                    with open(outcome_path, "w") as f:
                        json.dump(outcome_dict, f, indent=4)

                success_dict[task_suite_name][task_name][suffix] += 1
                success_dict[task_suite_name][suffix] += 1
                success_dict[suffix] += 1

                print_task_success_dict(
                    task_name, success_dict[task_suite_name][task_name]
                )

                # Save the complete success_dict directly
                with open(results_path, "w") as f:
                    json.dump(success_dict, f, indent=4)

            env.close()
            print_task_suite_success_dict(success_dict)

    print("Evaluation complete.")


def eval_run(
    env,
    model,
    cfg,
    language_instruction,
    init_state,
    max_steps,
    frame_skip,
    action_horizon,
    log_file_name,
    save_all_data,
    ensemble_prediction,
    ensemble_version,
    ensemble_2_weight,
):
    """
    Runs 1 episode of the task given the model and environment.

    Args:
        env (_type_): _description_
        model (_type_): _description_
        cfg (_type_): _description_
        language_instruction (_type_): _description_
        init_state (_type_): _description_
        max_steps (_type_): _description_
        frame_skip (_type_): _description_
        action_horizon (_type_): _description_
        log_file_name (_type_): _description_
        save_all_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    # reset to initial state
    env.reset()
    action_i = 0
    action_chunk = None
    obs = env.set_init_state(init_state)

    frames = []
    if save_all_data:
        all_actions = []
        all_obs = []

    if ensemble_prediction > 1:
        old_action_chunks = (
            []
        )  # maintian a list of action chunks from the previous steps

    t = 0
    for t in tqdm(range(max_steps + frame_skip)):
        # This is required for the simulator to settle down.
        if t < frame_skip:
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            if save_all_data:
                all_actions.append(LIBERO_DUMMY_ACTION)
                all_obs.append(obs)
            continue

        if action_i >= action_horizon or t == frame_skip:
            model_obs = libero_to_rv_obs(obs, language_instruction, cfg)
            out = model(**model_obs)
            action_chunk = out["out_ori_act"][0].numpy()  # 1 by 16 by 7
            if save_all_data:
                all_actions.append(action_chunk)
                all_obs.append(obs)

            if ensemble_prediction > 1:
                old_action_chunks.append(action_chunk)
                if len(old_action_chunks) > ensemble_prediction:
                    old_action_chunks.pop(0)

                # updating the previous action chuncks
                n_old_action_chunks = []
                action_chunk = np.zeros_like(action_chunk)
                action_chunk_count = np.zeros_like(action_chunk)
                for i, _action_chunk in enumerate(old_action_chunks[:-1]):
                    # not added to n_old_action_chunks if the action chunk is shorter than the action horizon
                    if len(_action_chunk) <= action_horizon:
                        continue
                    else:
                        _action_chunk = _action_chunk[action_horizon:]
                        n_old_action_chunks.append(_action_chunk)

                    if ensemble_version == 1:
                        action_chunk[0 : len(_action_chunk)] += 0.5 * _action_chunk
                        action_chunk_count[0 : len(_action_chunk)] += 0.5
                    if ensemble_version == 2:
                        action_chunk[0 : len(_action_chunk)] += (
                            ensemble_2_weight ** (len(old_action_chunks) - i - 1)
                        ) * _action_chunk
                        action_chunk_count[
                            0 : len(_action_chunk)
                        ] += ensemble_2_weight ** (len(old_action_chunks) - i - 1)
                # adding the last action chunk
                n_old_action_chunks.append(old_action_chunks[-1])
                action_chunk += old_action_chunks[-1]
                action_chunk_count += 1

                old_action_chunks = n_old_action_chunks
                action_chunk = action_chunk / action_chunk_count

            action_i = 0
            # Add a safeguard for the action horizon rollout.
            action_horizon = min(action_horizon, len(action_chunk))

        act = action_chunk[action_i]
        if not (act[-1] in [1, -1]):
            print(f"Action {act} is not in [1, -1]")
            if act[-1] > 0:
                act[-1] = 1
            else:
                act[-1] = -1
        obs, reward, done, info = env.step(act.tolist())

        frames.append(_img_to_numpy(obs["agentview_image"]))
        if done:
            if save_all_data:
                with open(log_file_name, "wb") as f:
                    pkl.dump(
                        {"actions": all_actions, "obs": all_obs},
                        f,
                    )
            return True, frames
        action_i += 1

    if save_all_data:
        with open(log_file_name, "wb") as f:
            pkl.dump(
                {"actions": all_actions, "obs": all_obs},
                f,
            )
    return False, frames


def _img_to_numpy(img):
    """
    image from the simulator is flipped. so we need to flip it back.
    """
    return np.ascontiguousarray(img[::-1, ::-1])


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def libero_to_lerobot_format(obs, language_instruction):
    # Convert LIBERO data to LEROBOT format

    # LEROBOT dataset format:
    # dict_keys(['image', 'wrist_image', 'state', 'actions', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index', 'state_is_pad', 'actions_is_pad', 'image_is_pad', 'wrist_image_is_pad', 'task'])
    # specifically:
    # dataset state: torch.Size([1, 8])
    # dataset actions: torch.Size([9, 7])
    # dataset image: torch.Size([1, 3, 256, 256])

    # Convert images from (H, W, C) to (C, H, W) format
    image = torch.tensor(
        _img_to_numpy(obs["agentview_image"]) / 255.0, dtype=torch.float32
    ).permute(
        2, 0, 1
    )  # torch.Size([3, 256, 256])
    wrist_image = torch.tensor(
        _img_to_numpy(obs["robot0_eye_in_hand_image"]) / 255.0, dtype=torch.float32
    ).permute(
        2, 0, 1
    )  # torch.Size([3, 256, 256])

    # Construct state vector (robot pose + gripper state)
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    )
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    # Create LEROBOT-compatible data structure
    lerobot_sample = {
        "image": image,  # torch.Size([1, 3, 256, 256])
        "wrist_image": wrist_image,  # torch.Size([1, 3, 256, 256])
        "state": state,  # torch.Size([1, 8])
        "task": language_instruction,  # Task description string
    }

    return lerobot_sample


def libero_to_rv_obs(obs, language_instruction, cfg):
    # TODO: we need to have ori_act in the obs. This is the last action by the model. Currently, we don't need it as no model is using it. But this might be needed in the future

    obs = libero_to_lerobot_format(obs, language_instruction)
    # Format LeRobot into RV Format, in numpy form.
    obs = le_sample_to_rv_sample(
        obs,
        history=cfg.history,
        horizon=cfg.horizon,
        **cfg.LEROBOT,
        add_ori_act=False,
        add_out_ori_act=False,  # always False as we don't know the ground truth action while evaluating
    )

    keys_to_remove = c.REQUIRED_KEYS_3D_COMPATIBLE
    if not cfg.IMAGE.return_ee:
        keys_to_remove.extend(c.REQUIRED_KEYS_EE_COMPATIBLE)
    if not cfg.IMAGE.return_ori_act:
        keys_to_remove.extend(c.REQUIRED_KEYS_ORIGINAL_ACTION)
    if not cfg.IMAGE.return_proprio:
        keys_to_remove.extend(c.REQUIRED_KEYS_PROPRIO)
    obs = remove_keys(obs, keys_to_remove)

    obs = image_unifier_transform(
        cfg, obs, sample_cam_list=cfg.IMAGE.cam_list, eval=True
    )

    # Final format on observations and send to tensor.
    for key in obs.keys():
        if key == "instr":
            obs["instr"] = [obs["instr"]]
        else:
            obs[key] = torch.tensor(obs[key][None], dtype=torch.float32).to(0)

    return obs


def print_task_success_dict(task_name, result):
    # ANSI color codes
    GREEN = "\033[92m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    task_total = result["success"] + result["failure"]
    print(
        f"{BOLD}TASK: '{task_name}'{RESET} {GREEN}Success: {result['success']}/{task_total} ({result['success']/task_total*100:.2f}%){RESET} {RED}Failure: {result['failure']}/{task_total} ({result['failure']/task_total*100:.2f}%){RESET}"  # noqa
    )
    return result["success"], result["failure"]


def print_task_suite_success_dict(success_dict):
    # ANSI color codes
    GREEN = "\033[92m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    for task_suite_name, task_suite in success_dict.items():
        if task_suite_name == "success" or task_suite_name == "failure":
            continue
        print(
            f"{BOLD}---------SUMMARY FOR TASK SUITE: '{task_suite_name}'---------{RESET}"
        )
        for task_name, result in task_suite.items():
            if task_name == "success" or task_name == "failure":
                continue
            succ, fail = print_task_success_dict(task_name, result)
        task_suite_success = success_dict[task_suite_name]["success"]
        task_suite_failure = success_dict[task_suite_name]["failure"]
        task_suite_runs = task_suite_success + task_suite_failure
        print(
            f"{BOLD}TASK SUITE: '{task_suite_name}'{RESET} {GREEN}Success: {task_suite_success}/{task_suite_runs} ({task_suite_success/(task_suite_runs)*100:.2f}%){RESET} {RED}Failure: {task_suite_failure}/{task_suite_runs} ({task_suite_failure/(task_suite_runs)*100:.2f}%){RESET}"  # noqa
        )
    total_success = success_dict["success"]
    total_failure = success_dict["failure"]
    total_runs = total_success + total_failure
    print(f"{BOLD}------------------------------------------------------{RESET}")
    print(
        f"TOTAL: {GREEN}Success: {total_success}/{total_runs} ({total_success/(total_runs)*100:.2f}%){RESET} {RED}Failure: {total_failure}/{total_runs} ({total_failure/(total_runs)*100:.2f}%){RESET}"  # noqa
    )
    print(f"{BOLD}------------------------------------------------------{RESET}")
