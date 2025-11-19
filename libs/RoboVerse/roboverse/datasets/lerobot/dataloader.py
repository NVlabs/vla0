# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].

# Suppress unnnecessary warnings
import logging
import warnings
from functools import cache

import numpy as np
import torch
from einops import rearrange
from lerobot.common.datasets.lerobot_dataset import (LeRobotDataset,
                                                     LeRobotDatasetMetadata,
                                                     MultiLeRobotDataset)
from roboverse import constants as c
from torch.utils.data import Dataset

from rv_train.utils.train_utils import ForkedPdb as debug  # noqa: F401

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


@cache
def get_lerobot_metadata(repo_id):
    """
    Get the metadata for a LeRobot dataset
    :param repo_id: (str) Repository ID from huggingface to load the dataset
    :return: (dict) Metadata for the dataset
    """
    return LeRobotDatasetMetadata(repo_id=repo_id)


@cache
def get_final_le_cam_list_rv_cam_list(metadata, le_cam_list, rv_cam_list):
    """
    This is a helper function that is used to get the final le_cam_list and rv_cam_list used in the dataset.
    It checks if the camera names are valid and returns the final le_cam_list and rv_cam_list.
    It works even when le_cam_list and rv_cam_list are None. The le_cam_list is one to one mapped to rv_cam_list.
    It is used in LeRobotRV.__init__ and le_sample_to_rv_sample.
    :param metadata:
    :param le_cam_list: (list) List of le robot camera names to be used as input. Similar to the one provided in LeRobotRV.__init__. Could be None.
    :param rv_cam_list: (list) Corresponding list of camera names in roboverse to be used as input. Similar to the one provided in LeRobotRV.__init__. Could be None.
    """
    if le_cam_list is None:
        le_cam_list = metadata.camera_keys
    else:
        print(f"le_cam_list: {le_cam_list}")
        print(f"metadata.camera_keys: {metadata.camera_keys}")
        for _cam in le_cam_list:
            assert _cam in metadata.camera_keys, f"Camera {_cam} not found in metadata"

    if rv_cam_list is None:
        rv_cam_list = c.CAMERA_NAMES[: len(le_cam_list)]
    else:
        for _cam in rv_cam_list:
            assert (
                _cam in c.CAMERA_NAMES
            ), f"Camera {_cam} not found in roboverse constants"

    assert len(le_cam_list) == len(
        rv_cam_list
    ), "Length of le_cam_list and rv_cam_list should be same"

    return le_cam_list, rv_cam_list


def le_sample_to_rv_sample(
    sample,
    history,
    horizon,
    repo_id,
    le_cam_list,
    rv_cam_list,
    action_key,
    state_key,
    add_ori_act=True,
    add_out_ori_act=True,
    convert_ori_act_to_delta_act=False,
    remove_noop_actions=False,
    fps=-1,
):
    """
    Convert a sample from LeRobot dataset to a sample in RV dataset. Should use the exact same arguments as LeRobotRV.__init__.
    outputs in Numpy.
    :param sample: (dict) A sample from LeRobot dataset
    :param history: (int) Number of history frames
    :param horizon: (int) Number of future action
    :param repo_id: (str) Repository ID from huggingface to load the dataset
    :param le_cam_list: (list) List of le robot camera names to be used as input. Similar to the one provided in LeRobotRV.__init__. Could be None.
    :param rv_cam_list: (list) Corresponding list of camera names in roboverse to be used as input. Similar to the one provided in LeRobotRV.__init__. Could be None.
    :param action_key: (str) Key for action in the dataset
    :param state_key: (str) Key for state in the dataset
    :param add_ori_act: (bool) Whether to add the original action to the sample
    :param add_out_ori_act: (bool) Whether to add the original action to the sample
    :param convert_ori_act_to_delta_act: (bool) Whether to convert the original action to delta action
    :return: (dict) A sample in RV dataset
    """
    rv_sample = {}
    metadata = get_lerobot_metadata(repo_id)
    le_cam_list, _ = get_final_le_cam_list_rv_cam_list(
        metadata, le_cam_list, rv_cam_list
    )

    rgb = [sample[x] * 255 for x in le_cam_list]
    if history > 1:
        rgb = [rearrange(x, "hi c h w -> hi 1 h w c") for x in rgb]
    else:
        if rgb[0].ndim == 3:
            rgb = [rearrange(x, "c h w -> 1 1 h w c") for x in rgb]
        else:
            # for libero, we find that the rgb is of shape (1, 3, 224, 224)
            # not ideal but we can still use it
            rgb = [rearrange(x, "1 c h w -> 1 1 h w c") for x in rgb]
    rv_sample["rgb"] = torch.cat(rgb, dim=1).numpy()
    rv_sample["instr"] = sample["task"]
    rv_sample["proprio"] = sample[state_key]
    if add_ori_act:
        rv_sample["ori_act"] = sample[action_key][:history]
    if add_out_ori_act:
        rv_sample["out_ori_act"] = sample[action_key][history:]
    if convert_ori_act_to_delta_act:
        rv_sample["out_ori_act"] = rv_sample["out_ori_act"] - rv_sample["ori_act"][-1]

    return rv_sample


class LeRobotRV(Dataset):
    """
    RV dataset that wraps on top of any LeRobot dataset
    Based on information provided here: https://github.com/huggingface/lerobot/blob/main/examples/1_load_lerobot_dataset.py
    """

    def __init__(
        self,
        history,
        horizon,
        repo_id,
        le_cam_list,
        rv_cam_list,
        action_key="action",
        state_key="observation.state",
        episodes=None,
        convert_ori_act_to_delta_act=False,
        remove_noop_actions=False,
        fps=-1,
    ):
        """
        :param history: (int) Number of history frames
        :param horizon: (int) Number of future action
        :param repo_id: (str) Repository ID from huggingface to load the dataset. For multiple datasets, use a comma separated string. It is the reposibility of the user to ensure that the datasets are compatible when using multiple datasets.
        :param le_cam_list: (list) List of le robot camera names to be used as input.
            If None, all cameras are used. The ID should match the keys in
            metadata.camera_keys
        :param rv_cam_list: (list) Corresponding list of camera names in
            roboverse to be used as input. The length of this list should match the
            length of le_cam_list. If None, each le_camera is mapped to a
            corresponding camera in CAMERA_NAMES in roboverse constants
        :param episodes: (list) List of episodes to be used. If None, all episodes are used.
        :param convert_ori_act_to_delta_act: (bool) Whether to convert the original action to delta action. This assumes that original actions are provided in absolution units. For all out_ori_act, we subtract the last ori_act from it to get the delta action.
        """
        if "," in repo_id:
            repo_id = repo_id.split(",")
        else:
            repo_id = [repo_id]

        self.repo_id = repo_id
        self.history = history
        self.horizon = horizon
        self.convert_ori_act_to_delta_act = convert_ori_act_to_delta_act
        self.remove_noop_actions = remove_noop_actions

        if self.remove_noop_actions:
            for repo_id in self.repo_id:
                assert (
                    "hugohadfieldnvidia" in repo_id
                ), "Remove noop actions is only supported for hugohadfieldnvidia datasets. For other dataset, make sure the condition for removing noop actions is met. Identifying noop condition is defined in the __getitem__ function."

        # get metadata
        metadata = get_lerobot_metadata(self.repo_id[0])

        # set action and state keys
        assert (
            action_key in metadata.features
        ), f"Action key '{action_key}' not found in metadata. Available metadata keys: {metadata}"
        assert (
            state_key in metadata.features
        ), f"Observation key '{state_key}' not found in metadata. Available metadata keys: {metadata}"
        self.action_key = action_key
        self.state_key = state_key

        # set other attributes
        self.threed_compatible = False
        self.ee_compatible = False
        self.has_original_action = True
        self.original_action_dim = metadata.features[self.action_key]["shape"][0]
        self.has_proprio = True
        self.proprio_dim = metadata.features[self.state_key]["shape"][0]

        # set cam_list that is required for RoboVerse
        self.le_cam_list = le_cam_list
        self.rv_cam_list = rv_cam_list
        # _le_cam_list is the final le_cam_list that is used in the dataset
        # it is not the same as le_cam_list, it does the processing in get_final_le_cam_list_rv_cam_list
        _le_cam_list, _rv_cam_list = get_final_le_cam_list_rv_cam_list(
            metadata, le_cam_list, rv_cam_list
        )
        print(f"LeRobot and RoboVerse Camera Mapping: {_le_cam_list} -> {_rv_cam_list}")
        self.cam_list = _rv_cam_list

        # get the LeRobotDataset
        if fps == -1:
            fps = metadata.fps
        delta_timestamps = {
            # loads proprio from -history+1 to 0
            self.state_key: [-x / fps for x in range(history - 1, -1, -1)],
            # loads action from timesteps -history to horizon-1
            self.action_key: [
                x / fps for x in range(-history, horizon)
            ],  # for both past and future actions
        }
        print(f"Delta Timestamps: {delta_timestamps}")
        for cam in _le_cam_list:
            # loads camera frames from -history+1 to 0
            delta_timestamps[cam] = [-x / fps for x in range(history - 1, -1, -1)]

        if len(repo_id) > 1:
            print(f"Loading MultiLeRobotDataset with repo_ids: {self.repo_id}")
            self.dataset = MultiLeRobotDataset(
                repo_ids=self.repo_id,
                delta_timestamps=delta_timestamps,
                episodes=episodes,
            )
        else:
            self.dataset = LeRobotDataset(
                repo_id=self.repo_id[0],
                delta_timestamps=delta_timestamps,
                episodes=episodes,
            )

        # CRITICAL FIX: Handle episode indexing bug in LeRobotDataset when episodes are filtered
        # This uses the same approach as PR #1062: https://github.com/huggingface/lerobot/pull/1062/files
        if episodes is not None:
            self._fix_episode_data_index()

        # for delta action, we need to compute the stats via the get_dataset_stats method in stats_utils.py
        # for len(self.repo_id) > 1, we haven't tested if the stats are computed correctly.
        if not self.convert_ori_act_to_delta_act and (not len(self.repo_id) > 1):
            act_stats = self.dataset.meta.stats[self.action_key]
            del act_stats["mean"]
            del act_stats["std"]
            self.stats = {
                "out_ori_act": act_stats,
            }

    def _fix_episode_data_index(self):
        """
        Fix episode_data_index mapping when episodes are filtered in LeRobotDataset.
        Uses the same approach as PR #1062: https://github.com/huggingface/lerobot/pull/1062/files

        THE PROBLEM:
        ------------
        LeRobotDataset has a critical bug when filtering episodes. Here's what happens:

        1. EPISODE FILTERING: When episodes=[10, 25, 428, 512] is passed, LeRobotDataset:
           - Loads only data files for those specific episodes
           - BUT the actual episode_index values in the data remain [10, 25, 428, 512]

        2. EPISODE_DATA_INDEX CREATION: The get_episode_data_index() function creates:
           - episode_data_index["from"] = [0, len_ep10, len_ep10+len_ep25, ...]  # 4 elements
           - episode_data_index["to"] = [len_ep10, len_ep10+len_ep25, ...]      # 4 elements
           - These arrays are indexed 0, 1, 2, 3 (sequential)

        3. THE BUG: In __getitem__, LeRobotDataset does:
           - item = self.hf_dataset[idx]
           - ep_idx = item["episode_index"].item()  # This is 428 (original episode number!)
           - self._get_query_indices(idx, ep_idx)
           - Inside _get_query_indices: episode_data_index["from"][ep_idx]
           - Tries to access episode_data_index["from"][428] but array only has 4 elements!
           - Result: IndexError: index 428 is out of bounds for dimension 0 with size 4

        THE PR #1062 SOLUTION:
        ----------------------
        Instead of creating sparse arrays, we recreate episode_data_index with:
        1. ALL episodes from the original dataset (not just filtered ones)
        2. Set length = 0 for episodes NOT in the filtered list
        3. Set length = actual_length for episodes in the filtered list
        4. Use cumulative sum to create proper from/to arrays

        EXAMPLE:
        --------
        Original dataset has episodes 0-999, we filter to episodes [10, 25, 428, 512]:

        BEFORE FIX:
        - episode_data_index["from"] has 4 elements: [0, 150, 300, 450]
        - When accessing episode 428: episode_data_index["from"][428] → IndexError!

        AFTER FIX (PR #1062 approach):
        - episode_data_index["from"] has 1000 elements: [0, 0, 0, ..., 0, 150, 0, ..., 300, ...]
        - episode_data_index["from"][10] = 0      ✓
        - episode_data_index["from"][25] = 150    ✓
        - episode_data_index["from"][428] = 300   ✓
        - episode_data_index["from"][512] = 450   ✓
        - episode_data_index["from"][427] = 300   (same as 428, because ep 427 has length 0)
        """
        from itertools import accumulate

        episodes = self.dataset.episodes
        metadata_episodes = self.dataset.meta.episodes

        # Edge case: if episodes is empty, nothing to fix
        if not episodes:
            return

        # Recreate episode_lengths using PR #1062 approach:
        # - For episodes in filtered list: use actual length
        # - For episodes NOT in filtered list: use length 0
        # This ensures all original episode indices are present in the arrays
        episodes_set = set(episodes)  # Convert to set for O(1) lookup
        episode_lengths = {
            ep_idx: ep_dict["length"] if ep_idx in episodes_set else 0
            for ep_idx, ep_dict in metadata_episodes.items()
        }

        # Recreate episode_data_index exactly like the original get_episode_data_index function
        # but with the fixed episode_lengths that includes all episodes
        cumulative_lengths = list(accumulate(episode_lengths.values()))
        fixed_episode_data_index = {
            "from": torch.LongTensor([0] + cumulative_lengths[:-1]),
            "to": torch.LongTensor(cumulative_lengths),
        }

        # Replace the buggy episode_data_index with our fixed version
        # Now when LeRobotDataset.__getitem__ calls episode_data_index["from"][428],
        # it will correctly return the start index instead of throwing IndexError
        self.dataset.episode_data_index = fixed_episode_data_index

        # VERIFICATION: After this fix (same as PR #1062):
        # - episode_data_index["from"][10] returns the correct start frame for episode 10
        # - episode_data_index["from"][428] returns the correct start frame for episode 428
        # - episode_data_index["from"][427] returns the same as episode 428 (because ep 427 has length 0)
        # - The validation loop in LeRobotLibero.__init__ will now pass
        # - Training will work correctly without IndexError

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset.__getitem__(idx)
        rv_sample = le_sample_to_rv_sample(
            sample,
            self.history,
            self.horizon,
            self.repo_id[0],
            self.le_cam_list,
            self.rv_cam_list,
            self.action_key,
            self.state_key,
            convert_ori_act_to_delta_act=self.convert_ori_act_to_delta_act,
        )
        if self.remove_noop_actions:
            is_noop = False
            if "hugohadfieldnvidia" in self.repo_id[0]:
                # check if the action is noop
                if self.convert_ori_act_to_delta_act:
                    delta_action = rv_sample["out_ori_act"]
                else:
                    delta_action = rv_sample["out_ori_act"] - rv_sample["ori_act"][-1]

                # all movements are less than 1 degrees
                if torch.all(torch.abs(delta_action) < 1).item():
                    is_noop = True

            if is_noop:
                return self.__getitem__(np.random.randint(0, len(self)))
        return rv_sample


if __name__ == "__main__":
    # Usage
    from roboverse.main import get_cfg

    # cfg = get_cfg("roboverse/configs/img_libero.yaml")
    cfg = get_cfg("roboverse/configs/img_real.yaml")
    # cfg.unifier = "image"

    dataset = LeRobotRV(
        **cfg.LEROBOT,
        history=1,
        horizon=8,
    )

    sample = dataset.__getitem__(idx=0)
    print(f"Length of dataset: {len(dataset)}")
    print(f"Sample: {sample}")
    print(f"Sample keys: {sample.keys()}")
    print(f"LeRobot Dataset keys: {dataset.dataset[0].keys()}")
    breakpoint()

    # for libero
    # print(f"dataset state: {dataset.dataset[0].get('state').shape}")
    # print(f"dataset actions: {dataset.dataset[0].get('actions').shape}")
    # print(f"dataset image: {dataset.dataset[0].get('image').shape}")
    # print(f"dataset wrist_image: {dataset.dataset[0].get('wrist_image').shape}")

    # sample = dataset.__getitem__(idx=0)
    # print(f"sample.keys(): {sample.keys()}")
    # print(f"state: {sample.get('state').shape}")
    # print(f"actions: {sample.get('actions').shape}")
    # print(f"image: {sample.get('image').shape}")
    # print(f"wrist_image: {sample.get('wrist_image').shape}")
