# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].

"""
Minimal example for testing the VLA deployment API.

This script demonstrates how to call the prediction endpoints without robot hardware.
Just replace SERVER_URL with your server address.
"""

import base64
import io
import json
from typing import List, Optional

import numpy as np
import requests
from tqdm import tqdm

# Configuration
SERVER_URL = "http://localhost:10000"  # Replace with your server address


def rgb_as_base64(rgb: np.ndarray) -> str:
    """Convert a numpy RGB image array to base64 string."""
    img = np.array(rgb, dtype=np.uint8)
    array_bytes = io.BytesIO()
    np.save(array_bytes, img)
    return base64.b64encode(array_bytes.getvalue()).decode("utf-8")


def predict_base64(
    images: List[np.ndarray], state: List[float], instruction: Optional[str] = None
):
    """
    Call the /predict_base64 endpoint to get full action sequence.

    Args:
        images: List of RGB images as numpy arrays (H, W, 3)
        state: Robot state as list of 6 floats [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
        instruction: Optional text instruction

    Returns:
        Action sequence as numpy array (horizon, 6)
    """
    payload = {
        "base64_rgb": [rgb_as_base64(img) for img in images],
        "state": state,
        "instr": instruction,
    }

    response = requests.post(f"{SERVER_URL}/predict_base64", json=payload)
    response.raise_for_status()

    action_array = np.array(response.json())
    return action_array


def predict_base64_stream(
    images: List[np.ndarray], state: List[float], instruction: Optional[str] = None
):
    """
    Call the /predict_base64_stream endpoint to get actions one at a time.

    Args:
        images: List of RGB images as numpy arrays (H, W, 3)
        state: Robot state as list of 6 floats
        instruction: Optional text instruction

    Yields:
        Dictionary with action data for each step
    """
    payload = {
        "base64_rgb": [rgb_as_base64(img) for img in images],
        "state": state,
        "instr": instruction,
    }

    response = requests.post(
        f"{SERVER_URL}/predict_base64_stream", json=payload, stream=True
    )
    response.raise_for_status()

    for line in response.iter_lines():
        data = json.loads(line)
        if "value" in data:
            action = np.array(data["value"][0])
            yield action


def example_usage():
    """Example of how to use the API with dummy data."""

    # Create dummy data (replace with real camera images and robot state)
    dummy_images = [
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),  # front camera
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),  # left camera
    ]

    dummy_state = [-8.177, -98.898, 99.637, 50.923, -1.651, 6.600]  # 6 joint positions

    instruction = "Push the apple to the block"

    print("Compiling the model...")
    actions = predict_base64(dummy_images, dummy_state, instruction)
    print(f"Received action sequence with shape: {actions.shape}")
    actions_stream = predict_base64_stream(dummy_images, dummy_state, instruction)
    for action in actions_stream:
        print(f"Received action sequence with shape: {action.shape}")
    print("Compiled the model.")

    print("=" * 60)
    print("Example 1: Full action sequence (/predict_base64)")
    print("Predicts all actions at once.")
    print("=" * 60)

    for _ in tqdm(range(10)):
        actions = predict_base64(dummy_images, dummy_state, instruction)
        print(f"Received action sequence with shape: {actions.shape}")

    print("\n" + "=" * 60)
    print("Example 2: Streaming actions (/predict_base64_stream)")
    print("Streams actions one at a time.")
    print("=" * 60)

    for i in range(10):
        for action_data in tqdm(
            predict_base64_stream(dummy_images, dummy_state, instruction)
        ):
            print(f"Received action sequence with shape: {action_data.shape}")


if __name__ == "__main__":
    # Check server health first
    try:
        response = requests.get(f"{SERVER_URL}/health")
        response.raise_for_status()
        print(f"✓ Server is healthy at {SERVER_URL}\n")
    except requests.exceptions.RequestException as e:
        print(f"✗ Cannot connect to server at {SERVER_URL}")
        print(f"Error: {e}")
        exit(1)

    # Run examples
    example_usage()
