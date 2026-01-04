#!/usr/bin/env python3
# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""
Convert Isaac Lab HDF5 demonstration files to LeRobot dataset format.

This script converts demonstration data recorded using Isaac Lab's record_demos.py
to the LeRobot dataset format, enabling training with LeRobot's imitation learning
algorithms (ACT, Diffusion Policy, etc.).

Usage:
    python convert_hdf5_to_lerobot.py \
        --input ./datasets/so_arm_demos.hdf5 \
        --output ./lerobot_datasets/so_arm_stack \
        --robot-type so_arm \
        --fps 30

For more options:
    python convert_hdf5_to_lerobot.py --help
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

# Check if lerobot is available
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("[WARNING] LeRobot not installed. Will output to simple format instead.")
    print("         Install with: pip install lerobot")


# ============================================================================
# Robot Configuration Presets
# ============================================================================

ROBOT_CONFIGS = {
    "so_arm": {
        "state_keys": ["joint_pos"],
        "state_names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
        "action_names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
        "state_dim": 6,
        "action_dim": 6,
    },
    "franka": {
        "state_keys": ["joint_pos"],
        "state_names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "finger_left", "finger_right"],
        "action_names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "finger_left", "finger_right"],
        "state_dim": 9,
        "action_dim": 9,
    },
    "generic": {
        "state_keys": ["joint_pos"],
        "state_names": None,  # Auto-detect from data
        "action_names": None,
        "state_dim": None,
        "action_dim": None,
    },
}


def explore_hdf5_structure(filepath: str) -> dict:
    """Explore and print the structure of an HDF5 file."""
    structure = {}
    
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            structure[name] = {
                "type": "dataset",
                "shape": obj.shape,
                "dtype": str(obj.dtype),
            }
        else:
            structure[name] = {"type": "group"}
    
    with h5py.File(filepath, 'r') as f:
        f.visititems(visitor)
    
    return structure


def print_hdf5_structure(filepath: str):
    """Print the structure of an HDF5 file in a readable format."""
    print(f"\n{'='*60}")
    print(f"HDF5 Structure: {filepath}")
    print('='*60)
    
    structure = explore_hdf5_structure(filepath)
    
    for path, info in sorted(structure.items()):
        indent = "  " * path.count("/")
        name = path.split("/")[-1]
        if info["type"] == "dataset":
            print(f"{indent}📊 {name}: shape={info['shape']}, dtype={info['dtype']}")
        else:
            print(f"{indent}📁 {name}/")
    
    print('='*60 + "\n")


def get_demo_keys(f: h5py.File) -> list:
    """Get all demo keys from the HDF5 file."""
    if "data" in f:
        return sorted([k for k in f["data"].keys() if k.startswith("demo")])
    else:
        # Maybe the demos are at root level
        return sorted([k for k in f.keys() if k.startswith("demo")])


def find_observation_path(demo_grp: h5py.Group, key: str) -> str | None:
    """Find the path to an observation key in the demo group."""
    # Try different possible paths
    paths_to_try = [
        f"observations/policy/{key}",
        f"observations/{key}",
        f"obs/{key}",
        key,
    ]
    
    for path in paths_to_try:
        if path in demo_grp:
            return path
    
    return None


def load_demo_data(demo_grp: h5py.Group, robot_config: dict) -> dict:
    """Load data from a single demo group."""
    data = {}
    
    # Load state observations
    state_arrays = []
    for key in robot_config.get("state_keys", ["joint_pos"]):
        path = find_observation_path(demo_grp, key)
        if path:
            arr = demo_grp[path][:]
            state_arrays.append(arr)
    
    if state_arrays:
        # Concatenate all state arrays along the last dimension
        data["state"] = np.concatenate(state_arrays, axis=-1).astype(np.float32)
    
    # Load actions
    action_paths = ["actions", "action"]
    for path in action_paths:
        if path in demo_grp:
            data["action"] = demo_grp[path][:].astype(np.float32)
            break
    
    # Load optional data
    if "rewards" in demo_grp:
        data["reward"] = demo_grp["rewards"][:].astype(np.float32)
    
    if "dones" in demo_grp:
        data["done"] = demo_grp["dones"][:].astype(bool)
    
    return data


def convert_to_lerobot_format(
    input_path: str,
    output_path: str,
    robot_type: str = "so_arm",
    fps: int = 30,
    repo_id: str | None = None,
    push_to_hub: bool = False,
):
    """Convert Isaac Lab HDF5 to LeRobot dataset format."""
    
    if not LEROBOT_AVAILABLE:
        print("[ERROR] LeRobot is required for this conversion.")
        print("        Please install with: pip install lerobot")
        return False
    
    robot_config = ROBOT_CONFIGS.get(robot_type, ROBOT_CONFIGS["generic"])
    
    print(f"[INFO] Converting {input_path} to LeRobot format")
    print(f"[INFO] Robot type: {robot_type}")
    print(f"[INFO] FPS: {fps}")
    
    # Print structure for debugging
    print_hdf5_structure(input_path)
    
    with h5py.File(input_path, 'r') as f:
        demo_keys = get_demo_keys(f)
        print(f"[INFO] Found {len(demo_keys)} demonstrations")
        
        if len(demo_keys) == 0:
            print("[ERROR] No demonstrations found in the file!")
            return False
        
        # Get data dimensions from first demo
        first_demo = f["data"][demo_keys[0]] if "data" in f else f[demo_keys[0]]
        sample_data = load_demo_data(first_demo, robot_config)
        
        state_dim = sample_data["state"].shape[-1] if "state" in sample_data else 0
        action_dim = sample_data["action"].shape[-1] if "action" in sample_data else 0
        
        print(f"[INFO] State dimension: {state_dim}")
        print(f"[INFO] Action dimension: {action_dim}")
        
        # Define features for LeRobot dataset
        # Generate state names if not provided
        state_names = robot_config.get("state_names") or [f"state_{i}" for i in range(state_dim)]
        action_names = robot_config.get("action_names") or [f"action_{i}" for i in range(action_dim)]
        
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": state_names[:state_dim],
            },
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": action_names[:action_dim],
            },
        }
        
        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create LeRobot dataset
        dataset_repo_id = repo_id or f"local/{output_dir.name}"
        
        print(f"[INFO] Creating LeRobot dataset: {dataset_repo_id}")
        
        dataset = LeRobotDataset.create(
            repo_id=dataset_repo_id,
            fps=fps,
            features=features,
            root=str(output_dir.parent),
        )
        
        # Convert each demonstration
        total_frames = 0
        for i, demo_key in enumerate(demo_keys):
            demo_grp = f["data"][demo_key] if "data" in f else f[demo_key]
            demo_data = load_demo_data(demo_grp, robot_config)
            
            num_frames = len(demo_data["action"])
            print(f"[INFO] Processing {demo_key}: {num_frames} frames")
            
            for frame_idx in range(num_frames):
                frame = {
                    "observation.state": demo_data["state"][frame_idx],
                    "action": demo_data["action"][frame_idx],
                }
                dataset.add_frame(frame)
            
            dataset.save_episode()
            total_frames += num_frames
        
        # Finalize dataset
        print(f"[INFO] Consolidating dataset...")
        dataset.consolidate()
        
        print(f"\n{'='*60}")
        print(f"[SUCCESS] Conversion complete!")
        print(f"  Total episodes: {len(demo_keys)}")
        print(f"  Total frames: {total_frames}")
        print(f"  Output: {output_path}")
        print('='*60)
        
        # Push to hub if requested
        if push_to_hub and repo_id:
            print(f"[INFO] Pushing to Hugging Face Hub: {repo_id}")
            dataset.push_to_hub()
        
        return True


def convert_to_simple_format(
    input_path: str,
    output_path: str,
    robot_type: str = "so_arm",
):
    """Convert to a simple numpy format when LeRobot is not available."""
    
    robot_config = ROBOT_CONFIGS.get(robot_type, ROBOT_CONFIGS["generic"])
    
    print(f"[INFO] Converting {input_path} to simple numpy format")
    print_hdf5_structure(input_path)
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_states = []
    all_actions = []
    episode_ends = []
    
    with h5py.File(input_path, 'r') as f:
        demo_keys = get_demo_keys(f)
        print(f"[INFO] Found {len(demo_keys)} demonstrations")
        
        frame_count = 0
        for demo_key in demo_keys:
            demo_grp = f["data"][demo_key] if "data" in f else f[demo_key]
            demo_data = load_demo_data(demo_grp, robot_config)
            
            all_states.append(demo_data["state"])
            all_actions.append(demo_data["action"])
            
            frame_count += len(demo_data["action"])
            episode_ends.append(frame_count)
    
    # Save as numpy files
    np.save(output_dir / "states.npy", np.concatenate(all_states, axis=0))
    np.save(output_dir / "actions.npy", np.concatenate(all_actions, axis=0))
    np.save(output_dir / "episode_ends.npy", np.array(episode_ends))
    
    # Save metadata
    metadata = {
        "num_episodes": len(demo_keys),
        "total_frames": frame_count,
        "state_dim": all_states[0].shape[-1],
        "action_dim": all_actions[0].shape[-1],
        "robot_type": robot_type,
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[SUCCESS] Saved to {output_path}")
    print(f"  - states.npy: {np.concatenate(all_states, axis=0).shape}")
    print(f"  - actions.npy: {np.concatenate(all_actions, axis=0).shape}")
    print(f"  - episode_ends.npy: {len(episode_ends)} episodes")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert Isaac Lab HDF5 demonstrations to LeRobot dataset format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python convert_hdf5_to_lerobot.py --input ./demos.hdf5 --output ./lerobot_data

  # With specific robot type and FPS
  python convert_hdf5_to_lerobot.py --input ./demos.hdf5 --output ./lerobot_data \\
      --robot-type so_arm --fps 50

  # Just explore the HDF5 structure
  python convert_hdf5_to_lerobot.py --input ./demos.hdf5 --explore-only
        """,
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input HDF5 file from Isaac Lab",
    )
    parser.add_argument(
        "--output", "-o",
        default="./lerobot_datasets/converted",
        help="Output directory for LeRobot dataset (default: ./lerobot_datasets/converted)",
    )
    parser.add_argument(
        "--robot-type",
        choices=list(ROBOT_CONFIGS.keys()),
        default="so_arm",
        help="Robot type for feature naming (default: so_arm)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Recording FPS (default: 30)",
    )
    parser.add_argument(
        "--repo-id",
        help="Hugging Face repository ID (e.g., your_username/dataset_name)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the dataset to Hugging Face Hub after conversion",
    )
    parser.add_argument(
        "--explore-only",
        action="store_true",
        help="Only explore and print HDF5 structure, don't convert",
    )
    parser.add_argument(
        "--simple-format",
        action="store_true",
        help="Output simple numpy format instead of LeRobot format",
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)
    
    # Explore only mode
    if args.explore_only:
        print_hdf5_structure(args.input)
        sys.exit(0)
    
    # Convert
    if args.simple_format or not LEROBOT_AVAILABLE:
        success = convert_to_simple_format(
            input_path=args.input,
            output_path=args.output,
            robot_type=args.robot_type,
        )
    else:
        success = convert_to_lerobot_format(
            input_path=args.input,
            output_path=args.output,
            robot_type=args.robot_type,
            fps=args.fps,
            repo_id=args.repo_id,
            push_to_hub=args.push_to_hub,
        )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
