#!/usr/bin/env python3
# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""
SO-ARM-101 Mimic Demo Launcher

This script is a convenient wrapper around Isaac Lab Mimic's consolidated_demo.py
that automatically:
1. Registers the SO-ARM-101 environment
2. Adds the leader arm device support
3. Passes the correct arguments

Usage:
    # Basic usage (uses default settings)
    ./isaaclab.sh -p source/isaaclab_mimic/robot/standalone_scripts/run_mimic_demo.py

    # Custom settings
    ./isaaclab.sh -p source/isaaclab_mimic/robot/standalone_scripts/run_mimic_demo.py \
        --num_demos 10 \
        --socket-port 5360 \
        --output_file ./my_demos/demo.hdf5
"""

import argparse
import os
import sys

# === Setup paths BEFORE any Isaac Lab imports ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_DIR = os.path.dirname(SCRIPT_DIR)

# Add robot directory to path for local imports
if ROBOT_DIR not in sys.path:
    sys.path.insert(0, ROBOT_DIR)

# NOTE: Do NOT import mimic_envs here - must wait until after SimulationApp

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="SO-ARM-101 Mimic Demo Collection")

# Demo collection settings
parser.add_argument(
    "--num_demos", type=int, default=5,
    help="Number of demonstrations to record (default: 5)"
)
parser.add_argument(
    "--output_file", type=str, default="./datasets/so_arm_demos.hdf5",
    help="Output file path for recorded demos"
)
parser.add_argument(
    "--generated_output_file", type=str, default=None,
    help="Output file for Mimic-generated demos (optional)"
)

# Socket settings for leader arm
parser.add_argument(
    "--socket-host", type=str, default="0.0.0.0",
    help="Socket host for leader arm (default: 0.0.0.0)"
)
parser.add_argument(
    "--socket-port", type=int, default=5359,
    help="Socket port for leader arm (default: 5359)"
)

# Environment settings
parser.add_argument(
    "--num_envs", type=int, default=1,
    help="Number of environments (use 1 for teleoperation only)"
)
parser.add_argument(
    "--step_hz", type=int, default=30,
    help="Environment stepping rate in Hz (default: 30)"
)

# Add Isaac Lab launcher args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of the script runs after Isaac Sim is launched."""

import gymnasium as gym
import numpy as np
import random
import torch

from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets import HDF5DatasetFileHandler

# NOW we can import and register mimic_envs (after SimulationApp is launched)
import mimic_envs
mimic_envs.register_envs()

# Import the environment config class directly
from mimic_envs.so_arm_stack_ik_abs_mimic_env_cfg import SOArmStackIKAbsMimicEnvCfg

# Import leader arm device
from controll_scripts import Se3LeaderArm, Se3LeaderArmCfg


class PreStepDatagenInfoRecorder(RecorderTerm):
    """Recorder term for datagen info."""

    def record_pre_step(self):
        eef_pose_dict = {}
        for eef_name in self._env.cfg.subtask_configs.keys():
            eef_pose_dict[eef_name] = self._env.get_robot_eef_pose(eef_name)

        datagen_info = {
            "object_pose": self._env.get_object_poses(),
            "eef_pose": eef_pose_dict,
            "target_eef_pose": self._env.action_to_target_eef_pose(self._env.action_manager.action),
        }
        return "obs/datagen_info", datagen_info


@configclass
class PreStepDatagenInfoRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PreStepDatagenInfoRecorder


class PreStepSubtaskTermsRecorder(RecorderTerm):
    """Recorder term for subtask signals."""

    def record_pre_step(self):
        return "obs/datagen_info/subtask_term_signals", self._env.get_subtask_term_signals()


@configclass
class PreStepSubtaskTermsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PreStepSubtaskTermsRecorder


@configclass
class MimicRecorderManagerCfg(ActionStateRecorderManagerCfg):
    record_pre_step_datagen_info = PreStepDatagenInfoRecorderCfg()
    record_pre_step_subtask_term_signals = PreStepSubtaskTermsRecorderCfg()


def pre_process_actions(delta_pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
    """Convert teleop command to environment action."""
    gripper_val = torch.zeros((delta_pose.shape[0], 1), dtype=torch.float, device=delta_pose.device)
    gripper_val[:] = -1 if gripper_command else 1
    return torch.concat([delta_pose, gripper_val], dim=1)


def main():
    print("\n" + "=" * 60)
    print("SO-ARM-101 Mimic Demo Collection")
    print("=" * 60)
    print(f"Task: Isaac-PickPlace-SOArm-IK-Abs-Mimic-v0")
    print(f"Demos to collect: {args_cli.num_demos}")
    print(f"Leader arm socket: {args_cli.socket_host}:{args_cli.socket_port}")
    print("=" * 60 + "\n")

    # Create environment config
    env_cfg = SOArmStackIKAbsMimicEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.env_name = "Isaac-PickPlace-SOArm-IK-Abs-Mimic-v0"

    # Disable automatic terminations (we handle success manually)
    success_term = env_cfg.terminations.success if hasattr(env_cfg.terminations, "success") else None
    env_cfg.terminations = None

    # Setup observations
    env_cfg.observations.policy.concatenate_terms = False

    # Setup recorder
    env_cfg.recorders = MimicRecorderManagerCfg()
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_NONE

    # Create environment
    env = gym.make("Isaac-PickPlace-SOArm-IK-Abs-Mimic-v0", cfg=env_cfg)

    # Verify it's a Mimic environment
    if not isinstance(env.unwrapped, ManagerBasedRLMimicEnv):
        raise ValueError("Environment must be ManagerBasedRLMimicEnv")

    # Set random seed
    random.seed(env.unwrapped.cfg.datagen_config.seed)
    np.random.seed(env.unwrapped.cfg.datagen_config.seed)
    torch.manual_seed(env.unwrapped.cfg.datagen_config.seed)

    # Create leader arm device
    teleop_cfg = Se3LeaderArmCfg(
        socket_host=args_cli.socket_host,
        socket_port=args_cli.socket_port,
        server_mode=True,
        sim_device=args_cli.device,
    )
    teleop = Se3LeaderArm(teleop_cfg)

    # Add reset callback
    def reset_callback():
        env.reset()
        print("[INFO] Scene reset")

    teleop.add_callback("R", reset_callback)
    teleop.reset()

    print(teleop)

    # Create dataset file handler
    os.makedirs(os.path.dirname(args_cli.output_file), exist_ok=True)
    dataset_handler = HDF5DatasetFileHandler()
    dataset_handler.create(args_cli.output_file, env_name=env_cfg.env_name)

    # Reset environment
    env.reset()

    # Main loop
    num_recorded = 0
    num_success_steps = 0
    env_id = 0
    device = env.unwrapped.device
    env_id_tensor = torch.tensor([env_id], dtype=torch.int64, device=device)

    print("\nStarted! Move the leader arm to control the robot.")
    print("Press 'R' key or reset on leader arm to start new demo.")

    try:
        while simulation_app.is_running() and num_recorded < args_cli.num_demos:
            # Get teleop command (delta pose + gripper)
            command = teleop.advance()

            # Extract delta pose and gripper
            if command.shape[0] >= 7:
                delta_pose = command[:6].unsqueeze(0)
                gripper_command = command[6].item() < 0  # True = close
            else:
                delta_pose = torch.zeros(1, 6, device=device)
                gripper_command = False

            # Create action
            action = pre_process_actions(delta_pose, gripper_command)

            # Step environment
            env.step(action)

            # Check success (simple timeout-based for now)
            # In practice, implement proper success checking

            # Print status periodically
            if env.unwrapped.episode_length_buf[0] % 100 == 0:
                connected = "Connected" if teleop.is_connected else "Waiting..."
                print(f"\r  Step: {env.unwrapped.episode_length_buf[0].item()} | "
                      f"Demos: {num_recorded}/{args_cli.num_demos} | {connected}", end="")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    finally:
        dataset_handler.close()
        env.close()
        print(f"\nCollection complete! Recorded {num_recorded} demos to {args_cli.output_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
