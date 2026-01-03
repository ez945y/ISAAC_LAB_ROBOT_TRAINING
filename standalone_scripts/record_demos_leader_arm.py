#!/usr/bin/env python3
# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""
Record demonstrations using Leader Arm for SO-ARM-101.

This script wraps Isaac Lab's record_demos.py to add leader_arm support.
It uses the standard Isaac Lab demo recording infrastructure.

Usage:
    ./isaaclab.sh -p source/isaaclab_mimic/robot/standalone_scripts/record_demos_leader_arm.py \
        --task Isaac-PickPlace-SOArm-IK-Rel-Mimic-v0 \
        --dataset_file ./datasets/so_arm_demos.hdf5

    # With socket settings
    ./isaaclab.sh -p source/isaaclab_mimic/robot/standalone_scripts/record_demos_leader_arm.py \
        --task Isaac-PickPlace-SOArm-IK-Rel-Mimic-v0 \
        --socket-port 5360
"""

import argparse
import contextlib
import os
import sys
import time

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROBOT_DIR not in sys.path:
    sys.path.insert(0, ROBOT_DIR)

from isaaclab.app import AppLauncher

# Arguments
parser = argparse.ArgumentParser(description="Record demos with Leader Arm for SO-ARM-101")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-SOArm-IK-Rel-Mimic-v0", help="Task name")
parser.add_argument("--dataset_file", type=str, default="./datasets/so_arm_demos.hdf5", help="Output file")
parser.add_argument("--step_hz", type=int, default=30, help="Step rate in Hz")
parser.add_argument("--num_demos", type=int, default=0, help="Number of demos (0=infinite)")
parser.add_argument("--num_success_steps", type=int, default=10, help="Steps to confirm success")
# Leader arm settings
parser.add_argument("--socket-host", type=str, default="0.0.0.0", help="Socket host")
parser.add_argument("--socket-port", type=int, default=5359, help="Socket port")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""After SimulationApp is launched."""

import gymnasium as gym
import torch

import omni.log
import omni.ui as ui

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.ui import EmptyWindow
from isaaclab.managers import DatasetExportMode

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Register SO-ARM-101 environment
import mimic_envs
mimic_envs.register_envs()

# Import leader arm
from controll_scripts import Se3LeaderArm, Se3LeaderArmCfg


class RateLimiter:
    def __init__(self, hz: int):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env):
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time = self.last_time + self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def main():
    print("\n" + "=" * 60)
    print("SO-ARM-101 Demo Recording with Leader Arm")
    print("=" * 60)
    print(f"Task: {args_cli.task}")
    print(f"Output: {args_cli.dataset_file}")
    print(f"Socket: {args_cli.socket_host}:{args_cli.socket_port}")
    print("=" * 60 + "\n")

    # Setup output directory
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Parse environment config
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.env_name = args_cli.task.split(":")[-1]

    # Extract success term
    success_term = None
    if hasattr(env_cfg, "terminations") and env_cfg.terminations is not None:
        if hasattr(env_cfg.terminations, "success"):
            success_term = env_cfg.terminations.success
            env_cfg.terminations.success = None
        # Disable timeout but keep the terminations config
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
    else:
        omni.log.warn("No terminations config found.")

    # Observations config
    if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.concatenate_terms = False

    # Setup recorder
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir if output_dir else "."
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Create leader arm teleop device
    teleop = Se3LeaderArm(Se3LeaderArmCfg(
        socket_host=args_cli.socket_host,
        socket_port=args_cli.socket_port,
        server_mode=True,
        pos_sensitivity=0.2,
        rot_sensitivity=0.5,
        sim_device=args_cli.device,
    ))

    # State
    should_reset = False
    current_recorded = 0
    success_step_count = 0

    def reset_callback():
        nonlocal should_reset
        should_reset = True
        print("[INFO] Reset requested")

    teleop.add_callback("R", reset_callback)

    # Setup UI
    window = EmptyWindow(env, "Demo Recording")
    with window.ui_window_elements["main_vstack"]:
        status_label = ui.Label(f"Recorded: {current_recorded} demos")

    # Reset
    env.sim.reset()
    env.reset()
    teleop.reset()

    rate_limiter = RateLimiter(args_cli.step_hz)

    print("\nRecording started!")
    print("  - Move leader arm to control robot")
    print("  - Press 'R' or leader arm reset to save/reset")
    print(f"  - Target: {args_cli.num_demos if args_cli.num_demos > 0 else 'unlimited'} demos\n")

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running():
            # Get teleop action
            action = teleop.advance()
            actions = action.repeat(env.num_envs, 1)

            # Step environment
            env.step(actions)

            # Check success
            if success_term is not None:
                if bool(success_term.func(env, **success_term.params)[0]):
                    success_step_count += 1
                    if success_step_count >= args_cli.num_success_steps:
                        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                        env.recorder_manager.set_success_to_episodes(
                            [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                        )
                        env.recorder_manager.export_episodes([0])
                        print("[INFO] Success! Demo saved.")
                        should_reset = True
                else:
                    success_step_count = 0

            # Update count
            if env.recorder_manager.exported_successful_episode_count > current_recorded:
                current_recorded = env.recorder_manager.exported_successful_episode_count
                status_label.text = f"Recorded: {current_recorded} demos"
                print(f"[INFO] Total demos: {current_recorded}")

            # Check if done
            if args_cli.num_demos > 0 and current_recorded >= args_cli.num_demos:
                print(f"\n[INFO] Completed {current_recorded} demos!")
                break

            # Handle reset
            if should_reset:
                env.sim.reset()
                env.recorder_manager.reset()
                env.reset()
                success_step_count = 0
                should_reset = False
                print("[INFO] Environment reset")

            if env.sim.is_stopped():
                break

            rate_limiter.sleep(env)

    env.close()
    print(f"\nRecording complete! Saved {current_recorded} demos to {args_cli.dataset_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        simulation_app.close()
