#!/usr/bin/env python3
# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""
Wrapper to use Isaac Lab Mimic consolidated_demo.py with SO-ARM-101

This script:
1. Registers the SO-ARM-101 environment
2. Patches consolidated_demo.py to support leader_arm device
3. Runs the original script

Usage:
    ./isaaclab.sh -p source/isaaclab_mimic/robot/standalone_scripts/mimic_consolidated_demo.py \
        --task Isaac-PickPlace-SOArm-IK-Rel-Mimic-v0 \
        --teleop_device leader_arm \
        --num_envs 1
"""

import argparse
import os
import sys

# === Setup paths ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_DIR = os.path.dirname(SCRIPT_DIR)
ISAACLAB_DIR = os.path.dirname(os.path.dirname(os.path.dirname(ROBOT_DIR)))

# Add robot directory to path
if ROBOT_DIR not in sys.path:
    sys.path.insert(0, ROBOT_DIR)

# Parse args early to check for leader_arm
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--teleop_device", type=str, default="keyboard")
parser.add_argument("--socket-host", type=str, default="0.0.0.0")
parser.add_argument("--socket-port", type=int, default=5359)
args_known, remaining = parser.parse_known_args()

# Import Isaac Lab app launcher
from isaaclab.app import AppLauncher

# Build full argument list for AppLauncher
full_parser = argparse.ArgumentParser(description="SO-ARM-101 Mimic Demo")
full_parser.add_argument("--task", type=str, default="Isaac-PickPlace-SOArm-IK-Rel-Mimic-v0")
full_parser.add_argument("--num_demos", type=int, default=0)
full_parser.add_argument("--num_success_steps", type=int, default=10)
full_parser.add_argument("--num_envs", type=int, default=1)
full_parser.add_argument("--teleop_env_index", type=int, default=0)
full_parser.add_argument("--teleop_device", type=str, default="leader_arm")
full_parser.add_argument("--step_hz", type=int, default=0)
full_parser.add_argument("--input_file", type=str, default=None)
full_parser.add_argument("--output_file", type=str, default="./datasets/so_arm_output.hdf5")
full_parser.add_argument("--generated_output_file", type=str, default=None)
full_parser.add_argument("--socket-host", type=str, default="0.0.0.0")
full_parser.add_argument("--socket-port", type=int, default=5359)
AppLauncher.add_app_launcher_args(full_parser)
args_cli = full_parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Now we can import Isaac Lab modules."""

import asyncio
import contextlib
import gymnasium as gym
import numpy as np
import random
import time
import torch

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets import HDF5DatasetFileHandler

# Register SO-ARM-101 environment
import mimic_envs
mimic_envs.register_envs()

# Import leader arm device
from controll_scripts import Se3LeaderArm, Se3LeaderArmCfg

# Import mimic modules
import isaaclab_mimic.envs  # noqa: F401
from isaaclab_mimic.datagen.data_generator import DataGenerator
from isaaclab_mimic.datagen.datagen_info_pool import DataGenInfoPool

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Global counters
num_recorded = 0
num_success = 0
num_failures = 0
num_attempts = 0


class PreStepDatagenInfoRecorder(RecorderTerm):
    """Recorder term that records the datagen info data in each step."""

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


class PreStepSubtaskTermsObservationsRecorder(RecorderTerm):
    """Recorder term that records the subtask completion observations."""

    def record_pre_step(self):
        return "obs/datagen_info/subtask_term_signals", self._env.get_subtask_term_signals()


@configclass
class PreStepSubtaskTermsObservationsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PreStepSubtaskTermsObservationsRecorder


@configclass
class MimicRecorderManagerCfg(ActionStateRecorderManagerCfg):
    record_pre_step_datagen_info = PreStepDatagenInfoRecorderCfg()
    record_pre_step_subtask_term_signals = PreStepSubtaskTermsObservationsRecorderCfg()


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env):
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.unwrapped.sim.render()
        self.last_time = self.last_time + self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def pre_process_actions(delta_pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    gripper_vel = torch.zeros((delta_pose.shape[0], 1), dtype=torch.float, device=delta_pose.device)
    gripper_vel[:] = -1 if gripper_command else 1
    return torch.concat([delta_pose, gripper_vel], dim=1)


def create_teleop_interface():
    """Create teleop interface based on args."""
    device_name = args_cli.teleop_device.lower()
    
    if device_name == "keyboard":
        return Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
    elif device_name == "spacemouse":
        return Se3SpaceMouse(Se3SpaceMouseCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
    elif device_name == "leader_arm":
        return Se3LeaderArm(Se3LeaderArmCfg(
            socket_host=args_cli.socket_host,
            socket_port=args_cli.socket_port,
            server_mode=True,
            pos_sensitivity=0.2,
            rot_sensitivity=0.5,
            sim_device=args_cli.device,
        ))
    else:
        raise ValueError(f"Unknown teleop device: {device_name}. Supported: keyboard, spacemouse, leader_arm")


async def run_teleop_robot(env, env_id, env_action_queue, shared_datagen_info_pool, success_term, exported_dataset_path):
    """Run teleop robot."""
    global num_recorded
    should_reset_teleop_instance = False
    
    teleop_interface = create_teleop_interface()

    def reset_teleop_instance():
        nonlocal should_reset_teleop_instance
        should_reset_teleop_instance = True

    teleop_interface.add_callback("R", reset_teleop_instance)
    teleop_interface.reset()
    print(teleop_interface)

    recorded_episode_dataset_file_handler = HDF5DatasetFileHandler()
    recorded_episode_dataset_file_handler.create(exported_dataset_path, env_name=env.unwrapped.cfg.env_name)

    env_id_tensor = torch.tensor([env_id], dtype=torch.int64, device=env.unwrapped.device)
    success_step_count = 0
    num_recorded = 0
    
    while True:
        if should_reset_teleop_instance:
            env.unwrapped.recorder_manager.reset(env_id_tensor)
            env.unwrapped.reset(env_ids=env_id_tensor)
            should_reset_teleop_instance = False
            success_step_count = 0

        # Get teleop command
        command = teleop_interface.advance()
        delta_pose = command[:6].unsqueeze(0) if command.shape[0] >= 6 else torch.zeros(1, 6, device=env.unwrapped.device)
        gripper_command = command[6].item() < 0 if command.shape[0] >= 7 else False
        teleop_action = pre_process_actions(delta_pose, gripper_command)

        await env_action_queue.put((env_id, teleop_action[0]))
        await env_action_queue.join()

        if success_term is not None:
            if bool(success_term.func(env, **success_term.params)[env_id]):
                success_step_count += 1
                if success_step_count >= args_cli.num_success_steps:
                    env.recorder_manager.set_success_to_episodes(
                        env_id_tensor, torch.tensor([[True]], dtype=torch.bool, device=env.unwrapped.device)
                    )
                    teleop_episode = env.unwrapped.recorder_manager.get_episode(env_id)
                    await shared_datagen_info_pool.add_episode(teleop_episode)
                    recorded_episode_dataset_file_handler.write_episode(teleop_episode)
                    recorded_episode_dataset_file_handler.flush()
                    env.recorder_manager.reset(env_id_tensor)
                    num_recorded += 1
                    should_reset_teleop_instance = True
            else:
                success_step_count = 0


async def run_data_generator(env, env_id, env_action_queue, shared_datagen_info_pool, success_term, export_demo=True):
    """Run data generator."""
    global num_success, num_failures, num_attempts
    data_generator = DataGenerator(env=env.unwrapped, src_demo_datagen_info_pool=shared_datagen_info_pool)
    idle_action = torch.zeros(env.unwrapped.action_space.shape)[0]
    
    while True:
        while data_generator.src_demo_datagen_info_pool.num_datagen_infos < 1:
            await env_action_queue.put((env_id, idle_action))
            await env_action_queue.join()

        results = await data_generator.generate(
            env_id=env_id,
            success_term=success_term,
            env_action_queue=env_action_queue,
            select_src_per_subtask=env.unwrapped.cfg.datagen_config.generation_select_src_per_subtask,
            transform_first_robot_pose=env.unwrapped.cfg.datagen_config.generation_transform_first_robot_pose,
            interpolate_from_last_target_pose=env.unwrapped.cfg.datagen_config.generation_interpolate_from_last_target_pose,
            export_demo=export_demo,
        )
        if bool(results["success"]):
            num_success += 1
        else:
            num_failures += 1
        num_attempts += 1


def env_loop(env, env_action_queue, shared_datagen_info_pool, asyncio_event_loop):
    """Main loop for the environment."""
    global num_recorded, num_success, num_failures, num_attempts
    prev_num_attempts = 0
    prev_num_recorded = 0

    rate_limiter = RateLimiter(args_cli.step_hz) if args_cli.step_hz > 0 else None

    is_first_print = True
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while True:
            actions = torch.zeros(env.unwrapped.action_space.shape)

            for i in range(env.unwrapped.num_envs):
                env_id, action = asyncio_event_loop.run_until_complete(env_action_queue.get())
                actions[env_id] = action

            env.step(actions)

            for i in range(env.unwrapped.num_envs):
                env_action_queue.task_done()

            if prev_num_attempts != num_attempts or prev_num_recorded != num_recorded:
                prev_num_attempts = num_attempts
                prev_num_recorded = num_recorded
                generated_sucess_rate = 100 * num_success / num_attempts if num_attempts > 0 else 0.0
                if is_first_print:
                    is_first_print = False
                else:
                    print("\r", "\033[F" * 5, end="")
                print("")
                print("*" * 50, "\033[K")
                print(f"{num_recorded} teleoperated demos recorded\033[K")
                print(f"{num_success}/{num_attempts} ({generated_sucess_rate:.1f}%) successful demos generated\033[K")
                print("*" * 50, "\033[K")

                if args_cli.num_demos > 0 and num_recorded >= args_cli.num_demos:
                    print(f"All {args_cli.num_demos} demonstrations recorded. Exiting.")
                    break

            if env.unwrapped.sim.is_stopped():
                break

            if rate_limiter:
                rate_limiter.sleep(env.unwrapped)
    env.close()


def main():
    print("\n" + "=" * 60)
    print("SO-ARM-101 Isaac Lab Mimic Demo Collection")
    print("=" * 60)
    print(f"Task: {args_cli.task}")
    print(f"Teleop device: {args_cli.teleop_device}")
    print(f"Num envs: {args_cli.num_envs}")
    if args_cli.teleop_device.lower() == "leader_arm":
        print(f"Socket: {args_cli.socket_host}:{args_cli.socket_port}")
    print("=" * 60 + "\n")

    num_envs = args_cli.num_envs

    # Create output directory
    os.makedirs(os.path.dirname(args_cli.output_file), exist_ok=True)

    # Parse environment config
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=num_envs)
    env_cfg.env_name = args_cli.task

    # Extract success term
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        print("[WARNING] No success termination term found.")

    env_cfg.terminations = None
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.recorders = MimicRecorderManagerCfg()
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_NONE

    if args_cli.generated_output_file:
        os.makedirs(os.path.dirname(args_cli.generated_output_file), exist_ok=True)
        generated_output_file_name = os.path.splitext(os.path.basename(args_cli.generated_output_file))[0]
        env_cfg.recorders.dataset_export_dir_path = os.path.dirname(args_cli.generated_output_file)
        env_cfg.recorders.dataset_filename = generated_output_file_name
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    if not isinstance(env.unwrapped, ManagerBasedRLMimicEnv):
        raise ValueError("Environment must be ManagerBasedRLMimicEnv")

    # Set seed
    random.seed(env.unwrapped.cfg.datagen_config.seed)
    np.random.seed(env.unwrapped.cfg.datagen_config.seed)
    torch.manual_seed(env.unwrapped.cfg.datagen_config.seed)

    env.reset()

    # Setup asyncio
    asyncio_event_loop = asyncio.get_event_loop()
    env_action_queue = asyncio.Queue()

    shared_datagen_info_pool_lock = asyncio.Lock()
    shared_datagen_info_pool = DataGenInfoPool(
        env.unwrapped, env.unwrapped.cfg, env.unwrapped.device, asyncio_lock=shared_datagen_info_pool_lock
    )
    
    if args_cli.input_file:
        shared_datagen_info_pool.load_from_dataset_file(args_cli.input_file)
        print(f"Loaded {shared_datagen_info_pool.num_datagen_infos} demos from {args_cli.input_file}")

    # Create async tasks
    data_generator_asyncio_tasks = []
    for i in range(num_envs):
        if args_cli.teleop_env_index is not None and i == args_cli.teleop_env_index:
            data_generator_asyncio_tasks.append(
                asyncio_event_loop.create_task(
                    run_teleop_robot(env, i, env_action_queue, shared_datagen_info_pool, success_term, args_cli.output_file)
                )
            )
            continue
        data_generator_asyncio_tasks.append(
            asyncio_event_loop.create_task(
                run_data_generator(
                    env, i, env_action_queue, shared_datagen_info_pool, success_term,
                    export_demo=bool(args_cli.generated_output_file),
                )
            )
        )

    try:
        asyncio.ensure_future(asyncio.gather(*data_generator_asyncio_tasks))
    except asyncio.CancelledError:
        print("Tasks cancelled.")

    env_loop(env, env_action_queue, shared_datagen_info_pool, asyncio_event_loop)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        simulation_app.close()
