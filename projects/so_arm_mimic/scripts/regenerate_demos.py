# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to regenerate demonstrations by replaying actions using the current Task Config."""

import argparse
import os
import contextlib
import gymnasium as gym
import torch

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Regenerate demonstrations by replaying actions with new env config.")
parser.add_argument("--task", type=str, required=True, help="Task name to load config from (e.g. Isaac-PickPlace-SOArm-Joint-Mimic-v0).")
parser.add_argument("--input_file", type=str, required=True, help="Input HDF5 dataset file (source actions).")
parser.add_argument("--output_file", type=str, required=True, help="Output HDF5 dataset file (new observations).")
parser.add_argument("--enable_pinocchio", action="store_true", default=False, help="Enable Pinocchio.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio

# Launch simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
import so_arm_mimic.envs  # Register custom environments
import isaaclab_tasks  # Register official tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

def main():
    if not os.path.exists(args_cli.input_file):
        raise FileNotFoundError(f"The input file {args_cli.input_file} does not exist.")

    # 1. Load Input Dataset (Source of Actions)
    input_handler = HDF5DatasetFileHandler()
    input_handler.open(args_cli.input_file)
    input_episode_count = input_handler.get_num_episodes()
    print(f"Loading actions from: {args_cli.input_file} ({input_episode_count} episodes)")

    # 2. Load Task Configuration (Source of Robot/Obs definition)
    # We use the explicit task name provided by user
    env_name = args_cli.task
    print(f"Loading task config for: {env_name}")
    
    # Parse config
    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=1)
    
    # Disable events (randomization) to ensure deterministic replay
    env_cfg.events = {}
    env_cfg.recorders = {} 
    env_cfg.terminations = {} 

    # Create environment
    env = gym.make(env_name, cfg=env_cfg).unwrapped
    
    # 3.5 Setup Visualization Viewports (Dynamic)
    if args_cli.enable_cameras:
        from view_port import setup_camera_viewports
        setup_camera_viewports(env_cfg, simulation_app)

    # 4. Init Output Handler
    output_handler = HDF5DatasetFileHandler()
    output_handler.create(args_cli.output_file, env_name=env_name)
    
    # Save Env Spec/Metadata
    if hasattr(env.cfg, "get_ep_meta"):
        output_handler.add_env_args(env.cfg.get_ep_meta())
    else:
        ep_meta = {
            "sim_args": {
                "dt": env.cfg.sim.dt,
                "decimation": env.cfg.decimation,
                "render_interval": env.cfg.sim.render_interval,
                "num_envs": env.cfg.scene.num_envs,
            }
        }
        output_handler.add_env_args(ep_meta)

    # 5. Regeneration Loop
    env.reset()
    episode_names = list(input_handler.get_episode_names())
    
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        for ep_idx in range(input_episode_count):
            ep_name = episode_names[ep_idx]
            print(f"Regenerating Episode {ep_idx + 1}/{input_episode_count}...")
            
            # Load source episode
            src_episode_data = input_handler.load_episode(ep_name, env.device)
            # Use length of actions to determine steps
            if "actions" in src_episode_data.data:
                total_steps = len(src_episode_data.data["actions"])
            else:
                # Fallback or error
                print(f"Warning: No actions found in episode {ep_name}. Skipping.")
                continue
            
            # --- Reset Physics State to Source Initial State ---
            initial_state = src_episode_data.get_initial_state()
            env_ids = torch.tensor([0], device=env.device)
            # Use reset_to to force physics state
            env.reset_to(initial_state, env_ids, is_relative=True)
            
            # --- Reset Managers (Important for clean state) ---
            if hasattr(env, "action_manager"):
                env.action_manager.reset(env_ids)
            if hasattr(env, "observation_manager"):
                env.observation_manager.reset(env_ids)
            
            # --- Create New Episode Data ---
            new_episode_data = EpisodeData()
            new_episode_data.env_id = 0
            # Keep initial state from source (physics state)
            if "initial_state" in src_episode_data.data:
                 new_episode_data.add("initial_state", src_episode_data.data["initial_state"])
            
            # Get first observation (t=0)
            obs_dict = env.observation_manager.compute()
            
            for step_i in range(total_steps):
                # 1. Get Action from Source
                action = src_episode_data.get_next_action()
                if action is None:
                    break
                if action.dim() == 1:
                    action = action.unsqueeze(0)
                
                # 2. Record Pre-Step Info (Obs, Action)
                # Store observations derived from current config
                for group_name, group_data in obs_dict.items():
                    new_episode_data.add(f"obs/{group_name}", group_data)
                
                new_episode_data.add("actions", action)
                
                # 3. Simulate Step
                obs_dict, rew, terminated, truncated, info = env.step(action)
                
                # 4. Record Result
                new_episode_data.add("rewards", rew)
                # Note: 'dones' are implicit in dataset length usually, or can be added
            
            # Mark as success (since we are regenerating known demos)
            new_episode_data.success = True
            
            # Check length consistency
            recorded_steps = 0
            if "actions" in new_episode_data.data:
                # Assuming list of tensors or tensor
                acts = new_episode_data.data["actions"]
                recorded_steps = len(acts) if isinstance(acts, list) else acts.shape[0]

            if recorded_steps != total_steps:
                print(f"Warning: Episode length mismatch. Src: {total_steps}, New: {recorded_steps}")

            # Prepare data for export (stack lists into tensors)
            new_episode_data.pre_export()

            # Export
            output_handler.write_episode(new_episode_data)
            output_handler.flush()
            
    output_handler.close()
    env.close()
    print("Regeneration Complete!")
    print(f"New dataset: {args_cli.output_file}")

if __name__ == "__main__":
    main()
