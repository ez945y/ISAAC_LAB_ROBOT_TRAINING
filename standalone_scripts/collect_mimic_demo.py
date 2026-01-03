#!/usr/bin/env python3
# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""
SO-ARM-101 Mimic Teleoperation Demo Collection Script

This script collects teleoperation demonstrations using LeaderArmInputDevice
for Isaac Lab Mimic data generation workflow.

The collected demonstrations can be used with:
1. annotate_demos.py - Annotate subtask boundaries
2. generate_dataset.py - Generate synthetic demonstrations
3. Train policies using Robomimic or other IL frameworks

Usage:
    # Start server mode (waiting for LeaderArm connection)
    python collect_mimic_demo.py --num-demos 10 --output-dir ./demos

    # With custom socket settings
    python collect_mimic_demo.py --socket-port 5360 --num-demos 5
"""

import argparse
import os
import sys
import time
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SO-ARM-101 Mimic Demo Collection")
parser.add_argument(
    "--num-demos",
    type=int,
    default=5,
    help="Number of demonstrations to collect (default: 5)",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="./mimic_demos",
    help="Output directory for collected demos (default: ./mimic_demos)",
)
parser.add_argument(
    "--socket-host",
    type=str,
    default="0.0.0.0",
    help="Socket host address (default: 0.0.0.0 for server mode)",
)
parser.add_argument(
    "--socket-port",
    type=int,
    default=5359,
    help="Socket port for leader arm data (default: 5359)",
)
parser.add_argument(
    "--position-scale",
    type=float,
    default=1.0,
    help="Scale factor for leader arm position (default: 1.0)",
)
parser.add_argument(
    "--episode-length",
    type=float,
    default=60.0,
    help="Maximum episode length in seconds (default: 60.0)",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import h5py
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
import isaaclab.utils.math as math_utils

# Import local modules
from controll_scripts import (
    ControllerFactory,
    ControllerType,
    LeaderArmInputDevice,
    SOArm101Config,
)


class MimicDemoCollector:
    """
    Collects teleoperation demonstrations for Isaac Lab Mimic workflow.
    
    This class:
    1. Initializes the simulation with SO-ARM-101 robot
    2. Uses LeaderArmInputDevice for real-time control
    3. Records state, action, and observation data
    4. Saves demonstrations in HDF5 format compatible with Isaac Lab Mimic
    """

    def __init__(
        self,
        robot_config: SOArm101Config,
        input_device: LeaderArmInputDevice,
        output_dir: str,
        num_demos: int,
        episode_length: float,
        device: str = "cuda:0",
    ):
        self.robot_config = robot_config
        self.input_device = input_device
        self.output_dir = output_dir
        self.num_demos = num_demos
        self.episode_length = episode_length
        self.device = device

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Simulation objects
        self.sim = None
        self.scene = None
        self.robot = None
        self.controller = None
        self.target_marker = None
        self.cube = None

        # Recording state
        self.current_demo_data = None
        self.demos_collected = 0
        self.is_recording = False
        self.step_count = 0

    def setup(self) -> bool:
        """Initialize simulation environment."""
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

        # Check USD file
        if not os.path.exists(self.robot_config.usd_path):
            print(f"[ERROR]: USD file not found: {self.robot_config.usd_path}")
            return False

        # Initialize simulation
        sim_cfg = sim_utils.SimulationCfg(device=self.device)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.sim.set_camera_view([-0.4, -0.4, 0.4], [0.0, 0.0, 0.15])
        self.sim_dt = self.sim.get_physics_dt()

        # Create scene
        class DemoSceneCfg(InteractiveSceneCfg):
            ground = AssetBaseCfg(
                prim_path="/World/defaultGroundPlane",
                spawn=sim_utils.GroundPlaneCfg(),
            )
            dome_light = AssetBaseCfg(
                prim_path="/World/Light",
                spawn=sim_utils.DomeLightCfg(intensity=3000.0),
            )
            robot = self.robot_config.get_articulation_cfg(for_osc=False).replace(
                prim_path="{ENV_REGEX_NS}/robot"
            )
            cube = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/cube",
                spawn=sim_utils.CuboidCfg(
                    size=(0.03, 0.03, 0.03),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.2)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, 0.0, 0.015)),
            )

        scene_cfg = DemoSceneCfg(num_envs=1, env_spacing=2.0)
        self.scene = InteractiveScene(scene_cfg)

        # Target marker
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.prim_path = "/World/Visuals/target_frame"
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.target_marker = VisualizationMarkers(frame_marker_cfg)

        # Reset simulation
        self.sim.reset()

        # Get robot and cube
        self.robot = self.scene["robot"]
        self.cube = self.scene["cube"]
        self.robot.update(dt=self.sim_dt)

        # Create controller
        self.controller = ControllerFactory.create(
            controller_type=ControllerType.IK,
            robot=self.robot,
            robot_config=self.robot_config,
            device=self.sim.device,
            num_envs=1,
        )

        # Initialize input device with current EE pose
        self.input_device.reset_target(self.controller.current_ee_pose)

        # Get joint info
        self.arm_joint_ids, self.arm_joint_names = self.robot.find_joints(
            self.robot_config.arm_joint_names
        )
        gripper_ids, _ = self.robot.find_joints([self.robot_config.gripper_joint_name])
        self.gripper_joint_id = gripper_ids[0] if gripper_ids else 0

        self._print_info()
        return True

    def _print_info(self):
        """Print collection info."""
        print(f"\n{'='*60}")
        print("SO-ARM-101 Mimic Demo Collection")
        print(f"{'='*60}")
        print(f"Output directory: {self.output_dir}")
        print(f"Demos to collect: {self.num_demos}")
        print(f"Episode length: {self.episode_length}s")
        print(f"Robot: {self.robot_config.name}")
        print(f"\nControls:")
        print("  - Use LeaderArm to control the robot")
        print("  - Demo recording starts when connected")
        print("  - Press 'r' in terminal to reset/save demo")
        print("  - Press 'q' in terminal to quit")
        print(f"{'='*60}\n")

    def _start_new_demo(self):
        """Start recording a new demonstration."""
        self.current_demo_data = {
            # Robot state
            "joint_positions": [],
            "joint_velocities": [],
            "ee_positions": [],
            "ee_orientations": [],
            "gripper_positions": [],
            # Actions
            "actions": [],
            "target_ee_positions": [],
            "target_ee_orientations": [],
            "gripper_commands": [],
            # Object state
            "cube_positions": [],
            "cube_orientations": [],
            # Metadata
            "timestamps": [],
        }
        self.is_recording = True
        self.step_count = 0
        print(f"\n[INFO] Started recording demo {self.demos_collected + 1}/{self.num_demos}")

    def _record_step(self, target_pose: torch.Tensor, gripper_cmd: float, action: torch.Tensor):
        """Record a single step of the demonstration."""
        if not self.is_recording:
            return

        # Robot state
        joint_pos = self.robot.data.joint_pos[0, self.arm_joint_ids].cpu().numpy()
        joint_vel = self.robot.data.joint_vel[0, self.arm_joint_ids].cpu().numpy()
        gripper_pos = self.robot.data.joint_pos[0, self.gripper_joint_id].cpu().numpy()

        # EE pose from controller
        ee_pose = self.controller.current_ee_pose[0].cpu().numpy()
        ee_pos = ee_pose[:3]
        ee_quat = ee_pose[3:7]

        # Cube state
        cube_pos = self.cube.data.root_pos_w[0].cpu().numpy()
        cube_quat = self.cube.data.root_quat_w[0].cpu().numpy()

        # Target pose
        target_pos = target_pose[0, :3].cpu().numpy()
        target_quat = target_pose[0, 3:7].cpu().numpy()

        # Store data
        data = self.current_demo_data
        data["joint_positions"].append(joint_pos)
        data["joint_velocities"].append(joint_vel)
        data["ee_positions"].append(ee_pos)
        data["ee_orientations"].append(ee_quat)
        data["gripper_positions"].append(gripper_pos)
        data["actions"].append(action.cpu().numpy())
        data["target_ee_positions"].append(target_pos)
        data["target_ee_orientations"].append(target_quat)
        data["gripper_commands"].append(gripper_cmd)
        data["cube_positions"].append(cube_pos)
        data["cube_orientations"].append(cube_quat)
        data["timestamps"].append(self.step_count * self.sim_dt)

        self.step_count += 1

    def _save_demo(self) -> bool:
        """Save the current demonstration to HDF5 file."""
        if not self.is_recording or len(self.current_demo_data["actions"]) < 10:
            print("[WARNING] Demo too short or not recording, skipping save")
            return False

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"demo_{self.demos_collected:03d}_{timestamp}.hdf5"
        filepath = os.path.join(self.output_dir, filename)

        # Convert lists to numpy arrays
        data = {}
        for key, value in self.current_demo_data.items():
            data[key] = np.array(value)

        # Save to HDF5
        with h5py.File(filepath, "w") as f:
            # Create main data group
            demo_grp = f.create_group("data/demo_0")
            
            # Store observations
            obs_grp = demo_grp.create_group("obs")
            obs_grp.create_dataset("joint_positions", data=data["joint_positions"])
            obs_grp.create_dataset("joint_velocities", data=data["joint_velocities"])
            obs_grp.create_dataset("ee_pos", data=data["ee_positions"])
            obs_grp.create_dataset("ee_quat", data=data["ee_orientations"])
            obs_grp.create_dataset("gripper_pos", data=data["gripper_positions"])
            obs_grp.create_dataset("cube_pos", data=data["cube_positions"])
            obs_grp.create_dataset("cube_quat", data=data["cube_orientations"])

            # Store actions
            demo_grp.create_dataset("actions", data=data["actions"])
            
            # Store target poses (for Mimic workflow)
            demo_grp.create_dataset("target_ee_pos", data=data["target_ee_positions"])
            demo_grp.create_dataset("target_ee_quat", data=data["target_ee_orientations"])
            demo_grp.create_dataset("gripper_commands", data=data["gripper_commands"])

            # Metadata
            demo_grp.attrs["num_samples"] = len(data["actions"])
            demo_grp.attrs["timestamp"] = timestamp
            f.attrs["total"] = 1
            f.attrs["env_name"] = "Isaac-SOArm-PickPlace-IK-Abs-Mimic-v0"

        print(f"[INFO] Saved demo to {filepath} ({len(data['actions'])} steps)")
        
        self.demos_collected += 1
        self.is_recording = False
        return True

    def _reset_scene(self):
        """Reset the scene to initial state."""
        # Reset robot
        self.controller.reset()
        self.input_device.reset_target(self.controller.current_ee_pose)

        # Reset cube
        cube_initial_pose = torch.tensor(
            [[0.25, 0.0, 0.015, 1.0, 0.0, 0.0, 0.0]], device=self.sim.device
        )
        self.cube.write_root_pose_to_sim(cube_initial_pose)
        self.cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=self.sim.device))

        print("[INFO] Scene reset")

    def run(self):
        """Main collection loop."""
        import select
        import sys

        print("\n[INFO] Waiting for LeaderArm connection...")
        print("[INFO] Press 's' to start/save demo, 'r' to reset, 'q' to quit\n")

        connection_warned = False

        while simulation_app.is_running() and self.demos_collected < self.num_demos:
            # Check for keyboard input (non-blocking)
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1).lower()
                if key == "q":
                    print("[INFO] Quit requested")
                    break
                elif key == "r":
                    if self.is_recording:
                        self._save_demo()
                    self._reset_scene()
                    self._start_new_demo()
                elif key == "s":
                    if not self.is_recording:
                        self._start_new_demo()
                    else:
                        self._save_demo()

            # Check connection
            if not self.input_device.is_connected:
                if not connection_warned:
                    print("[WARNING] Waiting for LeaderArm connection...")
                    connection_warned = True
                time.sleep(0.1)
                continue
            else:
                if connection_warned:
                    print("[INFO] LeaderArm connected!")
                    connection_warned = False
                    if not self.is_recording:
                        self._start_new_demo()

            # Get control input
            target_pose, gripper_pos, reset_requested = self.input_device.update()

            if reset_requested:
                if self.is_recording:
                    self._save_demo()
                self._reset_scene()
                self._start_new_demo()
                continue

            # Update target marker
            root_pos_w = self.robot.data.root_pos_w
            root_quat_w = self.robot.data.root_quat_w
            target_pos_w, target_quat_w = math_utils.combine_frame_transforms(
                root_pos_w, root_quat_w, target_pose[:, 0:3], target_pose[:, 3:7]
            )
            self.target_marker.visualize(target_pos_w, target_quat_w)

            # Compute control
            self.controller.compute(target_pose, gripper_pos)

            # Create action tensor for recording (absolute pose + gripper)
            action = torch.cat([
                target_pose[0, :3],  # position
                target_pose[0, 3:7],  # quaternion
                torch.tensor([gripper_pos], device=self.device),  # gripper
            ])

            # Record step
            self._record_step(target_pose, gripper_pos, action)

            # Step simulation
            self.sim.step()
            self.robot.update(self.sim_dt)
            self.cube.update(self.sim_dt)

            # Check episode timeout
            if self.is_recording and self.step_count * self.sim_dt >= self.episode_length:
                print(f"[INFO] Episode timeout ({self.episode_length}s)")
                self._save_demo()
                self._reset_scene()
                if self.demos_collected < self.num_demos:
                    self._start_new_demo()

            # Progress display
            if self.step_count % 100 == 0 and self.is_recording:
                elapsed = self.step_count * self.sim_dt
                print(
                    f"  Recording: {elapsed:.1f}s / {self.episode_length}s | "
                    f"Demo {self.demos_collected + 1}/{self.num_demos}"
                )

        # Save any remaining demo
        if self.is_recording:
            self._save_demo()

        # Cleanup
        self.input_device.close()
        print(f"\n[INFO] Collection complete! Saved {self.demos_collected} demos to {self.output_dir}")


def main():
    print(f"\n{'='*60}")
    print("SO-ARM-101 Mimic Demo Collection")
    print(f"{'='*60}")

    # Robot configuration
    robot_config = SOArm101Config()

    # Input device
    initial_pose = torch.tensor([0.25, 0.0, 0.15, 1.0, 0.0, 0.0, 0.0], device=args_cli.device)
    input_device = LeaderArmInputDevice(
        initial_pose=initial_pose,
        device=args_cli.device,
        socket_host=args_cli.socket_host,
        socket_port=args_cli.socket_port,
        server_mode=True,
        position_scale=args_cli.position_scale,
    )

    # Create collector
    collector = MimicDemoCollector(
        robot_config=robot_config,
        input_device=input_device,
        output_dir=args_cli.output_dir,
        num_demos=args_cli.num_demos,
        episode_length=args_cli.episode_length,
        device=args_cli.device,
    )

    if collector.setup():
        collector.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
