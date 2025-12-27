# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""
Leader Arm Teleoperation Demo

This script demonstrates teleoperation of a SO-ARM-101 robot using a leader arm.
The leader arm data is received from a socket connection (via leader_arm_sender.py).

Usage:
1. First, start the leader arm sender in a separate terminal:
   python leader_arm_sender.py --port /dev/ttyUSB0

2. Then run this demo:
   python 07_leader_arm_demo.py --controller ik

Controls:
- Move the leader arm to control the robot's end-effector position
- The gripper on the leader arm controls the robot's gripper
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# Add parent directory to path for controll_scripts module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(description="Leader Arm Teleoperation Demo")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--controller",
    type=str,
    default="osc",
    choices=["ik", "osc"],
    help="Controller type: ik (Differential IK) or osc (Operational Space)",
)
parser.add_argument(
    "--socket-host",
    type=str,
    default="0.0.0.0",
    help="Socket host address (server: bind address, client: target host)",
)
parser.add_argument(
    "--socket-port",
    type=int,
    default=5359,
    help="Socket port for leader arm data (default: 5359)",
)
parser.add_argument(
    "--server-mode",
    action="store_true",
    help="Run as socket server (wait for Mac to connect). Default is client mode.",
)
parser.add_argument(
    "--position-scale",
    type=float,
    default=1.0,
    help="Scale factor for leader arm position (default: 1.0)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import time
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

# Import controll_scripts library
from controll_scripts import (
    ControllerFactory,
    ControllerType,
    LeaderArmInputDevice,
    SOArm101Config,
)


# ========================================
# Scene Configuration
# ========================================

def create_scene_cfg(robot_config: SOArm101Config, for_osc: bool) -> type:
    """Dynamically create scene configuration class"""
    
    class DynamicSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.GroundPlaneCfg(),
        )
        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0),
        )
        robot = robot_config.get_articulation_cfg(for_osc=for_osc).replace(
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
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.0, 0.015)),
        )
    
    return DynamicSceneCfg


def main():
    # Select controller type
    if args_cli.controller == "ik":
        controller_type = ControllerType.IK
        for_osc = False
    else:
        controller_type = ControllerType.OSC if args_cli.controller == "osc" else ControllerType.IK
        for_osc = args_cli.controller == "osc"
    
    print(f"\n{'='*60}")
    print("Leader Arm Teleoperation Demo")
    print(f"{'='*60}")
    print(f"Controller: {args_cli.controller.upper()}")
    print(f"Socket: {args_cli.socket_host}:{args_cli.socket_port}")
    print(f"Position Scale: {args_cli.position_scale}")
    print(f"{'='*60}\n")
    
    # Create robot configuration
    robot_config = SOArm101Config()

    # Create input device (Move this before scene creation to check data_mode)
    # Give a default initial pose (7 components: pos(3) + quat(4))
    initial_pose = torch.tensor([0.25, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0], device=args_cli.device)
    input_device = LeaderArmInputDevice(
        initial_pose=initial_pose,
        device=args_cli.device,
        socket_host=args_cli.socket_host,
        socket_port=args_cli.socket_port,
        server_mode=args_cli.server_mode,
        position_scale=args_cli.position_scale
    )
    
    # Check USD file exists
    if not os.path.exists(robot_config.usd_path):
        print(f"[ERROR]: USD file not found: {robot_config.usd_path}")
        simulation_app.close()
        return
    
    # Initialize simulation
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([0.4, 0.4, 0.4], [0.0, 0.0, 0.15])
    sim_dt = sim.get_physics_dt()
    
    # Create scene
    # Deciding on stiffness based on the input device mode (defaulting to JOINT)
    # In JOINT mode, we ALWAYS want stiffness ON, so we override actual_for_osc to False
    initial_mode = input_device.data_mode
    actual_for_osc = for_osc
    if initial_mode == "joint":
        actual_for_osc = False
        print("[INFO]: Defaulting to JOINT mode. Forcing stiffness ON.")
    else:
        print(f"[INFO]: Starting in {initial_mode.upper()} mode.")
    
    SceneCfg = create_scene_cfg(robot_config, actual_for_osc)
    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Create target marker (frame visualization)
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.prim_path = "/World/Visuals/target_frame"
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    target_marker = VisualizationMarkers(frame_marker_cfg)
    
    # Reset simulation
    sim.reset()
    
    # Get robot
    robot = scene["robot"]
    robot.update(dt=sim_dt)
    
    # Create controller
    controller = ControllerFactory.create(
        controller_type=controller_type,
        robot=robot,
        robot_config=robot_config,
        device=sim.device,
        num_envs=args_cli.num_envs,
    )
    
    # Update input device initial pose
    input_device.reset_target(controller.current_ee_pose)
    
    # Save initial state for reset
    cube: RigidObject = scene['cube']
    cube_initial_pose = torch.tensor([[0.3, 0.0, 0.015, 1.0, 0.0, 0.0, 0.0]], device=sim.device)

    # Get joint IDs and limits for denormalization (0~1 -> actual joint angles)
    arm_joint_ids, arm_joint_names = robot.find_joints(robot_config.arm_joint_names)
    gripper_joint_ids = robot.find_joints([robot_config.gripper_joint_name])[0]

    joint_limits = robot.root_physx_view.get_dof_limits()
    arm_joint_lower = joint_limits[0, arm_joint_ids, 0].to(sim.device)
    arm_joint_upper = joint_limits[0, arm_joint_ids, 1].to(sim.device)

    gripper_lower = joint_limits[0, gripper_joint_ids, 0].item()
    gripper_upper = joint_limits[0, gripper_joint_ids, 1].item()

    # Pre-map IDs for the control loop
    arm_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    arm_sim_indices, _ = robot.find_joints(arm_names)
    gripper_sim_index = gripper_joint_ids[0] if hasattr(gripper_joint_ids, '__len__') else gripper_joint_ids

    print(f'\nArm joint names: {arm_joint_names}')
    print(f'Arm joint IDs: {arm_joint_ids}')
    print(f'Initial EE position: {controller.current_ee_pose[0, :3]}')
    print(f'Sim joint limits (lower, rad): {arm_joint_lower}')
    print(f'Sim joint limits (upper, rad): {arm_joint_upper}')
    print('Starting control loop...\n')

    step_count = 0
    connection_warned = False
    mode_announced = False

    while simulation_app.is_running():
        if not input_device.is_connected:
            if not connection_warned:
                print('[WARNING] Not connected to leader arm sender. Check if sender is running.')
                connection_warned = True
        else:
            if connection_warned:
                print('[INFO] Connected to leader arm sender!')
                connection_warned = False

        # Update input
        target_pose, gripper_pos, reset_requested = input_device.update()
        data_mode = input_device.data_mode

        # Announce mode once
        if not mode_announced and input_device.is_connected:
            if data_mode == 'joint':
                print('[INFO] Running in JOINT mode - direct joint control')
            else:
                print('[INFO] Running in EE mode - using IK/OSC controller')
            mode_announced = True

        # Handle reset
        if reset_requested:
            controller.reset()
            input_device.reset_target(controller.current_ee_pose)

            # Reset cube
            cube.write_root_pose_to_sim(cube_initial_pose)
            cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=sim.device))

            print('[INFO]: Scene reset')
            continue

        if data_mode == 'joint':
            # JOINT MODE: Directly set joint positions
            # joint_positions from InputDevice follow ID 1-5 order from Mac
            joint_normalized = input_device.joint_positions.to(sim.device)  # [5] values 0~1

            # Map Mac IDs 1-6 to Sim names explicitly to ensure correct order
            # Mac IDs: 1:pan, 2:lift, 3:elbow, 4:wrist_p, 5:wrist_r, 6:gripper
            arm_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
            arm_ids_sim, _ = robot.find_joints(arm_names)

            # Apply raw normalization (0-1)
            norm_j = joint_normalized 

            # Denormalize arm joints (0~1 -> actual radians)
            arm_joint_targets = arm_joint_lower + norm_j * (arm_joint_upper - arm_joint_lower)

            # Apply targets
            robot.set_joint_position_target(arm_joint_targets, joint_ids=arm_sim_indices)

            # Gripper (ID 6)
            gripper_target = gripper_lower + gripper_pos * (gripper_upper - gripper_lower)
            robot.set_joint_position_target(torch.tensor([[gripper_target]], device=sim.device), gripper_joint_ids)

            # Write to simulation
            robot.write_data_to_sim()

            # Step simulation
            sim.step()
            robot.update(sim_dt)
            step_count += 1

            # Debug output (Multi-line Table)
            if step_count % 100 == 0:
                current_joints = robot.data.joint_pos[0, arm_sim_indices]
                current_gripper = robot.data.joint_pos[0, gripper_sim_index].item()
                
                joint_errors = (arm_joint_targets - current_joints).abs()
                gripper_error = abs(gripper_target - current_gripper)
                total_error = torch.norm(joint_errors).item()
                
                conn_status = 'Connected' if input_device.is_connected else 'Disconnected'
                age = input_device.last_data_age
                
                print(f"\n{'-'*35} ROBOT STATUS {'-'*35}")
                print(f"Mode: {data_mode.upper():<10} | Conn: {conn_status:<12} | Age: {age:5.3f}s | Total Error: {total_error:6.4f}")
                header = f"{'Joint Name':<15} {'Norm':>8} {'Target':>10} {'Actual':>10} {'Error':>10} {'Limits':>22}"
                print(header)
                print("-" * len(header))
                
                for i, name in enumerate(arm_names):
                    lim_str = f"{arm_joint_lower[i]:.2f} / {arm_joint_upper[i]:.2f}"
                    print(f"{name:<15} {norm_j[i]:8.2f} {arm_joint_targets[i]:10.3f} {current_joints[i]:10.3f} {joint_errors[i]:10.4f} {lim_str:>22}")
                
                grip_lim = f"{gripper_lower:.2f} / {gripper_upper:.2f}"
                print(f"{'gripper':<15} {gripper_pos:8.2f} {gripper_target:10.3f} {current_gripper:10.3f} {gripper_error:10.4f} {grip_lim:>22}")
                print("-" * 105)
        else:
            # EE MODE: Use controller (IK/OSC)
            # Update target marker (position + orientation)
            root_pos_w = robot.data.root_pos_w
            root_quat_w = robot.data.root_quat_w
            import isaaclab.utils.math as math_utils
            target_pos_w, target_quat_w = math_utils.combine_frame_transforms(
                root_pos_w, root_quat_w, target_pose[:, 0:3], target_pose[:, 3:7]
            )
            target_marker.visualize(target_pos_w, target_quat_w)

            # Compute and apply control
            controller.compute(target_pose, gripper_pos)

            # Get current pose for debug output
            current_pose = controller.current_ee_pose
            error = torch.norm(target_pose[0, 0:3] - current_pose[0, 0:3]).item()

            # Step simulation
            sim.step()
            robot.update(sim_dt)
            step_count += 1

            # Debug output every 200 steps
            if step_count % 200 == 0:
                connection_status = 'Connected' if input_device.is_connected else 'Disconnected'
                data_age = input_device.last_data_age
                print(
                    f'Target: [{target_pose[0, 0]:.3f}, {target_pose[0, 1]:.3f}, {target_pose[0, 2]:.3f}] | '
                    f'Actual: [{current_pose[0, 0]:.3f}, {current_pose[0, 1]:.3f}, {current_pose[0, 2]:.3f}] | '
                    f'Error: {error:.4f} | Gripper: {gripper_pos:.2f} | '
                    f'Status: {connection_status} | Data Age: {data_age:.2f}s'
                )
    
    # Cleanup
    input_device.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
