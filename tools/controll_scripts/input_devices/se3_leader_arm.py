# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""
Leader Arm Device for Isaac Lab

This module provides a DeviceBase-compatible wrapper around LeaderArmInputDevice,
allowing it to be used with Isaac Lab's standard teleoperation scripts like
`consolidated_demo.py`.

Usage with consolidated_demo.py:
    ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/consolidated_demo.py \
        --task Isaac-PickPlace-SOArm-IK-Abs-Mimic-v0 \
        --teleop_device leader_arm \
        --num_demos 10
"""

import numpy as np
import torch
from collections.abc import Callable
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation
from typing import Any, Optional

from isaaclab.devices.device_base import DeviceBase, DeviceCfg


@dataclass
class Se3LeaderArmCfg(DeviceCfg):
    """Configuration for Leader Arm SE(3) device.
    
    This configuration allows customizing the leader arm connection and behavior.
    """
    
    # Socket settings
    socket_host: str = "0.0.0.0"
    """Socket host address. Use '0.0.0.0' for server mode (listening)."""
    
    socket_port: int = 5359
    """Socket port for leader arm data."""
    
    server_mode: bool = True
    """If True, run as server (wait for connection). If False, connect as client."""
    
    # Control settings
    pos_sensitivity: float = 1.0
    """Position sensitivity multiplier."""
    
    rot_sensitivity: float = 1.0
    """Rotation sensitivity multiplier."""
    
    gripper_term: bool = True
    """Whether to include gripper command in output."""
    
    # Device settings
    sim_device: str = "cuda:0"
    """Device to place tensors on."""
    
    retargeters: None = None
    """Retargeters (not used for leader arm, kept for compatibility)."""


class Se3LeaderArm(DeviceBase):
    """A leader arm controller for sending SE(3) commands as delta poses.
    
    This class wraps the LeaderArmInputDevice to conform to Isaac Lab's DeviceBase
    interface, enabling seamless integration with existing teleoperation scripts.
    
    The command comprises of two parts:
    * delta pose: a 6D vector of (x, y, z, roll, pitch, yaw) in meters and radians.
    * gripper: a binary command to open or close the gripper.
    
    Unlike keyboard control which sends delta commands, the leader arm sends
    absolute poses. This class tracks the previous pose to compute deltas
    for compatibility with Isaac Lab's delta-pose action space.
    """
    
    def __init__(self, cfg: Se3LeaderArmCfg):
        """Initialize the leader arm device.
        
        Args:
            cfg: Configuration for the leader arm.
        """
        super().__init__(retargeters=None)
        
        self._cfg = cfg
        self._sim_device = cfg.sim_device
        self._pos_sensitivity = cfg.pos_sensitivity
        self._rot_sensitivity = cfg.rot_sensitivity
        self._gripper_term = cfg.gripper_term
        
        # Import and create the actual leader arm input device
        # We do lazy import to avoid circular dependencies
        self._leader_arm: Optional["LeaderArmInputDevice"] = None
        self._init_leader_arm()
        
        # State tracking for delta computation
        self._prev_pos = np.zeros(3)
        self._prev_quat = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        self._close_gripper = False
        self._initialized = False
        
        # Callbacks
        self._additional_callbacks: dict[str, Callable] = {}
        
        # Connection state tracking for auto-reset on disconnect->reconnect
        self._was_connected = False
        self._pending_reset = False
        
    def _init_leader_arm(self):
        """Initialize the leader arm input device."""
        try:
            # Import using isaaclab_mimic.controll_scripts path
            from controll_scripts.input_devices.leader_arm import LeaderArmInputDevice
            
            initial_pose = torch.tensor(
                [0.25, 0.0, 0.15, 1.0, 0.0, 0.0, 0.0],
                device=self._sim_device
            )
            
            self._leader_arm = LeaderArmInputDevice(
                initial_pose=initial_pose,
                device=self._sim_device,
                socket_host=self._cfg.socket_host,
                socket_port=self._cfg.socket_port,
                server_mode=self._cfg.server_mode,
            )
            print(f"[Se3LeaderArm] Initialized on {self._cfg.socket_host}:{self._cfg.socket_port}")
            
        except ImportError as e:
            print(f"[Se3LeaderArm] Warning: Could not import LeaderArmInputDevice: {e}")
            print("[Se3LeaderArm] Falling back to zero commands")
            self._leader_arm = None
    
    def __del__(self):
        """Cleanup resources."""
        if self._leader_arm is not None:
            self._leader_arm.close()
    
    def __str__(self) -> str:
        """Returns: A string containing device information."""
        msg = f"Leader Arm Controller for SE(3): {self.__class__.__name__}\n"
        msg += f"\tSocket: {self._cfg.socket_host}:{self._cfg.socket_port}\n"
        msg += f"\tMode: {'Server' if self._cfg.server_mode else 'Client'}\n"
        msg += "\t----------------------------------------------\n"
        if self._leader_arm is not None:
            connected = "Connected" if self._leader_arm.is_connected else "Waiting..."
            msg += f"\tStatus: {connected}\n"
        msg += "\tControl: Move the physical leader arm\n"
        msg += "\tGripper: Use gripper on leader arm"
        return msg
    
    def reset(self):
        """Reset the controller state."""
        self._prev_pos = np.zeros(3)
        self._prev_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._close_gripper = False
        self._initialized = False
        
        if self._leader_arm is not None:
            # Try to get current pose as reference
            try:
                target_pose, _, _ = self._leader_arm.update()
                # Check if target_pose has the expected shape (1, 7) or (7,)
                if target_pose is not None and target_pose.numel() >= 7:
                    if target_pose.dim() == 2:
                        pose_np = target_pose[0].cpu().numpy()
                    else:
                        pose_np = target_pose.cpu().numpy()
                    
                    if pose_np.shape[0] >= 7:
                        self._prev_pos = pose_np[:3].copy()
                        self._prev_quat = pose_np[3:7].copy()
                        self._initialized = True
            except Exception as e:
                # If anything goes wrong, just use default values
                print(f"[Se3LeaderArm] Reset with default pose (not connected yet): {e}")
    
    def add_callback(self, key: Any, func: Callable):
        """Add callback function.
        
        Note: Leader arm doesn't have keyboard-like callbacks,
        but we store them for compatibility.
        
        Args:
            key: The key/event to bind.
            func: The callback function.
        """
        self._additional_callbacks[key] = func
    
    def advance(self) -> torch.Tensor:
        """Provides the result from leader arm state.
        
        Computes delta pose from the current and previous absolute poses
        to match Isaac Lab's action space (delta pose + gripper).
        
        Returns:
            torch.Tensor: A 7-element tensor containing:
                - delta pose: First 6 elements as [dx, dy, dz, drx, dry, drz]
                - gripper command: Last element as binary (+1.0 open, -1.0 close)
        """
        # Check connection status and handle disconnect -> reconnect -> reset
        currently_connected = self._leader_arm is not None and self.is_connected
        
        if self._was_connected and not currently_connected:
            # Connection was lost! Mark as disconnected but don't reset yet
            print("\n>> [DISCONNECT] Leader arm disconnected! Reconnect to reset and start new episode...")
            self._pending_reset = True
            self._was_connected = False
        
        if not currently_connected:
            # Return default pose (curled up) while waiting for reconnect
            default_pose = torch.tensor([0.055, -1.74, 1.665, 1.233, -0.077, -0.17], 
                                      dtype=torch.float32, device=self._sim_device)
            return default_pose
        
        # Check for reconnection after disconnect
        if getattr(self, '_pending_reset', False) and currently_connected:
            # Just reconnected! Now trigger reset
            print(">> [RECONNECT] Leader arm reconnected! Triggering reset...")
            if "R" in self._additional_callbacks:
                self._additional_callbacks["R"]()
            self._pending_reset = False
        
        # Update connection state
        self._was_connected = True
        
        # Check current mode
        if hasattr(self._leader_arm, 'data_mode') and self._leader_arm.data_mode == 'joint':
            # === Joint Control Mode ===
            # Return absolute joint positions directly
            # 5 Arm Joints + 1 Gripper
            
            # Get latest joint positions (5 normalized values)
            joints = self._leader_arm.joint_positions
            
            # Get gripper position
            _, gripper_pos, _ = self._leader_arm.update()
            
            # Scale joint positions if needed (assuming 0-1 normalized input)
            # You might need to map these to your robot's actual joint limits!
            # For now, we assume the environment handles normalized inputs or 
            # the input is already in radians. 
            # If input is 0-1, we might need mapping. 
            # Let's assume input is radians for now or handled by lerobot.
            # (If lerobot sends 0-1, we need to scale. SO-ARM-100 limits are approx -3.14 to 3.14 for most joints)
            
            # TODO: Implement scaling if input is 0-1. 
            # Assuming input is radians if not specified otherwise.
            # If input device says "normalized joint positions (0~1)", we usually maps 0.5 to 0 radians.
            
            # Map 0~1 to -3.14~3.14 (approx)
            # This is a Rough Estimation!
            # === Joint Correction Configuration ===
            # Adjust these to match your Leader Arm with the Simulation
            
            # 1. Reorder: Map input index to output index
            # Default: [0, 1, 2, 3, 4] (Pass-through)
            # If input[0] should control sim[1], then put 0 at index 1? No.
            # Output[i] = Input[MAP[i]]
            JOINT_MAP = [0, 1, 2, 3, 4] 
            
            # 2. Invert: Multiply by 1.0 or -1.0
            JOINT_SIGNS = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], device=self._sim_device)
            
            # 3. Offset: Add constant (in radians)
            # Reverted to 0.0 as user sets leader arm to center/neutral
            JOINT_OFFSETS = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], device=self._sim_device)
            
            # Apply corrections
            # Reorder
            ordered_joints = joints[JOINT_MAP]
            
            # === Explicit Joint Mapping ===
            # Based on 06_teleoperate_demo.py logic: 
            # target = lower + norm * (upper - lower)
            
            # SO-ARM-100 Limits (Updated based on observed actual limits)
            # J1: -1.92 / 1.92
            # J2: -1.75 / 1.75
            # J3: -1.69 / 1.69
            # J4: -1.66 / 1.66
            # J5: -2.74 / 2.84
            self.arm_lower = torch.tensor([-1.92, -1.92, -1.92, -1.66, -2.74], device=self._sim_device)
            self.arm_upper = torch.tensor([ 1.92,  1.75,  1.69,  1.66,  2.84], device=self._sim_device)
            
            # Map normalized [0,1] to radians
            # Note: We apply JOINT_MAP reordering *before* this if needed, 
            # currently ordered_joints is joints[JOINT_MAP]
            
            # Calculation
            scaled_joints = self.arm_lower + ordered_joints * (self.arm_upper - self.arm_lower)
            
            # Apply signs and offsets (Correction Layer)
            scaled_joints = scaled_joints * JOINT_SIGNS + JOINT_OFFSETS
            
            # Gripper Mapping
            # Observed Limit: -0.17 / 1.75
            # Input 0 (Close) -> Sim -0.17 (Close?) or 0? 
            # Input 1 (Open) -> Sim 1.75 (Open?)
            
            # Let's verify interaction:
            # "gripper 1.00 Target 1.745"
            # This implies Input 1.0 -> Target 1.75.
            # So Gripper Upper is 1.75.
            # And Gripper Lower is -0.17.
            
            # Let's use these observed limits
            self.gripper_lower = -0.17
            self.gripper_upper = 1.75
            
            sim_gripper = self.gripper_lower + gripper_pos * (self.gripper_upper - self.gripper_lower)
            
            # Debug Print (Table Format) - Commented out for cleaner output
            # if not hasattr(self, "_debug_print_counter"):
            #     self._debug_print_counter = 0
            # self._debug_print_counter += 1
            # if self._debug_print_counter % 30 == 0:
            #     print(f"\n{'-'*35} LEADER ARM STATUS {'-'*35}")
            #     header = f"{'Joint Id':<10} {'Norm (In)':>12} {'Target (Rad)':>15} {'Limits':>22}"
            #     print(header)
            #     print('-' * len(header))
            #     
            #     # Arm Joints
            #     for i in range(5):
            #         lim_str = f'{self.arm_lower[i]:.2f} / {self.arm_upper[i]:.2f}'
            #         print(f'Joint {i:<4} {ordered_joints[i]:12.2f} {scaled_joints[i]:15.3f} {lim_str:>22}')
            #     
            #     # Gripper
            #     grip_lim = f'{self.gripper_lower:.2f} / {self.gripper_upper:.2f}'
            #     print(f'Gripper    {gripper_pos:12.2f} {sim_gripper:15.3f} {grip_lim:>22}')
            #     print('-' * 80)
            
            command = torch.cat([scaled_joints, torch.tensor([sim_gripper], device=self._sim_device)])
            return command
            
            command = torch.cat([scaled_joints, torch.tensor([sim_gripper], device=self._sim_device)])
            return command

        # === End-Effector Delta Control Mode ===
        # Get current pose from leader arm
        target_pose, gripper_pos, reset_requested = self._leader_arm.update()
        
        # Handle reset callback
        if reset_requested and "R" in self._additional_callbacks:
            self._additional_callbacks["R"]()
        
        # Extract current pose safely
        try:
            # Check if target_pose has expected shape
            if target_pose is None or target_pose.numel() < 7:
                # Return default pose if no valid data
                default_pose = torch.tensor([0.055, -1.74, 1.665, 1.233, -0.077, -0.17], 
                                          dtype=torch.float32, device=self._sim_device)
                return default_pose
            
            if target_pose.dim() == 2:
                pose_np = target_pose[0].cpu().numpy()
            else:
                pose_np = target_pose.cpu().numpy()
            
            if pose_np.shape[0] < 7:
                default_pose = torch.tensor([0.055, -1.74, 1.665, 1.233, -0.077, -0.17], 
                                          dtype=torch.float32, device=self._sim_device)
                return default_pose
                
            curr_pos = pose_np[:3]
            curr_quat = pose_np[3:7]  # w, x, y, z
        except Exception:
            default_pose = torch.tensor([0.055, -1.74, 1.665, 1.233, -0.077, -0.17], 
                                      dtype=torch.float32, device=self._sim_device)
            return default_pose
        
        # Initialize on first valid reading
        if not self._initialized and self._leader_arm.is_connected:
            self._prev_pos = curr_pos.copy()
            self._prev_quat = curr_quat.copy()
            self._initialized = True
        
        # Compute delta position
        delta_pos = (curr_pos - self._prev_pos) * self._pos_sensitivity
        
        # Compute delta rotation (quaternion difference to axis-angle)
        try:
            # Convert quaternions to rotation objects
            # Isaac uses w,x,y,z but scipy uses x,y,z,w
            prev_rot = Rotation.from_quat([
                self._prev_quat[1], self._prev_quat[2], 
                self._prev_quat[3], self._prev_quat[0]
            ])
            curr_rot = Rotation.from_quat([
                curr_quat[1], curr_quat[2], 
                curr_quat[3], curr_quat[0]
            ])
            
            # Delta rotation = curr * prev^-1
            delta_rot = (curr_rot * prev_rot.inv()).as_rotvec()
            delta_rot = delta_rot * self._rot_sensitivity
        except Exception:
            delta_rot = np.zeros(3)
        
        # Update previous state
        self._prev_pos = curr_pos.copy()
        self._prev_quat = curr_quat.copy()
        
        # Gripper command: gripper_pos is 0~1, map to binary
        self._close_gripper = gripper_pos < 0.5
        
        # Build command tensor
        command = np.concatenate([delta_pos, delta_rot])
        
        if self._gripper_term:
            gripper_value = -1.0 if self._close_gripper else 1.0
            command = np.append(command, gripper_value)
        
        return torch.tensor(command, dtype=torch.float32, device=self._sim_device)
    
    @property
    def is_connected(self) -> bool:
        """Check if leader arm is connected."""
        if self._leader_arm is None:
            return False
        return self._leader_arm.is_connected
