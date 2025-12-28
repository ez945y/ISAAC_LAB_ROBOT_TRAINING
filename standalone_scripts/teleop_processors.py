# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""
Teleoperation Data Processors.

This module provides different strategies for processing leader arm data
before sending it to the simulation. Each processor handles normalization
and data formatting differently based on the control mode.

Classes:
    BaseTeleoperationProcessor: Abstract base class for all processors.
    JointModeProcessor: Sends normalized joint positions directly.
    FKModeProcessor: Computes forward kinematics and sends EE pose.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class JointLimits:
    """Joint limits configuration in degrees."""
    shoulder_pan: Tuple[float, float] = (-110.0, 110.0)
    shoulder_lift: Tuple[float, float] = (-100.0, 100.0)
    elbow_flex: Tuple[float, float] = (-97.0, 97.0)
    wrist_flex: Tuple[float, float] = (-95.0, 95.0)
    wrist_roll: Tuple[float, float] = (-90.0, 90.0)  # Reduced to half rotation

    def as_list(self) -> List[Tuple[float, float]]:
        """Return limits as ordered list."""
        return [
            self.shoulder_pan,
            self.shoulder_lift,
            self.elbow_flex,
            self.wrist_flex,
            self.wrist_roll,
        ]


class BaseTeleoperationProcessor(ABC):
    """
    Abstract base class for teleoperation data processing.

    Subclasses implement different strategies for converting raw joint data
    from the leader arm into a format suitable for the simulation.
    """

    def __init__(self, joint_limits: Optional[JointLimits] = None):
        """
        Initialize the processor.

        Args:
            joint_limits: Custom joint limits. If None, uses defaults.
        """
        self.joint_limits = joint_limits or JointLimits()

    @property
    @abstractmethod
    def mode(self) -> str:
        """Return the mode identifier ('joint' or 'ee')."""
        pass

    @abstractmethod
    def process(self, action: Dict[str, float]) -> Dict:
        """
        Process raw action data from teleoperator.

        Args:
            action: Dictionary with keys like 'shoulder_pan.pos', etc.

        Returns:
            Dict: Processed data packet ready to send over socket.
        """
        pass

    def _extract_joint_angles(self, action: Dict[str, float]) -> np.ndarray:
        """Extract joint angles in degrees from action dict."""
        return np.array([
            action['shoulder_pan.pos'],
            action['shoulder_lift.pos'],
            action['elbow_flex.pos'],
            action['wrist_flex.pos'],
            action['wrist_roll.pos'],
            action['gripper.pos'],
        ])

    def _normalize_joints(self, joint_angles_deg: np.ndarray) -> List[float]:
        """Normalize joint angles to 0~1 range."""
        limits = self.joint_limits.as_list()
        normalized = []
        for i in range(5):
            low, high = limits[i]
            norm = (joint_angles_deg[i] - low) / (high - low)
            norm = max(0.0, min(1.0, norm))
            normalized.append(norm)
        return normalized

    def _normalize_gripper(self, gripper_value: float) -> float:
        """Normalize gripper value (0-100 -> 0-1)."""
        return max(0.0, min(1.0, gripper_value / 100.0))


class JointModeProcessor(BaseTeleoperationProcessor):
    """
    Processor for direct joint position control.

    Normalizes joint positions to 0~1 range and sends them directly.
    The simulation denormalizes them using its own joint limits.
    """

    @property
    def mode(self) -> str:
        return 'joint'

    def process(self, action: Dict[str, float]) -> Dict:
        """Process action into normalized joint data packet."""
        joint_angles_deg = self._extract_joint_angles(action)
        joint_normalized = self._normalize_joints(joint_angles_deg)
        gripper_normalized = self._normalize_gripper(joint_angles_deg[5])

        # Build data packet
        data = {
            'mode': self.mode,
            'joints': [float(j) for j in joint_normalized] + [float(gripper_normalized)],
            'timestamp': time.time(),
        }

        # Backward compatibility keys
        joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
        for i, name in enumerate(joint_names):
            data[name] = float(joint_normalized[i])
        data['gripper'] = float(gripper_normalized)

        return data


class FKModeProcessor(BaseTeleoperationProcessor):
    """
    Processor for end-effector pose control using forward kinematics.

    Computes the end-effector position and orientation from joint angles
    using the robot's URDF model.
    """

    def __init__(
        self,
        urdf_path: Optional[str] = None,
        joint_limits: Optional[JointLimits] = None,
    ):
        """
        Initialize FK processor with URDF.

        Args:
            urdf_path: Path to URDF file. If None, auto-discovers.
            joint_limits: Custom joint limits.
        """
        super().__init__(joint_limits)
        self.urdf_path = urdf_path
        self._kinematics = None
        self._init_kinematics()

    def _init_kinematics(self):
        """Initialize the kinematics solver."""
        import os
        from lerobot.model.kinematics import RobotKinematics

        urdf_path = self.urdf_path

        # Auto-discover URDF if not provided
        if urdf_path is None:
            possible_paths = [
                os.path.expanduser('~/robot_config/so101_new_calib.urdf'),
                os.path.expanduser('~/robot/controll_scripts/so_arm_101/description/so101_new_calib.urdf'),
                os.path.join(os.getcwd(), 'so101_new_calib.urdf'),
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    urdf_path = p
                    break

            if urdf_path is None:
                raise FileNotFoundError(
                    'Could not find SO-101 URDF. Please provide urdf_path explicitly.'
                )

        logger.info(f'Loading URDF from: {urdf_path}')

        urdf_joint_names = [
            'shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll'
        ]

        # Try different target frames
        target_frames = ['gripper_frame_link', 'gripper_link', 'wrist_link']
        for frame in target_frames:
            try:
                self._kinematics = RobotKinematics(
                    urdf_path=urdf_path,
                    target_frame_name=frame,
                    joint_names=urdf_joint_names,
                )
                logger.info(f'Using target frame: {frame}')
                break
            except Exception as e:
                logger.debug(f'Could not use "{frame}" frame: {e}')
        else:
            raise RuntimeError('Failed to create kinematics solver')

    @property
    def mode(self) -> str:
        return 'ee'

    def process(self, action: Dict[str, float]) -> Dict:
        """Process action into end-effector pose data packet."""
        joint_angles_deg = self._extract_joint_angles(action)

        # Compute FK
        T = self._kinematics.forward_kinematics(joint_angles_deg[:5])
        position = T[:3, 3]
        quaternion = self._rotation_matrix_to_quaternion(T[:3, :3])

        # Normalize gripper
        gripper_normalized = self._normalize_gripper(joint_angles_deg[5])

        data = {
            'mode': self.mode,
            'x': float(position[0]),
            'y': float(position[1]),
            'z': float(position[2]),
            'qw': float(quaternion[0]),
            'qx': float(quaternion[1]),
            'qy': float(quaternion[2]),
            'qz': float(quaternion[3]),
            'gripper': float(gripper_normalized * 100.0),  # Back to 0-100 for EE mode
            'timestamp': time.time(),
        }

        return data

    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return np.array([w, x, y, z])


def create_processor(
    mode: str = 'joint',
    urdf_path: Optional[str] = None,
    joint_limits: Optional[JointLimits] = None,
) -> BaseTeleoperationProcessor:
    """
    Factory function to create the appropriate processor.

    Args:
        mode: 'joint' for direct joint control, 'ee' for FK-based EE control.
        urdf_path: Path to URDF (only needed for 'ee' mode).
        joint_limits: Custom joint limits.

    Returns:
        BaseTeleoperationProcessor: The configured processor instance.
    """
    if mode == 'joint':
        return JointModeProcessor(joint_limits=joint_limits)
    elif mode == 'ee':
        return FKModeProcessor(urdf_path=urdf_path, joint_limits=joint_limits)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'joint' or 'ee'.")
