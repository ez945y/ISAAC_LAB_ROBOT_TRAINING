#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Leader Arm Teleoperate with Socket Output

This script reads joint data from a SO-101 leader arm, computes forward kinematics
to calculate the end-effector position (x, y, z) in meters, and sends it to a socket
on port 5359 (configurable).

Usage:
    lerobot-teleoperate_port \
        --teleop.type=so101_leader \
        --teleop.port=/dev/tty.usbmodem5AA90244081 \
        --teleop.id=my_awesome_leader_arm

Or run directly:
    python lerobot_teleoperate_port.py \
        --teleop.type=so101_leader \
        --teleop.port=/dev/tty.usbmodem5AA90244081 \
        --teleop.id=my_awesome_leader_arm

The data format sent over the socket is JSON:
{
    "x": float,      # End-effector X position in meters
    "y": float,      # End-effector Y position in meters
    "z": float,      # End-effector Z position in meters
    "qw": float,     # Quaternion w
    "qx": float,     # Quaternion x
    "qy": float,     # Quaternion y
    "qz": float,     # Quaternion z
    "gripper": float # Gripper position [0, 100]
}
"""

import json
import logging
import socket
import threading
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import numpy as np

from lerobot.configs import parser
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    gamepad,
    homunculus,
    keyboard,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TeleoperatePortConfig:
    """Configuration for leader arm teleoperation with socket output."""
    teleop: TeleoperatorConfig
    # Socket configuration
    socket_host: str = "0.0.0.0"  # Server mode: bind address; Client mode: target host
    socket_port: int = 5359
    # Client mode: connect to remote server instead of running local server
    client_mode: bool = True
    # Path to URDF file for forward kinematics
    urdf_path: str | None = None
    # Limit the maximum frames per second
    fps: int = 60
    teleop_time_s: float | None = None


class SO101ForwardKinematics:
    """
    Forward kinematics for SO-101 arm using placo library via RobotKinematics.

    Uses the actual URDF for accurate kinematics calculation.
    """

    def __init__(self, urdf_path: str | None = None):
        """
        Initialize the forward kinematics solver.

        Args:
            urdf_path: Path to the URDF file. If None, uses default SO-101 URDF.
        """
        import os
        from lerobot.model.kinematics import RobotKinematics

        # Default URDF path
        if urdf_path is None:
            # Try to find the URDF in various locations
            script_dir = os.path.dirname(os.path.abspath(__file__))
            lerobot_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
            cwd = os.getcwd()

            possible_paths = [
                # Current directory
                os.path.join(cwd, 'so101_new_calib.urdf'),
                os.path.join(cwd, 'so101_new_calib.urdf'),
                # Robot config directory
                os.path.expanduser('~/robot_config/so101_new_calib.urdf'),
                os.path.expanduser('~/robot_config/so101_new_calib.urdf'),
                # Relative to lerobot package
                os.path.join(lerobot_root, 'robots', 'so101', 'so101_new_calib.urdf'),
                os.path.join(lerobot_root, 'robots', 'so101', 'so101_new_calib.urdf'),
                # Common project structure
                os.path.expanduser('~/robot/controll_scripts/so_arm_101/description/so101_new_calib.urdf'),
                os.path.expanduser('~/robot/controll_scripts/so_arm_101/description/so101_new_calib.urdf'),
            ]

            urdf_path = None
            for p in possible_paths:
                if os.path.exists(p):
                    urdf_path = p
                    break

            if urdf_path is None:
                raise FileNotFoundError(
                    'Could not find SO-101 URDF. Please provide --urdf_path explicitly.\n'
                    'Example: --urdf_path=/path/to/so101_new_calib.urdf'
                )

        logger.info(f'Loading URDF from: {urdf_path}')

        # Joint names matching both teleoperator motor names and URDF joint names
        # These are identical in the so101_new_calib.urdf
        urdf_joint_names = [
            'shoulder_pan',
            'shoulder_lift',
            'elbow_flex',
            'wrist_flex',
            'wrist_roll',
        ]

        # Create kinematics solver
        # Try different target frames in order of preference
        target_frames = ['gripper_frame_link', 'gripper_link', 'wrist_link']
        last_error = None

        for frame in target_frames:
            try:
                self.kinematics = RobotKinematics(
                    urdf_path=urdf_path,
                    target_frame_name=frame,
                    joint_names=urdf_joint_names,
                )
                logger.info(f'Using target frame: {frame}')
                logger.info(f'Joint names: {self.kinematics.joint_names}')
                break
            except Exception as e:
                last_error = e
                logger.debug(f'Could not use "{frame}" frame: {e}')
        else:
            raise RuntimeError(f'Failed to create kinematics solver: {last_error}')

    def forward_kinematics(self, joint_angles_deg: np.ndarray) -> tuple:
        """
        Compute forward kinematics.

        Args:
            joint_angles_deg: Array of 6 joint angles in degrees
                [rotation, pitch, elbow, wrist_pitch, wrist_roll, gripper]

        Returns:
            Tuple of (position [x, y, z], quaternion [w, x, y, z])
        """
        # Get 4x4 transformation matrix from placo
        T = self.kinematics.forward_kinematics(joint_angles_deg[:5])

        # Extract position
        position = T[:3, 3]

        # Extract rotation matrix and convert to quaternion
        R = T[:3, :3]
        quaternion = self._rotation_matrix_to_quaternion(R)

        return position, quaternion

    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
        # Using Shepperd's method for numerical stability
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


class SocketServer:
    """Socket server for broadcasting pose data."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5359):
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.running = False
        self.data_lock = threading.Lock()
        self.accept_thread = None

    def start(self):
        """Start the socket server."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.server_socket.settimeout(1.0)
        self.running = True

        logger.info(f"Socket server listening on {self.host}:{self.port}")

        # Start client accept thread
        self.accept_thread = threading.Thread(target=self._accept_clients, daemon=True)
        self.accept_thread.start()

    def _accept_clients(self):
        """Accept client connections."""
        while self.running:
            try:
                client, addr = self.server_socket.accept()
                logger.info(f"Client connected from {addr}")
                with self.data_lock:
                    if self.client_socket:
                        try:
                            self.client_socket.close()
                        except Exception:
                            pass
                    self.client_socket = client
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"Error accepting client: {e}")

    def send(self, data: dict):
        """Send data to connected client."""
        with self.data_lock:
            if self.client_socket is None:
                return
            client = self.client_socket

        try:
            message = json.dumps(data) + "\n"
            client.sendall(message.encode('utf-8'))
        except (BrokenPipeError, ConnectionResetError, OSError):
            logger.warning("Client disconnected")
            with self.data_lock:
                self.client_socket = None

    def stop(self):
        """Stop the socket server."""
        self.running = False

        with self.data_lock:
            if self.client_socket:
                try:
                    self.client_socket.close()
                except Exception:
                    pass
                self.client_socket = None

        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass
            self.server_socket = None

        logger.info("Socket server stopped")


class SocketClient:
    """Socket client for connecting to remote server and sending pose data."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False

    def connect(self):
        """Connect to remote server."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            logger.info(f"Connecting to {self.host}:{self.port}...")
            self.socket.connect((self.host, self.port))
            self.connected = True
            logger.info(f"Connected to {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self.connected = False
            raise

    def send(self, data: dict):
        """Send data to server."""
        if not self.connected:
            return

        try:
            message = json.dumps(data) + "\n"
            self.socket.sendall(message.encode('utf-8'))
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            logger.warning(f"Connection lost: {e}")
            self.connected = False

    def stop(self):
        """Close connection."""
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None
        self.connected = False
        logger.info("Socket client stopped")


def teleoperate_port_loop(
    teleop: Teleoperator,
    socket_server: SocketServer,
    fps: int,
    duration: float | None = None,
):
    """
    Main teleoperation loop that reads leader arm and sends joint positions over socket.

    Args:
        teleop: The teleoperator device instance providing control actions.
        socket_server: Socket server for sending joint data.
        fps: The target frequency for the control loop in frames per second.
        duration: The maximum duration of the teleoperation loop in seconds. If None, runs indefinitely.
    """
    logger.info(f"Starting teleoperation loop at {fps} fps")
    logger.info("Press Ctrl+C to stop")

    start = time.perf_counter()

    while True:
        loop_start = time.perf_counter()

        # Get teleop action (joint positions)
        action = teleop.get_action()

        # Extract joint angles (in degrees)
        # Joint order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
        joint_angles_deg = np.array([
            action["shoulder_pan.pos"],
            action["shoulder_lift.pos"],
            action["elbow_flex.pos"],
            action["wrist_flex.pos"],
            action["wrist_roll.pos"],
            action["gripper.pos"],
        ])

        # Joint limits for normalization (in degrees)
        # These should match the physical range of the leader arm motors
        joint_limits_deg = [
            (-110.0, 110.0),   # shoulder_pan
            (-100.0, 100.0),   # shoulder_lift
            (-97.0, 97.0),     # elbow_flex
            (-95.0, 95.0),     # wrist_flex
            (-157.0, 163.0),   # wrist_roll
        ]

        # Normalize joint angles to 0~1
        joint_normalized = []
        for i in range(5):
            low, high = joint_limits_deg[i]
            # Standard normalization formula: (val - low) / (high - low)
            norm = (joint_angles_deg[i] - low) / (high - low)
            norm = max(0.0, min(1.0, norm))
            joint_normalized.append(norm)

        # Gripper is 0-100, normalize to 0-1
        gripper_normalized = joint_angles_deg[5] / 100.0
        gripper_normalized = max(0.0, min(1.0, gripper_normalized))

        # Prepare data packet - sending ordered list to be name-agnostic
        # joints[0:5] = arm ids 1-5, joints[5] = gripper id 6
        data = {
            "mode": "joint",
            "joints": [float(j) for j in joint_normalized] + [float(gripper_normalized)],
            "timestamp": time.time(),
        }

        # For backward compatibility with older receivers, also keep individual keys
        data.update({
            "shoulder_pan": float(joint_normalized[0]),
            "shoulder_lift": float(joint_normalized[1]),
            "elbow_flex": float(joint_normalized[2]),
            "wrist_flex": float(joint_normalized[3]),
            "wrist_roll": float(joint_normalized[4]),
            "gripper": float(gripper_normalized),
        })

        # Send data to connected clients
        socket_server.send(data)

        # Display current state (overwrite same line)
        msg = (
            f"Joints (1-6): [{', '.join([f'{j:.2f}' for j in data['joints']])}]"
        )
        print(f"\r{msg:<80}", end="", flush=True)

        # Maintain loop rate
        dt_s = time.perf_counter() - loop_start
        precise_sleep(1 / fps - dt_s)

        if duration is not None and time.perf_counter() - start >= duration:
            return


@parser.wrap()
def teleoperate_port(cfg: TeleoperatePortConfig):
    """Main entry point for leader arm teleoperation with socket output."""
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Create teleoperator
    teleop = make_teleoperator_from_config(cfg.teleop)
    teleop.connect()

    # Create socket connection (server or client mode)
    if cfg.client_mode:
        # Client mode: connect to remote server (e.g., Ubuntu)
        socket_conn = SocketClient(host=cfg.socket_host, port=cfg.socket_port)
        socket_conn.connect()
    else:
        # Server mode: wait for client to connect
        socket_conn = SocketServer(host=cfg.socket_host, port=cfg.socket_port)
        socket_conn.start()

    try:
        teleoperate_port_loop(
            teleop=teleop,
            socket_server=socket_conn,
            fps=cfg.fps,
            duration=cfg.teleop_time_s,
        )
    except KeyboardInterrupt:
        print()  # New line after the \r output
        logger.info("Interrupted by user")
    finally:
        socket_conn.stop()
        teleop.disconnect()
        logger.info("Cleanup complete")


def main():
    register_third_party_plugins()
    teleoperate_port()


if __name__ == "__main__":
    main()
