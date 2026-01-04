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

This script reads joint data from a SO-101 leader arm and sends it to a socket.
Uses pluggable processors for different control modes (joint/ee).

Usage:
    python 07_lerobot_teleoperate.py \\
        --teleop.type=so101_leader \\
        --teleop.port=/dev/tty.usbmodem5AA90244081 \\
        --teleop.id=my_awesome_leader_arm

Modes:
    - joint (default): Send normalized joint positions directly.
    - ee: Compute forward kinematics and send end-effector pose.
"""

import json
import logging
import socket
import time
from dataclasses import asdict, dataclass
from pprint import pformat

from lerobot.configs import parser
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
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

# Import the refactored processors
from lerobot.scripts.teleop_processors import create_processor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TeleoperatePortConfig:
    """Configuration for leader arm teleoperation with socket output."""
    teleop: TeleoperatorConfig
    # Socket configuration (connects to simulation server)
    socket_host: str = '127.0.0.1'  # Target server host
    socket_port: int = 5359
    # Physical robot configuration (optional)
    robot: RobotConfig | None = None
    # Control mode: 'joint' for direct joint control, 'ee' for FK-based EE control
    control_mode: str = 'joint'
    # Path to URDF file for forward kinematics (only used in 'ee' mode)
    urdf_path: str | None = None
    # Limit the maximum frames per second
    fps: int = 60
    teleop_time_s: float | None = None


class SocketClient:
    """Socket client for connecting to simulation server."""

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
    socket_conn: SocketClient,
    processor,
    fps: int,
    robot: Robot | None = None,
    duration: float | None = None,
):
    """
    Main teleoperation loop.

    Args:
        teleop: The teleoperator device instance.
        socket_conn: Socket server or client for sending data.
        processor: The data processor (JointModeProcessor or FKModeProcessor).
        fps: Target frequency for the control loop.
        duration: Maximum duration in seconds. If None, runs indefinitely.
    """
    logger.info(f"Starting teleoperation loop at {fps} fps (mode: {processor.mode})")
    if robot:
        logger.info(f"Physical robot control ENABLED: {robot.__class__.__name__}")
    logger.info("Press Ctrl+C to stop")

    start = time.perf_counter()

    while True:
        loop_start = time.perf_counter()

        # Get teleop action (joint positions)
        action = teleop.get_action()

        # Process data using the configured processor
        data = processor.process(action)

        # Send data to connected clients
        socket_conn.send(data)

        # Send action to physical robot if enabled
        if robot:
            robot.send_action(action)

        # Display current state (overwrite same line)
        if processor.mode == 'joint':
            joints_str = ', '.join([f'{j:.2f}' for j in data['joints']])
            msg = f"Joints (0-1): [{joints_str}]"
        else:
            msg = f"EE: [{data['x']:.3f}, {data['y']:.3f}, {data['z']:.3f}]"
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

    # Create data processor based on mode
    processor = create_processor(
        mode=cfg.control_mode,
        urdf_path=cfg.urdf_path,
    )
    logger.info(f"Using {processor.mode.upper()} mode processor")

    # Create socket connection (always client mode)
    socket_conn = SocketClient(host=cfg.socket_host, port=cfg.socket_port)
    socket_conn.connect()

    # Create physical robot if configured
    robot = None
    if cfg.robot:
        robot = make_robot_from_config(cfg.robot)
        robot.connect()

    try:
        teleoperate_port_loop(
            teleop=teleop,
            socket_conn=socket_conn,
            processor=processor,
            fps=cfg.fps,
            robot=robot,
            duration=cfg.teleop_time_s,
        )
    except KeyboardInterrupt:
        print()  # New line after the \r output
        logger.info("Interrupted by user")
    finally:
        socket_conn.stop()
        teleop.disconnect()
        if robot:
            robot.disconnect()
        logger.info("Cleanup complete")


def main():
    register_third_party_plugins()
    teleoperate_port()


if __name__ == "__main__":
    main()
