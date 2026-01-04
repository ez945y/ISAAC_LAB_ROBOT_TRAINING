# Isaac Lab Practice

This is a personal practice project for learning [Isaac Lab](https://isaac-sim.github.io/IsaacLab/).

## Overview

This repository serves as a workspace to explore Isaac Lab features, create custom environments, and experiment with robot simulation.

## Requirements

- **OS**: Linux (Ubuntu 24.04)
- **Python**: 3.11
- **NVIDIA Isaac Sim**: 5.1+
- **Isaac Lab**: 2.3.0+

## Directory Structure

```
robot/
├── projects/                    # Isaac Lab extension projects
│   ├── mimic_envs/              # Imitation learning environment
│   ├── rl_direct_hand_locomotion/   # Hand control RL
│   └── rl_manager_a1_navigation/    # A1 navigation RL
├── tools/                       # Utility scripts and modules
│   ├── controll_scripts/        # Robot control and assets
│   ├── 06_teleoperate.py        # Leader arm sender
│   └── convert_hdf5_to_lerobot.py
└── scripts/          # Quick experiment scripts
```

## Projects

All Isaac Lab extension projects are located in the `projects/` directory.

### [projects](./projects)

- **[mimic_envs](./projects/mimic_envs)**: A specialized environment for recording demonstrations (Imitation Learning).
  - Configures SO-ARM robot for stacking tasks.
  - Supports Leader Arm teleoperation.
  - Recording data to HDF5 for training.
  - Detailed setup instructions in [projects/mimic_envs/README.md](./projects/mimic_envs/README.md).

- **[rl_direct_hand_locomotion](./projects/rl_direct_hand_locomotion)**: A custom Isaac Lab extension for robotic hand control experiments. Uses Direct-based RL with RSL-RL PPO training, modified from the Ant locomotion task.

- **[rl_manager_a1_navigation](./projects/rl_manager_a1_navigation)**: A custom Isaac Lab extension for Unitree A1 robot waypoint navigation. Uses Manager-based RL with RSL-RL PPO training, modified from a1 locomotion task.

### Standalone Scripts

- **[scripts](./scripts)**: Independent Isaac Lab experiment scripts for quick prototyping and learning.
  - `01_basic_auto_drive.py` - Basic scene with Jetbot auto-driving
  - `02_keyboard_control.py` - WASD keyboard control with command smoothing
  - `03_domino_fpv.py` - Domino physics + first-person camera following
  - `04_trajectory_record.py` - Full-featured: trajectory recording/playback, dominoes, deformable objects
  - `05_robot_demo.py` - SO-ARM-101 robot arm demo with keyboard control
  - `06_teleoperate_demo.py` - **Real-to-Sim Teleoperation** receiver (runs on Ubuntu)

### Tools

- **[tools](./tools)**: Utility scripts, robot control modules, and assets.
  - `controll_scripts/` - Robot control modules and USD model (SO-ARM-101)
  - `06_teleoperate.py` - **Real-to-Sim Teleoperation** sender (runs on Mac with LeRobot)
  - `06_teleop_processors.py` - Supporting module for teleoperation
  - `convert_hdf5_to_lerobot.py` - Convert Isaac Lab HDF5 demos to LeRobot dataset format

### Real-to-Sim Teleoperation

Control a simulated SO-ARM-101 robot in Isaac Sim using a physical leader arm:
- **Mac** runs `tools/06_teleoperate.py` (requires [LeRobot](https://github.com/huggingface/lerobot)) to read physical arm joint positions
- **Ubuntu** runs `scripts/06_teleoperate_demo.py` to receive joint data and mirror movements in simulation
- Supports direct joint control mode with normalized joint positions (0~1)
- See [tools/README.md](./tools/README.md) for detailed setup instructions
- **Integration with Isaac Mimic**: This teleoperation setup is fully integrated with **[projects/mimic_envs](./projects/mimic_envs)**. You can use the same Leader Arm setup to record structured task demonstrations (Stacking Task) using `record_demos.py` as described in the Mimic Envs section.

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/ez945y/ISAAC_LAB_REBOT_TRAINING.git
cd ISAAC_LAB_REBOT_TRAINING
```

### 2. Install Project Dependencies
Each RL project is a standalone Isaac Lab extension. To install one (e.g., `rl_manager_a1_navigation`), navigate to its directory and install in editable mode:

```bash
cd projects/rl_manager_a1_navigation
python -m pip install -e source/rl_manager_a1_navigation
```

### 3. Run Standalone Scripts
The standalone scripts can be run directly without installation:

```bash
python scripts/04_trajectory_record.py
```

## License

BSD-3-Clause
