# Isaac Lab Practice

This is a personal practice project for learning [Isaac Lab](https://isaac-sim.github.io/IsaacLab/).

## Overview

This repository serves as a workspace to explore Isaac Lab features, create custom environments, and experiment with robot simulation.

## Requirements

- **OS**: Linux (Ubuntu 24.04)
- **Python**: 3.11
- **NVIDIA Isaac Sim**: 5.1+
- **Isaac Lab**: 2.3.0+

## Projects

Currently, the repository contains:

### RL Extensions

- **[rl_direct_hand_locomotion](./rl_direct_hand_locomotion)**: A custom Isaac Lab extension for robotic hand control experiments. Uses Direct-based RL with RSL-RL PPO training, modified from the Ant locomotion task.

- **[rl_manager_a1_navigation](./rl_manager_a1_navigation)**: A custom Isaac Lab extension for Unitree A1 robot waypoint navigation. Uses Manager-based RL with RSL-RL PPO training, modified from a1 locomotion task.

### Mimic Envs (Demonstration Recording)

- **[mimic_envs](./mimic_envs)**: A specialized environment for recording demonstrations (Imitation Learning).
  - Configures SO-ARM robot for stacking tasks.
  - Supports Leader Arm teleoperation.
  - Recording data to HDF5 for training.
  - Detailed setup instructions in [mimic_envs/README.md](./mimic_envs/README.md).

### Standalone Scripts

- **[standalone_scripts](./standalone_scripts)**: Independent Isaac Lab experiment scripts for quick prototyping and learning.
  - `01_basic_auto_drive.py` - Basic scene with Jetbot auto-driving
  - `02_keyboard_control.py` - WASD keyboard control with command smoothing
  - `03_domino_fpv.py` - Domino physics + first-person camera following
  - `04_trajectory_record.py` - Full-featured: trajectory recording/playback, dominoes, deformable objects
  - `05_robot_demo.py` - SO-ARM-101 robot arm demo with keyboard control
  - `06_teleoperate_demo.py` - **Real-to-Sim Teleoperation** receiver (runs on Ubuntu)
  - `06_teleoperate.py, 06_teleop_processors.py` - **Real-to-Sim Teleoperation** sender (runs on Mac with LeRobot)

### Real-to-Sim Teleoperation

Control a simulated SO-ARM-101 robot in Isaac Sim using a physical leader arm:
- **Mac** runs `06_teleoperate.py` (requires [LeRobot](https://github.com/huggingface/lerobot)) to read physical arm joint positions
- **Ubuntu** runs `06_teleoperate_demo.py` to receive joint data and mirror movements in simulation
- Supports direct joint control mode with normalized joint positions (0~1)
- See [standalone_scripts/README.md](./standalone_scripts/README.md) for detailed setup instructions
- **Integration with Isaac Mimic**: This teleoperation setup is fully integrated with **[mimic_envs](./mimic_envs)**. You can use the same Leader Arm setup to record structured task demonstrations (Stacking Task) using `record_demos.py` as described in the Mimic Envs section.

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/ez945y/ISAAC_LAB_REBOT_TRAINING.git
cd ISAAC_LAB_REBOT_TRAINING
```

### 2. Install Project Dependencies
Each RL project is a standalone Isaac Lab extension. To install one (e.g., `rl_manager_a1_navigation`), navigate to its directory and install in editable mode:

```bash
cd rl_manager_a1_navigation
python -m pip install -e source/rl_manager_a1_navigation
```

### 3. Run Standalone Scripts
The standalone scripts can be run directly without installation:

```bash
python standalone_scripts/04_trajectory_record.py
```

## License

BSD-3-Clause
