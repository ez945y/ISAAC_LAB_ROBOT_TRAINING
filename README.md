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

### Standalone Scripts

- **[standalone_scripts](./standalone_scripts)**: Independent Isaac Lab experiment scripts for quick prototyping and learning.
  - `01_basic_auto_drive.py` - Basic scene with Jetbot auto-driving
  - `02_keyboard_control.py` - WASD keyboard control with command smoothing
  - `03_domino_fpv.py` - Domino physics + first-person camera following
  - `04_trajectory_record.py` - Full-featured: trajectory recording/playback, dominoes, deformable objects

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
