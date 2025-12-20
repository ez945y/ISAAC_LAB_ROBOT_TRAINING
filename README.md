# Isaac Lab Practice

This is a personal practice project for learning [Isaac Lab](https://isaac-sim.github.io/IsaacLab/).

## Overview

This repository serves as a workspace to explore Isaac Lab features, create custom environments, and experiment with robot simulation.

## Requirements

- **OS**: Linux (Ubuntu 24.04)
- **Python**: 3.11
- **NVIDIA Isaac Sim**: 5.0+
- **Isaac Lab**: 2.2+

## Projects

Currently, the repository contains:

- **[rl_direct_hand_locomotion](./rl_direct_hand_locomotion)**: A custom Isaac Lab extension for robotic hand control experiments. modified from ant locomotion task.

- **[rl_manager_a1_navigation](./rl_manager_a1_navigation)**: A custom Isaac Lab extension for robotic navigation experiments. modified from a1 locomotion task.

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/ez945y/ISAAC_LAB_REBOT_TRAINING.git
cd ISAAC_LAB_REBOT_TRAINING
```

### 2. Install Project Dependencies
Each project is a standalone Isaac Lab extension. To install one (e.g., `rl_direct_hand_locomotion`), navigate to its directory and install in editable mode:

```bash
cd rl_direct_hand_locomotion
python -m pip install -e source/rl_direct_hand_locomotion
```
