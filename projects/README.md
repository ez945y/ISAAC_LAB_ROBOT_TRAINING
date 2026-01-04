# Robot Projects

This directory contains Isaac Lab extension projects for robot learning experiments.

## Projects

### [mimic_envs](./mimic_envs)

**SO-ARM-101 Imitation Learning Environment**

A specialized Isaac Lab environment for recording demonstrations (Imitation Learning) with the SO-ARM robot.

- **Task**: 3-cube stacking task
- **Control**: Leader arm teleoperation (Real-to-Sim)
- **Output**: HDF5 demonstrations for training
- **Integration**: Isaac Lab Mimic framework

See [mimic_envs/README.md](./mimic_envs/README.md) for setup instructions.

---

### [rl_direct_hand_locomotion](./rl_direct_hand_locomotion)

**Robotic Hand Control with Direct RL**

A custom Isaac Lab extension for robotic hand control experiments.

- **Method**: Direct-based RL
- **Training**: RSL-RL PPO
- **Base**: Modified from Ant locomotion task

---

### [rl_manager_a1_navigation](./rl_manager_a1_navigation)

**Unitree A1 Waypoint Navigation**

A custom Isaac Lab extension for Unitree A1 robot waypoint navigation.

- **Method**: Manager-based RL
- **Training**: RSL-RL PPO
- **Base**: Modified from A1 locomotion task
