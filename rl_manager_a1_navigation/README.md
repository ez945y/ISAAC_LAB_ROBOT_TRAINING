# RL Manager based A1 Navigation

A custom Isaac Lab extension for A1 robot navigation using Manager-based RL.

## Installation

### 1. Install Isaac Lab
Follow the [Isaac Lab Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

### 2. Install this Extension
Run the following command from this directory:

```bash
cd rl_manager_a1_navigation
python -m pip install -e source/rl_manager_a1_navigation
```

## Available Tasks

| Task ID | Description |
|---------|-------------|
| `Isaac-Waypoint-Navigation-v0` | A1 robot waypoint navigation task |

## Usage

### Training
```bash
python scripts/rsl_rl/train.py --task=Isaac-Waypoint-Navigation-v0
```

### Play Trained Policy
```bash
python scripts/rsl_rl/play.py --task=Isaac-Waypoint-Navigation-v0
```

### Testing
```bash
# Zero-action agent
python scripts/zero_agent.py --task=Isaac-Waypoint-Navigation-v0

# Random-action agent
python scripts/random_agent.py --task=Isaac-Waypoint-Navigation-v0
```

## Project Structure

```
source/rl_manager_a1_navigation/rl_manager_a1_navigation/tasks/manager_based/navigation/
├── navigation_env_cfg.py # Environment configuration
├── mdp/
│   ├── rewards.py        # Reward functions
│   └── command.py        # Command generators
├── assets/
│   └── unitree.py        # Robot configuration
└── agents/
    └── rsl_rl_ppo_cfg.py # PPO training config
```

## License

BSD-3-Clause