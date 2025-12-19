# Robot Simulation - Isaac Lab Extension

A custom Isaac Lab extension for robotic hand locomotion and balance control using Direct RL.

## Installation

1. Install [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)

2. Install this extension:
   ```bash
   python -m pip install -e source/rl_direct_hand_locomotion
   ```

## Available Tasks

| Task ID | Description |
|---------|-------------|
| `Isaac-Hand-Direct-v0` | Hand balance and locomotion task (12 DOF, 48-dim obs) |

## Usage

### Training
```bash
python scripts/rsl_rl/train.py --task=Isaac-Hand-Direct-v0
```

### Play Trained Policy
```bash
python scripts/rsl_rl/play.py --task=Isaac-Hand-Direct-v0
```

### Testing
```bash
# Zero-action agent
python scripts/zero_agent.py --task=Isaac-Hand-Direct-v0

# Random-action agent
python scripts/random_agent.py --task=Isaac-Hand-Direct-v0
```

## Project Structure

```
source/rl_direct_hand_locomotion/rl_direct_hand_locomotion/tasks/direct/hand/
├── hand_env.py          # Environment implementation
├── assets/
│   ├── hand_cfg.py      # Robot config
│   └── h1_hand_left.usd # Robot model
└── agents/
    └── rsl_rl_ppo_cfg.py # PPO training config
```

## License

BSD-3-Clause