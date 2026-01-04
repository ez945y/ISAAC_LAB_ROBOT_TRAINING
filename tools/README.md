# Robot Tools

This directory contains utility scripts for working with robot data, models, and teleoperation.

## Directory Structure

```
tools/
├── controll_scripts/           # Robot control modules and assets
│   ├── input_devices/          # Teleoperation device implementations
│   │   ├── se3_leader_arm.py   # Isaac Lab compatible leader arm device
│   │   └── leader_arm.py       # Base leader arm implementation
│   ├── so_arm_101/             # SO-ARM-101 robot assets
│   │   └── SO-ARM101.usd       # Robot USD model
│   ├── controllers/            # Robot controllers
│   └── configs/                # Configuration files
├── teleoperate_port.py           # Leader arm sender (Mac side)
├── teleop_processors.py     # Teleoperation support module
├── convert_hdf5_to_lerobot.py  # HDF5 to LeRobot converter
└── README.md
```

## controll_scripts

The `controll_scripts` directory contains the core robot control modules and assets:

- **input_devices/**: Device implementations for teleoperation
  - `se3_leader_arm.py`: Isaac Lab `DeviceBase` compatible wrapper for leader arm
  - `leader_arm.py`: Base implementation for reading physical leader arm data
  
- **so_arm_101/**: SO-ARM-101 robot model and configurations
  - `SO-ARM101.usd`: The robot's USD model file used in Isaac Sim

**Note:** This directory is symlinked to `isaaclab_mimic/isaaclab_mimic/controll_scripts` to enable imports like `from isaaclab_mimic.controll_scripts.input_devices import Se3LeaderArm`.

---

## Scripts

### Teleoperation (Real-to-Sim)

#### teleoperate_port.py (Mac/LeRobot Side)

Reads joint positions from a physical leader arm using LeRobot and sends them over network.

**Requirements:**
- macOS with LeRobot installed
- Physical SO-ARM leader arm connected

**Usage:**
```bash
python teleoperate_port.py
```

#### teleop_processors.py (Supporting Module)

Data processing utilities for teleoperation. Used by `teleoperate_port.py`.

---

### Data Regeneration

#### regenerate_demos.py

Replays actions from an existing HDF5 dataset in a new environment configuration and records new observations. This is useful for:
- Tuning physics parameters (e.g., stiffness) without re-recording.
- Generating visual observations (rendering images) from purely state-based recordings.

**Usage:**
```bash
./isaaclab.sh -p scripts/tools/regenerate_demos.py \
    --task [Target Task Name] \
    --input_file [Source HDF5] \
    --output_file [Output HDF5] \
    [--enable_cameras]
```

---

### Data Conversion

#### convert_hdf5_to_lerobot.py

Converts Isaac Lab HDF5 demonstration files to LeRobot dataset format for training imitation learning models.

#### Requirements

```bash
pip install h5py numpy
pip install lerobot  # Optional, for full LeRobot format support
```

#### Usage

```bash
# Basic conversion (SO-ARM robot)
python convert_hdf5_to_lerobot.py \
    --input ./datasets/so_arm_demos.hdf5 \
    --output ./lerobot_datasets/so_arm_stack \
    --robot-type so_arm \
    --fps 30

# Explore HDF5 structure without converting
python convert_hdf5_to_lerobot.py --input ./datasets/so_arm_demos.hdf5 --explore-only

# Output simple numpy format (if LeRobot not installed)
python convert_hdf5_to_lerobot.py \
    --input ./datasets/so_arm_demos.hdf5 \
    --output ./numpy_data \
    --simple-format

# Push to Hugging Face Hub
python convert_hdf5_to_lerobot.py \
    --input ./datasets/so_arm_demos.hdf5 \
    --output ./lerobot_datasets/so_arm_stack \
    --repo-id your_username/so_arm_stack_sim \
    --push-to-hub
```

#### Supported Robot Types

- `so_arm`: SO-ARM-101 (6 DOF: 5 arm joints + 1 gripper)
- `franka`: Franka Emika Panda (9 DOF: 7 arm joints + 2 fingers)
- `generic`: Auto-detect dimensions from data

#### Output Formats

1. **LeRobot Format** (default): Full LeRobot dataset compatible with ACT, Diffusion Policy, etc.
2. **Simple Format** (`--simple-format`): NumPy arrays for custom training pipelines.

#### Data Mapping

| Isaac Lab Key | LeRobot Key | Description |
|---------------|-------------|-------------|
| `observations/policy/joint_pos` | `observation.state` | Robot joint positions |
| `actions` | `action` | Control commands (joint positions) |

#### Notes

- Ensure your Isaac Lab `ObservationsCfg` outputs raw physical units (radians, meters) for best compatibility.
- LeRobot will automatically compute normalization statistics during training.
- If using image observations, additional configuration is needed (see LeRobot documentation).
