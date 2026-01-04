# SO-ARM Mimic Extension

This extension provides environment configurations and tools for the SO-ARM 101 robot within the Isaac Lab Mimic framework.

## Installation

```bash
cd projects/so_arm_mimic
pip install -e .
```

## Directory Structure

```
so_arm_mimic/
├── scripts/                    # Utility scripts
│   ├── record_demos.py         # Data recording
│   ├── replay_demos.py         # Data replay
│   ├── regenerate_demos.py     # Data regeneration (visual)
│   ├── convert_hdf5_to_lerobot.py # Format conversion
│   └── view_port.py            # Camera utility
├── so_arm_mimic/               # Source package
│   ├── envs/                   # Environment definitions
│   └── controll_scripts/       # Robot assets & input devices
└── setup.py
```

## Usage

### 1. Recording Demonstrations

Record new demonstrations using a teleop device (e.g. Leader Arm).

```bash
python scripts/record_demos.py \
    --task Isaac-PickPlace-SOArm-Joint-Mimic-v0 \
    --teleop_device leader_arm \
    --num_demos 10 \
    --enable_cameras
```

### 2. Replaying Demonstrations

Replay recorded HDF5 demonstrations to verify correctness.

```bash
python scripts/replay_demos.py \
    --task Isaac-PickPlace-SOArm-Joint-Mimic-v0 \
    --dataset_file ./datasets/so_arm_demos.hdf5 \
    --enable_cameras
```

### 3. Regenerating for Visual Training

Replays actions from an existing HDF5 dataset in a new environment configuration and records new observations (e.g. images).

```bash
python scripts/regenerate_demos.py \
    --task Isaac-PickPlace-SOArm-Joint-Mimic-v0 \
    --input_file ./datasets/so_arm_demos.hdf5 \
    --output_file ./datasets/so_arm_demos_camera.hdf5 \
    --enable_cameras
```

### 4. Data Conversion (to LeRobot)

Converts Isaac Lab HDF5 demonstration files to LeRobot dataset format for training imitation learning models.

**Requirements:**
```bash
pip install h5py numpy
pip install lerobot  # Optional
```

**Usage:**
```bash
python scripts/convert_hdf5_to_lerobot.py \
    --input ./datasets/so_arm_demos.hdf5 \
    --output ./lerobot_datasets/so_arm_stack \
    --robot-type so_arm \
    --fps 30
```

| Isaac Lab Key | LeRobot Key | Description |
|---------------|-------------|-------------|
| `observations/policy/joint_pos` | `observation.state` | Robot joint positions |
| `actions` | `action` | Control commands (joint positions) |
