# SO-ARM-101 Mimic Environment Setup

## Required Symlink Setup

Before using this mimic environment, the following symlinks need to be created in the `isaaclab_mimic/isaaclab_mimic/` directory:

### 1. controll_scripts Symlink

```bash
# Create controll_scripts symlink
ln -s /home/rst_spark/ISAAC_LAB_REBOT_TRAINING/controll_scripts /home/rst_spark/IsaacLab/source/isaaclab_mimic/isaaclab_mimic/controll_scripts
```

This symlink allows Isaac Lab to find:
- SO-ARM-101 USD model file (`so_arm_101/SO-ARM101.usd`)
- Se3LeaderArm teleoperation device (`input_devices/se3_leader_arm.py`)

### 2. so_arm_stack_joint_mimic_env.py and so_arm_stack_joint_mimic_env_cfg.py Hard Links

These files are connected to `isaaclab_mimic/isaaclab_mimic/envs/` directory via hard links:

```bash
# Create hard links (if not already created)
ln /home/rst_spark/ISAAC_LAB_REBOT_TRAINING/mimic_envs/so_arm_stack_joint_mimic_env.py \
   /home/rst_spark/IsaacLab/source/isaaclab_mimic/isaaclab_mimic/envs/so_arm_stack_joint_mimic_env.py

ln /home/rst_spark/ISAAC_LAB_REBOT_TRAINING/mimic_envs/so_arm_stack_joint_mimic_env_cfg.py \
   /home/rst_spark/IsaacLab/source/isaaclab_mimic/isaaclab_mimic/envs/so_arm_stack_joint_mimic_env_cfg.py
```

## File Structure

After setup, the structure should look like this:

```
source/isaaclab_mimic/
├── isaaclab_mimic/
│   ├── controll_scripts/  -> symlink to ISAAC_LAB_REBOT_TRAINING/controll_scripts
│   │   ├── input_devices/
│   │   │   ├── se3_leader_arm.py
│   │   │   └── leader_arm.py
│   │   └── so_arm_101/
│   │       └── SO-ARM101.usd
│   └── envs/
│       ├── __init__.py  (contains environment registration)
│       ├── so_arm_stack_joint_mimic_env.py  (hard link)
│       └── so_arm_stack_joint_mimic_env_cfg.py  (hard link)
└── robot/  -> symlink to ISAAC_LAB_REBOT_TRAINING (for editing and viewing)
```

## Usage

After setup is complete, you can run the following command:

```bash
LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1" ./isaaclab.sh -p scripts/tools/record_demos.py \
    --task Isaac-PickPlace-SOArm-Joint-Mimic-v0 \
    --teleop_device leader_arm \
    --dataset_file ./datasets/so_arm_demos.hdf5
```
