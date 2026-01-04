# SO-ARM-101 Mimic Environment Setup

## Required Symlink Setup

The mimic environment requires a symlink in `isaaclab_mimic/isaaclab_mimic/` to access robot control modules.

### controll_scripts Symlink

This symlink should already be set up pointing to `robot/tools/controll_scripts`:

```bash
# Verify the symlink exists
ls -la /home/rst_spark/IsaacLab/source/isaaclab_mimic/isaaclab_mimic/controll_scripts

# If missing, create it:
ln -s ../robot/tools/controll_scripts /home/rst_spark/IsaacLab/source/isaaclab_mimic/isaaclab_mimic/controll_scripts
```

This symlink allows Isaac Lab to find:
- SO-ARM-101 USD model file (`so_arm_101/SO-ARM101.usd`)
- Se3LeaderArm teleoperation device (`input_devices/se3_leader_arm.py`)

## File Structure

After setup, the structure should look like this:

```
source/isaaclab_mimic/
├── isaaclab_mimic/
│   ├── controll_scripts/  -> symlink to ../robot/tools/controll_scripts
│   │   ├── input_devices/
│   │   │   ├── se3_leader_arm.py
│   │   │   └── leader_arm.py
│   │   └── so_arm_101/
│   │       └── SO-ARM101.usd
│   └── envs/
│       ├── __init__.py  (contains environment registration)
│       ├── so_arm_stack_joint_mimic_env.py
│       └── so_arm_stack_joint_mimic_env_cfg.py
└── robot/
    ├── projects/
    │   ├── mimic_envs/        <- You are here
    │   ├── rl_direct_hand_locomotion/
    │   └── rl_manager_a1_navigation/
    ├── tools/
    │   ├── controll_scripts/  (actual location)
    │   ├── 06_teleoperate.py
    │   └── convert_hdf5_to_lerobot.py
    └── scripts/
```

## Configuration for Leader Arm Teleoperation

To use the `leader_arm` device, you need to update Isaac Lab's device factory. We have provided a modified file for this purpose.

Run the following command to overwrite the default factory file:

```bash
cp /home/rst_spark/IsaacLab/source/isaaclab_mimic/robot/projects/mimic_envs/teleop_device_factory_modified.py /home/rst_spark/IsaacLab/source/isaaclab/isaaclab/devices/teleop_device_factory.py
```

## Usage

After setup is complete, you can run the following command:

```bash
LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1" ./isaaclab.sh -p scripts/tools/record_demos.py \
    --task Isaac-PickPlace-SOArm-Joint-Mimic-v0 \
    --teleop_device leader_arm \
    --dataset_file ./datasets/so_arm_demos.hdf5
```
