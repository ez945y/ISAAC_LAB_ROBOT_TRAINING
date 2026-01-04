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

## Configuration for Leader Arm Teleoperation

To use the `leader_arm` device, you need to update Isaac Lab's device factory:

```bash
cp /home/rst_spark/IsaacLab/source/isaaclab_mimic/robot/projects/mimic_envs/teleop_device_factory_modified.py \
   /home/rst_spark/IsaacLab/source/isaaclab/isaaclab/devices/teleop_device_factory.py
```

## Usage

### Step 1: Record Demonstrations

```bash
LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1" ./isaaclab.sh -p scripts/tools/record_demos.py \
    --task Isaac-PickPlace-SOArm-Joint-Mimic-v0 \
    --teleop_device leader_arm \
    --dataset_file ./datasets/so_arm_demos.hdf5 \
    --num_demos 10
```

**Recording Controls:**
- **Disconnect Leader Arm**: The robot holds its position, current episode is paused
- **Reconnect Leader Arm**: Triggers reset and **skips** the current episode, starts fresh
- The reset message will show how many demos have been successfully saved
- **Ctrl+C**: Exits gracefully, discarding any incomplete episode

### Step 2: Replay Demonstrations

```bash
./isaaclab.sh -p scripts/tools/replay_demos.py \
    --task Isaac-PickPlace-SOArm-Joint-Mimic-v0 \
    --dataset_file ./datasets/so_arm_demos.hdf5
```

**Note:** We added `import isaaclab_mimic.envs` to `replay_demos.py` so it can find our custom environment. 
If you're using a fresh Isaac Lab installation, you may need to add this import manually.

### Step 3: Annotate Subtasks

After recording, annotate demonstrations with subtask boundaries:

**Automatic Annotation (Recommended):**
```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
    --task Isaac-PickPlace-SOArm-Joint-Mimic-v0 \
    --input_file ./datasets/so_arm_demos.hdf5 \
    --output_file ./datasets/so_arm_demos_annotated.hdf5 \
    --auto
```

**Manual Annotation Controls:**
- **N**: Play/Resume
- **B**: Pause
- **S**: Mark subtask completion
- **Q**: Skip episode

The task has 3 subtask signals to mark:
1. `grasp_1` - When red cube is grasped
2. `stack_1` - When red cube is stacked on blue
3. `grasp_2` - When green cube is grasped

### Step 4: Convert to LeRobot Format (Optional)

```bash
python robot/tools/convert_hdf5_to_lerobot.py \
    --input ./datasets/so_arm_demos_annotated.hdf5 \
    --output ./lerobot_datasets/so_arm_stack \
    --robot-type so_arm
```

## Troubleshooting

### Environment Not Found Error

If you see `gymnasium.error.NameNotFound: Environment 'Isaac-PickPlace-SOArm-Joint-Mimic' doesn't exist`:

The script needs to import `isaaclab_mimic.envs` to register our custom environment.
We've already added this to `replay_demos.py`, but if using other scripts, add:

```python
import isaaclab_mimic.envs  # noqa: F401
```

after the `import isaaclab_tasks` line.

### Disconnect Auto-Reset

When the leader arm disconnects during recording:
- The current episode is **automatically skipped** (not saved)
- The environment resets to the initial state
- You can reconnect and continue recording

This is useful for discarding failed attempts without restarting the entire recording session.
