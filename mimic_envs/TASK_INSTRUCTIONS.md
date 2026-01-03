# SO-ARM Stack Task Instructions

This document outlines how to successfully complete the `Isaac-PickPlace-SOArm-Joint-Mimic-v0` stacking task for demo recording.

## Goal
Stack three colored cubes in a specific order: Green on Red, and Red on Blue.

## Cubes Setup
There are three colored cubes in the workspace:

1. 🔵 **Blue Cube** (`cube_1`): The base cube. Do not move this one.
   - Position: Fixed on the ground (target base).
2. 🔴 **Red Cube** (`cube_2`): The middle cube.
3. 🟢 **Green Cube** (`cube_3`): The top cube.

## Task Sequence (How to record a successful demo)

Follow these steps precisely. The environment will only register a "Success" and automatically reset if **ALL** conditions are met.

### Step 1: Grasp Red Cube
- Move the robot to the **Red Cube**.
- Close the gripper to grasp it firmly.

### Step 2: Stack Red on Blue
- Lift the **Red Cube**.
- Place it carefully on top of the **Blue Cube**.
- **Crucial:** Release the gripper completely. The cube must be stable.

### Step 3: Grasp Green Cube
- Move the robot to the **Green Cube**.
- Close the gripper to grasp it firmly.

### Step 4: Stack Green on Red
- Lift the **Green Cube**.
- Place it carefully on top of the **Red Cube** (which is already on the Blue Cube).
- **CRITICAL STEP:** **Open the gripper fully.**
  - The environment checks if the gripper is open to confirm you are done.
  - If you hold the cube, it will NOT count as success.

## Success Criteria (Technical)
The episode will strictly check for:
1. **Stack 1 OK:** Red cube is on Blue cube (XY distance < 5cm, Height diff ≈ 5cm).
2. **Stack 2 OK:** Green cube is on Red cube (XY distance < 5cm, Height diff ≈ 5cm).
3. **Gripper Open:** The gripper must be fully open (value > 1.45).

## Automatic Reset
- Once all success criteria are met, the environment will automatically reset for the next demo.
- If any cube drops on the floor (height < -0.05m), the environment might reset (if enabled).

## Recording Settings
- **Default:** Unlimited demos. Press `Ctrl+C` to stop.
- **Set Limit:** Run with `--num_demos N` (e.g., `... --num_demos 10`).

## Troubleshooting
- **Cubes sliding off?** Try to align them more perfectly centered.
- **Not resetting?** Make sure you open the gripper **ALL THE WAY** after placing the last cube.

## System Configuration (Required for Leader Arm)
To use the `leader_arm` device, overwrite the default device factory with the provided modified version:

```bash
cp /home/rst_spark/IsaacLab/source/isaaclab_mimic/robot/mimic_envs/teleop_device_factory_modified.py /home/rst_spark/IsaacLab/source/isaaclab/isaaclab/devices/teleop_device_factory.py
```
