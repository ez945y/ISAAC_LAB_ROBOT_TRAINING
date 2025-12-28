# Standalone Scripts - Isaac Lab Experiments

This directory contains standalone Isaac Lab experiment scripts for quick prototyping and learning.

## Scripts Overview

### 01_basic_auto_drive.py - Basic Scene + Auto Driving
Basic Isaac Lab scene setup with:
- Jetbot mobile robot
- Deformable cube
- Static collision stand
- Auto-driving logic (straight + turning)

```bash
python 01_basic_auto_drive.py
```

### 02_keyboard_control.py - Keyboard Control
Extended scene with keyboard control:
- WASD / Arrow keys to control Jetbot
- Command smoothing (prevents jitter)
- Differential drive model
- Press R to reset scene

```bash
python 02_keyboard_control.py
```

### 03_domino_fpv.py - Domino + First-Person View
Complex physics interactions:
- 4 dominoes with increasing height for physics demo
- Orange deformable cube hanging in air (top node fixed)
- First-person camera following Jetbot
- Keyboard control + scene reset

```bash
python 03_domino_fpv.py
```

### 04_trajectory_record.py - Trajectory Recording/Playback
Full-featured experiment script with all capabilities:

#### Features
- **Keyboard Control**: WASD or Arrow keys to control Jetbot
- **Domino Physics**: 4 dominoes with increasing height
- **Deformable Object**: Orange cube hanging in air
- **Trajectory Recording/Playback**: 
  - Press `I` to start recording
  - Press `O` to stop recording and start loop playback
  - Press `R` to reset scene and clear trajectory
- **First-Person View**: Camera automatically follows Jetbot

#### Controls
| Key | Action |
|-----|--------|
| W / ↑ | Move forward |
| S / ↓ | Move backward |
| A / ← | Turn left |
| D / → | Turn right |
| R | Reset scene |
| I | Start recording trajectory |
| O | Stop recording, start playback |

```bash
python 04_trajectory_record.py
```

### 05_robot_demo.py - SO-ARM-101 Robot Demo
Basic robotic arm demonstration with keyboard control.

```bash
python 05_robot_demo.py
```

### 06_teleoperate_demo.py - Leader Arm Teleoperation
Teleoperation of SO-ARM-101 robot using a physical leader arm via socket connection.

#### Prerequisites
1. Start the leader arm sender (in a separate terminal, using lerobot environment):
```bash
lerobot-teleoperate_port \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5AA90244081 \
    --teleop.id=my_awesome_leader_arm
```

2. Run this demo (from Isaac Lab environment):
```bash
python 06_teleoperate_demo.py
```

### 07_lerobot_teleoperate.py - LeRobot Bridge (Leader Arm Sender)
This script acts as the bridge for the physical leader arm. It reads joint data from the hardware using the `LeRobot` library and streams it over a socket network.

#### Installation (On Mac/Leader Arm Machine)
To use this script, you must first install the `LeRobot` library:
1. Clone the LeRobot repository: `git clone https://github.com/huggingface/lerobot.git`
2. Install dependencies: `cd lerobot && pip install -e .`
3. **Crucial Step**: Copy `07_lerobot_teleoperate.py` from this repository into your `lerobot` source folder or run it from this location ensuring the `lerobot` environment is active.

#### Usage
```bash
python 07_lerobot_teleoperate.py \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5AA90244081 \
    --teleop.id=my_awesome_leader_arm
```

#### Features
- Streams normalized joint positions (0~1) via JSON over Socket.
- Supports both Server and Client modes.
- Real-time terminal feedback of joint values.

#### Features
- Reads end-effector pose (x, y, z in meters) from socket port 5359
- Forward kinematics computed on the leader arm sender side
- Supports IK controller for simulation
- Real-time pose visualization with target marker

#### Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--controller` | `ik` | Controller type: `ik` or `osc` |
| `--socket-host` | `127.0.0.1` | Socket host for leader arm |
| `--socket-port` | `5359` | Socket port for leader arm |
| `--position-scale` | `1.0` | Scale factor for position |

## Script Evolution

```
01_basic_auto_drive.py     (Basic)
        ↓
02_keyboard_control.py     (+Keyboard control)
        ↓
03_domino_fpv.py           (+Dominoes +First-person view)
        ↓
04_trajectory_record.py    (+Trajectory recording/playback)
        ↓
05_robot_demo.py           (+SO-ARM-101 robot)
        ↓
06_teleoperate_demo.py     (+Leader arm teleoperation)
        ↓
07_lerobot_teleoperate.py  (+LeRobot bridge sender)
```

## Architecture

The leader arm teleoperation system consists of two parts:

1. **Leader Arm Sender** (`07_lerobot_teleoperate.py` on Mac)
   - Requires `lerobot` library installed.
   - Puts and rename teleoperate.py and teleop_processors.py in lerobot/scripts/ folder for 07 section to work. and register them on lerobot/pyproject.toml
   - Connects to physical SO-101 leader arm via serial port.
   - Completes calibration of the leader arm.
   - Reads joint positions and normalizes them to [0, 1] range.
   - Sends joint data via socket (port 5359).

2. **Simulation Demo** (`06_teleoperate_demo.py` on Ubuntu)
   - Receives joint data from socket
   - Directly sets joint positions on the simulated robot
   - Supports both **Joint Mode** (direct) and **EE Mode** (IK/OSC)

### Running the Demo

**Step 1: Start simulator on Ubuntu (server mode)**
```bash
cd ~/robot/standalone_scripts
python 06_teleoperate_demo.py
```

**Step 2: Start leader arm on Mac (client mode)**
```bash
# Ensure you are in the environment where lerobot is installed
python teleoperate.py \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5AA90244081 \
    --teleop.id=my_awesome_leader_arm \
    --socket_host=192.168.0.1
```

**Step 2b (Optional): Control both Leader and Follower arms simultaneously**

To also control a physical follower arm at the same time, add three more parameters:
```bash
python teleoperate.py \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5AA90244081 \
    --teleop.id=my_awesome_leader_arm \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5AA90244141 \
    --robot.id=my_awesome_follower_arm \
    --socket_host=192.168.0.1
```

### Data Flow Diagram

```
┌─────────────────────────────┐      Socket (5359)      ┌─────────────────────┐
│  07_lerobot_teleoperate.py  │  ──────────────────────▶│  06_teleoperate_    │
│  (Physical Leader Arm)      │  JSON: {mode: "joint",  │  demo.py            │
│  Mac - Reads Joints         │   shoulder_pan: 0.5,    │  (Isaac Lab Sim)    │
│                             │   shoulder_lift: 0.3,   │  Ubuntu             │
│                             │   ...}                  │                     │
└─────────────────────────────┘                         └─────────────────────┘
```

### Control Modes

| Mode | Description | When Used |
|------|-------------|-----------|
| **Joint** | Direct joint position control | Default (faster, more stable) |
| **EE** | End-effector pose with IK/OSC | Legacy mode (for pose-based control) |