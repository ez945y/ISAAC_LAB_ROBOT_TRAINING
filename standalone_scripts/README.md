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

## Script Evolution

```
01_basic_auto_drive.py     (Basic)
        ↓
02_keyboard_control.py     (+Keyboard control)
        ↓
03_domino_fpv.py           (+Dominoes +First-person view)
        ↓
04_trajectory_record.py    (+Trajectory recording/playback)
```