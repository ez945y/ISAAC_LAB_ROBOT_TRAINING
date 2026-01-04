# Robot Tools

This directory contains utility scripts and assets for working with robot data, models, and teleoperation.

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
├── contoller_client/   
│   ├── teleoperate_port.py     # Leader arm sender (Mac side)
│   ├── teleop_processors.py    # Teleoperation support module
└── README.md
```

## controll_scripts

The `controll_scripts` directory contains shared robot control modules and assets. 
This directory is symlinked to `isaaclab_mimic/isaaclab_mimic/controll_scripts`, allowing imports like:

```python
from isaaclab_mimic.controll_scripts.input_devices.se3_leader_arm import Se3LeaderArm
```

This ensures these assets are accessible to all projects.

## Teleoperation Client (For Mac/LeRobot Side)

Scripts in `contoller_client/` are for the workstation connected to the physical leader arm.

### teleoperate_port.py

Reads joint positions from a physical leader arm using LeRobot and sends them over network.
