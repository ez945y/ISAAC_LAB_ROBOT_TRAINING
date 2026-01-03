# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""SO-ARM-101 Mimic Environments

This module registers SO-ARM-101 environments with gymnasium.
The environments can be used with Isaac Lab Mimic scripts.

IMPORTANT: Due to Isaac Sim requirements, this module should be imported
AFTER SimulationApp is instantiated. Use the register_envs() function
to trigger registration at the right time.

Usage in your script:
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    
    # Now import and register
    import mimic_envs
    mimic_envs.register_envs()
"""

# Flag to track if environments have been registered
_ENVS_REGISTERED = False


def register_envs():
    """Register SO-ARM-101 environments with gymnasium.
    
    This function should be called AFTER SimulationApp is instantiated.
    It is safe to call multiple times - environments are only registered once.
    """
    global _ENVS_REGISTERED
    
    if _ENVS_REGISTERED:
        return
    
    import gymnasium as gym
    
    # Register with gymnasium
    gym.register(
        id="Isaac-PickPlace-SOArm-Joint-Mimic-v0",
        entry_point="mimic_envs.so_arm_stack_joint_mimic_env:SOArmStackJointMimicEnv",
        kwargs={
            "env_cfg_entry_point": "mimic_envs.so_arm_stack_joint_mimic_env_cfg:SOArmStackJointMimicEnvCfg",
        },
        disable_env_checker=True,
    )
    
    _ENVS_REGISTERED = True
    print("[mimic_envs] Registered: Isaac-PickPlace-SOArm-Joint-Mimic-v0")


# Export the register function
__all__ = ["register_envs"]
