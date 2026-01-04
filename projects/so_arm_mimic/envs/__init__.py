# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from .so_arm_stack_joint_mimic_env import SOArmStackJointMimicEnv
from .so_arm_stack_joint_mimic_env_cfg import SOArmStackJointMimicEnvCfg
from .so_arm_stack_camera_mimic_env_cfg import SOArmStackCameraMimicEnvCfg

##
# SO-ARM-101 Pick and Place - Joint Control
##
gym.register(
    id="Isaac-PickPlace-SOArm-Joint-Mimic-v0",
    entry_point="so_arm_mimic.envs.so_arm_stack_joint_mimic_env:SOArmStackJointMimicEnv",
    kwargs={
        "env_cfg_entry_point": "so_arm_mimic.envs.so_arm_stack_joint_mimic_env_cfg:SOArmStackJointMimicEnvCfg",
    },
    disable_env_checker=True,
)

##
# SO-ARM-101 Pick and Place - Camera/Visual Control
##
gym.register(
    id="Isaac-PickPlace-SOArm-Camera-Mimic-v0",
    entry_point="so_arm_mimic.envs.so_arm_stack_joint_mimic_env:SOArmStackJointMimicEnv",
    kwargs={
        "env_cfg_entry_point": "so_arm_mimic.envs.so_arm_stack_camera_mimic_env_cfg:SOArmStackCameraMimicEnvCfg",
    },
    disable_env_checker=True,
)
