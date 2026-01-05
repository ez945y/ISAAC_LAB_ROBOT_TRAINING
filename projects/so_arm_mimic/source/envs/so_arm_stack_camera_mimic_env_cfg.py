# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""
SO-ARM-101 Isaac Lab Mimic Environment Configuration (Camera/Visual)
"""

import torch
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils # For spawning PinholeCameraCfg

from .so_arm_stack_joint_mimic_env_cfg import SOArmStackJointMimicEnvCfg, SOArmSceneCfg

@configclass
class SOArmStackCameraMimicEnvCfg(SOArmStackJointMimicEnvCfg):
    """
    SO-ARM-101 Stack Task with Camera Observations.
    Inherits from Joint Config but replaces observations.
    """
    
    # Use the base scene which now includes cameras
    scene: SOArmSceneCfg = SOArmSceneCfg(num_envs=1, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        
        # --- Update Observations ---
        # We want to remove ground-truth object states and use visual features.
        
        # 1. Modify Policy Group (The main input to agent)
        # We keep proprioception (joint pos) but replace object states with images.
        
        self.observations.policy.cube_positions = None # Remove GT positions
        self.observations.policy.cube_orientations = None # Remove GT orientations (Fix sim-to-real leak)
        
        # Add Cameras
        # Note: In Isaac Lab, image terms in "policy" group are flattened and concatenated 
        # if the group is set to concatenate (default).
        # This creates a large vector (128*128*3 * 2).
        
        self.observations.policy.wrist_rgb = ObsTerm(
            func=lambda env, asset_cfg: env.scene[asset_cfg.name].data.output["rgb"].view(env.num_envs, -1),
            params={"asset_cfg": SceneEntityCfg("wrist_camera")},
        )
        
        self.observations.policy.front_rgb = ObsTerm(
            func=lambda env, asset_cfg: env.scene[asset_cfg.name].data.output["rgb"].view(env.num_envs, -1),
            params={"asset_cfg": SceneEntityCfg("front_camera")},
        )
        
        # We also keep:
        # - joint_pos
        # - joint_vel
        # - gripper_pos_obs
        # - actions (last action)
        # - eef_pos/quat (maybe remove? usually visual agents don't need EEF gt if they have joints)
        # Keeping them doesn't hurt.

