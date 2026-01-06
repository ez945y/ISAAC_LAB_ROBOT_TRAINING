# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""
SO-ARM-101 Isaac Lab Mimic Environment (Joint Control)

Uses Differential IK to convert EEF pose targets to joint position actions.
"""

import torch
from collections.abc import Sequence

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLMimicEnv


class SOArmStackJointMimicEnv(ManagerBasedRLMimicEnv):
    """
    Isaac Lab Mimic environment wrapper class for SO-ARM-101 with Joint Control.
    
    Uses Differential IK to convert end-effector pose commands to joint position actions.
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Controller configuration is now handled via Action Config (DifferentialInverseKinematicsActionCfg)
        # No manual controller setup needed here.
        pass

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get current robot end effector pose as 4x4 matrix."""
        if env_ids is None:
            env_ids = slice(None)

        # Retrieve end effector pose from the observation buffer
        eef_pos = self.obs_buf["policy"]["eef_pos"][env_ids]
        eef_quat = self.obs_buf["policy"]["eef_quat"][env_ids]
        return math_utils.make_pose(eef_pos, math_utils.matrix_from_quat(eef_quat))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        noise: float | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Convert target EEF pose to action (Pose + Gripper)."""
        # Extract target pose
        (target_pose_matrix,) = target_eef_pose_dict.values()
        target_pos, target_rot = math_utils.unmake_pose(target_pose_matrix)
        target_quat = math_utils.quat_from_matrix(target_rot)
        
        # Get gripper action
        (gripper_action,) = gripper_action_dict.values()
        
        # Combine pose and gripper into action
        # Action format: [pos(3), rot(4), gripper(1)]
        pose_action = torch.cat([target_pos, target_quat], dim=-1)
        
        # Add noise if specified
        if action_noise_dict is not None:
             noise_scale = list(action_noise_dict.values())[0]
             noise_vec = torch.randn_like(pose_action) * noise_scale
             pose_action += noise_vec
        
        # Combine
        action = torch.cat([pose_action, gripper_action], dim=-1)
        
        return action.unsqueeze(0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert joint position action to target EEF pose using Forward Kinematics.
        
        This uses the current EE pose from the scene (frame transformer computes FK).
        Note: This is approximate since we use current pose, not the pose that would
        result from applying the action.
        """
        eef_name = list(self.cfg.subtask_configs.keys())[0]
        
        # Get current EE pose from observations (already computed by frame transformer)
        eef_pos = self.obs_buf["policy"]["eef_pos"]
        eef_quat = self.obs_buf["policy"]["eef_quat"]
        
        # Create 4x4 pose matrices
        eef_rot = math_utils.matrix_from_quat(eef_quat)
        target_poses = math_utils.make_pose(eef_pos, eef_rot)
        
        return {eef_name: target_poses}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract gripper actions from full action tensor."""
        # Last dimension is gripper action
        return {list(self.cfg.subtask_configs.keys())[0]: actions[:, -1:]}

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Get subtask termination signals for MimicGen annotation."""
        if env_ids is None:
            env_ids = slice(None)

        signals = dict()
        subtask_terms = self.obs_buf["subtask_terms"]
        
        signals["grasp_1"] = subtask_terms["grasp_1"][env_ids]
        signals["stack_1"] = subtask_terms["stack_1"][env_ids]
        signals["grasp_2"] = subtask_terms["grasp_2"][env_ids]
        
        return signals
