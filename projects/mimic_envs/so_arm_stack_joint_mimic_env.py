# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""
SO-ARM-101 Isaac Lab Mimic Environment (Joint Control)
"""

import torch
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv


class SOArmStackJointMimicEnv(ManagerBasedRLMimicEnv):
    """
    Isaac Lab Mimic environment wrapper class for SO-ARM-101 with Joint Control.
    """

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get current robot end effector pose."""
        if env_ids is None:
            env_ids = slice(None)

        # Retrieve end effector pose from the observation buffer
        eef_pos = self.obs_buf["policy"]["eef_pos"][env_ids]
        eef_quat = self.obs_buf["policy"]["eef_quat"][env_ids]
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def target_eef_pose_to_action(
        self, target_eef_pose_dict: dict, gripper_action_dict: dict, noise: float | None = None, env_id: int = 0
    ) -> torch.Tensor:
        """Convert target pose to action.
        
        NOTE: For Joint Control, converting EE pose to Joint Angles requires IK.
        For simple demo playback (Action Playback), this method might not be used.
        But for MimicGen generation, it is used.
        
        For now, we raise NotImplementedError or return zeros since we are only Recording Demos.
        """
        # raise NotImplementedError("Inverse Kinematics not implemented for SOArm Joint Env yet.")
        return torch.zeros((1, 6), device=self.device)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert action to target pose.
        
        For Joint Control, converting Joint Angles to EE Pose requires FK.
        But self.scene already computes FK for us (via frame transformer), so we could potentially use that.
        However, this method is usually for visualizing the goal of an action.
        """
        # Placeholder
        return {list(self.cfg.subtask_configs.keys())[0]: torch.zeros((action.shape[0], 7), device=self.device)}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract gripper actions."""
        # last dimension is gripper action
        return {list(self.cfg.subtask_configs.keys())[0]: actions[:, -1:]}

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Get subtask termination signals."""
        if env_ids is None:
            env_ids = slice(None)

        signals = dict()
        subtask_terms = self.obs_buf["subtask_terms"]
        signals["grasp_1"] = subtask_terms["grasp_1"][env_ids]
        return signals
