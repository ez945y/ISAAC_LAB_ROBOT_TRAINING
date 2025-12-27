# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""SO-ARM-101 機器人配置"""

import os
from dataclasses import dataclass

from .base import BaseRobotConfig


@dataclass
class SOArm101Config(BaseRobotConfig):
    """SO-ARM-101 機器人配置"""
    
    name: str = "SO-ARM-101"
    usd_path: str = ""  # 將在 __post_init__ 中設定
    arm_joint_names: list = None
    gripper_joint_name: str = "gripper"
    ee_body_name: str = "gripper_link"
    
    # IK 控制器配置
    ik_stiffness: float = 1000.0  # 提高剃化以加快響應
    ik_damping: float = 100.0
    ik_lambda_val: float = 0.001  # 極小 lambda = 極快響應
    ik_orientation_weight: float = 0.7  # 提高姿態權重減少耦合
    
    # OSC 控制器配置
    osc_motion_stiffness: tuple = (150.0, 150.0, 150.0, 50.0, 50.0, 50.0)
    osc_motion_damping_ratio: tuple = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    osc_effort_limit: float = 50.0
    
    # 夾爪配置
    gripper_open_pos: float = 0.5
    gripper_close_pos: float = 0.0
    
    def __post_init__(self):
        # 設定預設 USD 路徑（相對於 scripts 目錄）
        if not self.usd_path:
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.usd_path = os.path.join(script_dir, "so_arm_101", "SO-ARM101.usd")
        
        # 設定預設關節名稱
        if self.arm_joint_names is None:
            self.arm_joint_names = [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ]
