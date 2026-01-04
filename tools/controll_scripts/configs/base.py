# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""機器人配置抽象基類"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


@dataclass
class BaseRobotConfig(ABC):
    """機器人配置抽象基類
    
    定義機器人的基本屬性和配置方法，支持不同的控制器類型。
    """
    
    # 基本屬性
    name: str = ""
    usd_path: str = ""
    arm_joint_names: list = field(default_factory=list)
    gripper_joint_name: str = ""
    ee_body_name: str = ""
    
    # IK 控制器配置
    ik_stiffness: float = 1000.0  # 提高剃化以加快響應
    ik_damping: float = 100.0
    ik_lambda_val: float = 0.005 
    ik_orientation_weight: float = 0.3  # 姿態權重（0-1），低值讓位置優先
    # OSC 控制器配置
    osc_motion_stiffness: tuple = (150.0, 150.0, 150.0, 50.0, 50.0, 50.0)
    osc_motion_damping_ratio: tuple = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    osc_effort_limit: float = 50.0
    
    # 夾爪配置
    gripper_stiffness: float = 100.0
    gripper_damping: float = 10.0
    gripper_open_pos: float = 0.5
    gripper_close_pos: float = 0.0
    
    def get_articulation_cfg(self, for_osc: bool = False) -> ArticulationCfg:
        """獲取 ArticulationCfg
        
        Args:
            for_osc: 是否用於 OSC 控制器（需要 stiffness=0, damping=0）
            
        Returns:
            ArticulationCfg: 機器人配置
        """
        if for_osc:
            arm_stiffness = 0.0
            arm_damping = 0.0
        else:
            arm_stiffness = self.ik_stiffness
            arm_damping = self.ik_damping
        
        return ArticulationCfg(
            spawn=sim_utils.UsdFileCfg(
                usd_path=self.usd_path,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    fix_root_link=True,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
            ),
            actuators={
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=self.arm_joint_names,
                    effort_limit=self.osc_effort_limit if for_osc else 1000.0,
                    stiffness=arm_stiffness,
                    damping=arm_damping,
                ),
                "gripper": ImplicitActuatorCfg(
                    joint_names_expr=[self.gripper_joint_name],
                    effort_limit=2.0,  # Limit gripper force to prevent "explosion"
                    stiffness=self.gripper_stiffness,
                    damping=self.gripper_damping,
                ),
            },
        )
    
    @property
    def num_arm_joints(self) -> int:
        """手臂關節數量"""
        return len(self.arm_joint_names)
