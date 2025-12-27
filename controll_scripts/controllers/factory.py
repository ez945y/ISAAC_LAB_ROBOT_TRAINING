# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""控制器工廠"""

from enum import Enum, auto
from typing import Type

from isaaclab.assets import Articulation

from .base import BaseController
from .ik_controller import IKController
from .osc_controller import OSCController
from ..configs.base import BaseRobotConfig


class ControllerType(Enum):
    """控制器類型枚舉"""
    IK = auto()
    OSC = auto()


class ControllerFactory:
    """控制器工廠
    
    使用工廠模式創建不同類型的控制器。
    """
    
    _controllers: dict[ControllerType, Type[BaseController]] = {
        ControllerType.IK: IKController,
        ControllerType.OSC: OSCController,
    }
    
    @classmethod
    def create(
        cls,
        controller_type: ControllerType,
        robot: Articulation,
        robot_config: BaseRobotConfig,
        device: str,
        num_envs: int = 1,
    ) -> BaseController:
        """創建控制器
        
        Args:
            controller_type: 控制器類型
            robot: Isaac Lab Articulation 對象
            robot_config: 機器人配置
            device: 計算設備
            num_envs: 環境數量
            
        Returns:
            BaseController: 控制器實例
            
        Raises:
            ValueError: 未知的控制器類型
        """
        if controller_type not in cls._controllers:
            raise ValueError(f"Unknown controller type: {controller_type}")
        
        controller_class = cls._controllers[controller_type]
        print(f"[ControllerFactory] 創建 {controller_class.__name__}")
        
        return controller_class(
            robot=robot,
            robot_config=robot_config,
            device=device,
            num_envs=num_envs,
        )
    
    @classmethod
    def register(cls, controller_type: ControllerType, controller_class: Type[BaseController]) -> None:
        """註冊新的控制器類型
        
        Args:
            controller_type: 控制器類型
            controller_class: 控制器類
        """
        cls._controllers[controller_type] = controller_class
    
    @classmethod
    def available_types(cls) -> list[ControllerType]:
        """獲取可用的控制器類型"""
        return list(cls._controllers.keys())
