# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""輸入設備抽象基類"""

from abc import ABC, abstractmethod
from typing import Tuple

import torch


class BaseInputDevice(ABC):
    """輸入設備抽象基類
    
    定義統一的輸入設備介面，輸出絕對目標姿態。
    可以是鍵盤、leader arm、遙操作設備等。
    """
    
    @abstractmethod
    def update(self) -> Tuple[torch.Tensor, float, bool]:
        """更新輸入狀態
        
        Returns:
            Tuple[torch.Tensor, float, bool]:
                - target_pose: 絕對目標姿態 [pos(3) + quat(4)]
                - gripper_pos: 夾爪位置 [0, 1] (0=關閉, 1=全開)
                - reset_requested: 是否請求重置
        """
        pass
    
    @property
    @abstractmethod
    def target_pose(self) -> torch.Tensor:
        """當前目標姿態"""
        pass
    
    @property
    @abstractmethod
    def gripper_pos(self) -> float:
        """當前夾爪位置"""
        pass
    
    @abstractmethod
    def reset_target(self, pose: torch.Tensor) -> None:
        """重置目標姿態
        
        Args:
            pose: 新的目標姿態
        """
        pass
    
    @abstractmethod
    def sync_to_actual(self, actual_pose: torch.Tensor) -> None:
        """同步目標到實際姿態（當目標不可達時）
        
        Args:
            actual_pose: 實際 EE 姿態
        """
        pass
