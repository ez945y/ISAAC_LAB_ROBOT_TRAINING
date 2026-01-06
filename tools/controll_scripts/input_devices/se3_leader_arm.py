import numpy as np
import torch
import pinocchio as pin
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation
from typing import Any, Optional

from isaaclab.devices.device_base import DeviceBase, DeviceCfg

@dataclass
class Se3LeaderArmCfg(DeviceCfg):
    """Configuration for Leader Arm SE(3) device."""
    
    socket_host: str = "0.0.0.0"
    socket_port: int = 5359
    server_mode: bool = True
    
    pos_sensitivity: float = 1.0
    rot_sensitivity: float = 1.0
    gripper_term: bool = True
    
    sim_device: str = "cuda:0"
    urdf_path: str = "tools/controll_scripts/so_arm_101/description/so101_new_calib.urdf" 
    ee_frame_name: str = "gripper" 
    
    debug_mode: bool = True

    # [設定 1] 關節方向修正
    # 根據你的描述，如果方向反了 (往左變往右)，把第一個 1.0 改成 -1.0
    joint_signs: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0])

    # [設定 2] 關節歸零偏移 (Rad)
    # URDF 的第一軸 origin rpy 有 1.5708 (90度)
    # 為了抵消這個旋轉，我們通常需要設 -1.57 或 +1.57
    # 根據你之前的測試 (左邊變前方)，建議從 -1.5708 開始試
    joint_offsets: list[float] = field(default_factory=lambda: [-1.5708, 0.0, 0.0, 0.0, 0.0])


class Se3LeaderArm(DeviceBase):
    
    def __init__(self, cfg: Se3LeaderArmCfg):
        super().__init__(retargeters=None)
        
        self._cfg = cfg
        self._sim_device = cfg.sim_device
        
        self._joint_signs = torch.tensor(cfg.joint_signs, device=self._sim_device)
        self._joint_offsets = torch.tensor(cfg.joint_offsets, device=self._sim_device)
        
        self._leader_arm: Optional["LeaderArmInputDevice"] = None
        self._init_leader_arm()
        self._init_pinocchio()
        
        self._was_connected = False
        self._pending_reset = False
        self._additional_callbacks: dict[str, Callable] = {}

        # [UPDATED] Joint Limits based on URDF
        # J2 (Elbow) 修正為 URDF 的 [-1.75, 1.57]
        self.arm_lower = torch.tensor([-1.92, -1.75, -1.75, -1.66, -2.79], device=self._sim_device)
        self.arm_upper = torch.tensor([ 1.92,  1.75,  1.57,  1.66,  2.79], device=self._sim_device)
        
        # Gripper limit (J5 in URDF but handled separately here)
        self.gripper_lower = -0.17
        self.gripper_upper = 1.75

    def _init_leader_arm(self):
        try:
            from controll_scripts.input_devices.leader_arm import LeaderArmInputDevice
            initial_pose = torch.tensor([0.25, 0.0, 0.15, 1.0, 0.0, 0.0, 0.0], device=self._sim_device)
            self._leader_arm = LeaderArmInputDevice(
                initial_pose=initial_pose,
                device=self._sim_device,
                socket_host=self._cfg.socket_host,
                socket_port=self._cfg.socket_port,
                server_mode=self._cfg.server_mode,
            )
            print(f"[Se3LeaderArm] Initialized on {self._cfg.socket_host}:{self._cfg.socket_port}")
        except ImportError as e:
            print(f"[Se3LeaderArm] Warning: Could not import LeaderArmInputDevice: {e}")
            self._leader_arm = None

    def _init_pinocchio(self):
        urdf_path = self._cfg.urdf_path
        if not os.path.exists(urdf_path):
            potential_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            urdf_path = os.path.join(potential_root, self._cfg.urdf_path)
        
        if not os.path.exists(urdf_path):
            print(f"[Se3LeaderArm] Error: URDF not found")
            self.pin_model = None
            return

        try:
            self.pin_model = pin.buildModelFromUrdf(urdf_path)
            self.pin_data = self.pin_model.createData()
            
            if self.pin_model.existFrame(self._cfg.ee_frame_name):
                self.ee_frame_id = self.pin_model.getFrameId(self._cfg.ee_frame_name)
            else:
                print(f"[Se3LeaderArm] Error: Frame '{self._cfg.ee_frame_name}' not found!")
                self.pin_model = None
        except Exception as e:
            print(f"[Se3LeaderArm] Pinocchio Init Failed: {e}")
            self.pin_model = None

    def __del__(self):
        if self._leader_arm is not None:
            self._leader_arm.close()
    
    def reset(self):
        pass

    def add_callback(self, key: Any, func: Callable):
        self._additional_callbacks[key] = func
    
    @property
    def is_connected(self) -> bool:
        return self._leader_arm is not None and self._leader_arm.is_connected

    def advance(self) -> torch.Tensor:
        currently_connected = self._leader_arm is not None and self.is_connected
        
        if self._was_connected and not currently_connected:
            print("\n>> [DISCONNECT] Leader arm disconnected!")
            self._pending_reset = True
            self._was_connected = False
        
        if not currently_connected or self.pin_model is None:
            return torch.tensor([0.2, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0, 0.0], device=self._sim_device)

        if getattr(self, '_pending_reset', False) and currently_connected:
            print(">> [RECONNECT] Leader arm reconnected!")
            if "R" in self._additional_callbacks:
                self._additional_callbacks["R"]()
            self._pending_reset = False
        
        self._was_connected = True

        # --- FK Logic ---
        raw_joints = self._leader_arm.joint_positions 
        _, raw_gripper, _ = self._leader_arm.update()
        
        # 1. Map 0~1 to Radians
        q_mapped = self.arm_lower + raw_joints * (self.arm_upper - self.arm_lower)
        
        # 2. Apply Correction (Sign & Offset)
        q_corrected = (q_mapped * self._joint_signs) + self._joint_offsets
        
        # 3. Pinocchio FK
        q_np = q_corrected.cpu().numpy().astype(np.float64)
        
        if q_np.shape[0] != self.pin_model.nq:
            padded_q = np.zeros(self.pin_model.nq)
            min_len = min(q_np.shape[0], self.pin_model.nq)
            padded_q[:min_len] = q_np[:min_len]
            # 第 6 軸 (Jaw) 自動補 0，這符合 URDF 結構
            q_in = padded_q
        else:
            q_in = q_np

        pin.forwardKinematics(self.pin_model, self.pin_data, q_in)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        M_ee = self.pin_data.oMf[self.ee_frame_id]
        
        trans = M_ee.translation
        rot_mat = M_ee.rotation
        quat_obj = Rotation.from_matrix(rot_mat)
        qx, qy, qz, qw = quat_obj.as_quat()

        # Debug Output
        # print(f"J0 (Input rad): {q_corrected[0]:.2f}")
        # print(f"XYZ: [{trans[0]:.2f}, {trans[1]:.2f}, {trans[2]:.2f}]")
        # # 簡單指引
        # if abs(trans[1]) < 0.05 and trans[0] > 0.1:
        #     print("-> 狀態：正前方 (X+)")
        # elif trans[1] > 0.1 and abs(trans[0]) < 0.05:
        #     print("-> 狀態：左邊 (Y+)")
        # elif trans[1] < -0.1 and abs(trans[0]) < 0.05:
        #     print("-> 狀態：右邊 (Y-)")

        q_gripper_scalar = self.gripper_lower + raw_gripper * (self.gripper_upper - self.gripper_lower)
        
        pos_t = torch.from_numpy(trans).to(dtype=torch.float32, device=self._sim_device)
        rot_t = torch.tensor([qw, qx, qy, qz], dtype=torch.float32, device=self._sim_device)
        grip_t = torch.tensor([q_gripper_scalar], dtype=torch.float32, device=self._sim_device)
        
        return torch.cat([pos_t, rot_t, grip_t])