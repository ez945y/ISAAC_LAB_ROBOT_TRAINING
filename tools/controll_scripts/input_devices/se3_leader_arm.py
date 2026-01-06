import numpy as np
import torch
import pinocchio as pin  # 引入 Pinocchio
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation
from typing import Any, Optional

from isaaclab.devices.device_base import DeviceBase, DeviceCfg

@dataclass
class Se3LeaderArmCfg(DeviceCfg):
    """Configuration for Leader Arm SE(3) device."""
    
    # Socket settings
    socket_host: str = "0.0.0.0"
    socket_port: int = 5359
    server_mode: bool = True
    
    # Control settings
    pos_sensitivity: float = 1.0
    rot_sensitivity: float = 1.0
    gripper_term: bool = True
    
    # Device settings
    sim_device: str = "cuda:0"
    
    # [NEW] Pinocchio Settings
    # 請修改為你的 URDF 絕對路徑或相對路徑
    urdf_path: str = "tools/controll_scripts/so_arm_101/description/so101_new_calib.urdf" 
    # 請確認這是你 URDF 中末端 effector 的 link 名稱
    ee_frame_name: str = "gripper" 


class Se3LeaderArm(DeviceBase):
    """A leader arm controller using Pinocchio for Forward Kinematics."""
    
    def __init__(self, cfg: Se3LeaderArmCfg):
        super().__init__(retargeters=None)
        
        self._cfg = cfg
        self._sim_device = cfg.sim_device
        
        # Import leader arm (lazy import)
        self._leader_arm: Optional["LeaderArmInputDevice"] = None
        self._init_leader_arm()
        
        # Initialize Pinocchio
        self._init_pinocchio()
        
        # State tracking
        self._was_connected = False
        self._pending_reset = False
        self._additional_callbacks: dict[str, Callable] = {}

        # Joint Limits (SO-ARM-100 values from your previous code)
        self.arm_lower = torch.tensor([-1.92, -1.92, -1.92, -1.66, -2.74], device=self._sim_device)
        self.arm_upper = torch.tensor([ 1.92,  1.75,  1.69,  1.66,  2.84], device=self._sim_device)
        self.gripper_lower = -0.17
        self.gripper_upper = 1.75

    def _init_leader_arm(self):
        """Initialize the leader arm input device."""
        try:
            from controll_scripts.input_devices.leader_arm import LeaderArmInputDevice
            # Initial pose is less relevant now as we compute FK from joints
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
        """Initialize Pinocchio model and data."""
        urdf_path = self._cfg.urdf_path
        if not os.path.exists(urdf_path):
            # Try to resolve relative to the workspace root if it's a relative path
            # We assume the workspace root is where the current script's 'tools' directory starts
            potential_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            urdf_path = os.path.join(potential_root, self._cfg.urdf_path)
            
        if not os.path.exists(urdf_path):
            print(f"[Se3LeaderArm] Error: URDF not found at {self._cfg.urdf_path} or {urdf_path}")
            # Fallback to a dummy model or raise error depending on preference
            self.pin_model = None
            return

        try:
            # Load model
            self.pin_model = pin.buildModelFromUrdf(urdf_path)
            self.pin_data = self.pin_model.createData()
            
            # Get frame ID for the end effector
            if self.pin_model.existFrame(self._cfg.ee_frame_name):
                self.ee_frame_id = self.pin_model.getFrameId(self._cfg.ee_frame_name)
                print(f"[Se3LeaderArm] Pinocchio loaded. Tracking frame: {self._cfg.ee_frame_name} (ID: {self.ee_frame_id})")
            else:
                print(f"[Se3LeaderArm] Error: Frame '{self._cfg.ee_frame_name}' not found in URDF!")
                self.pin_model = None
        except Exception as e:
            print(f"[Se3LeaderArm] Pinocchio Initialization Failed: {e}")
            self.pin_model = None

    def __del__(self):
        if self._leader_arm is not None:
            self._leader_arm.close()

    def reset(self):
        # Implementation depends on if you need to reset internal filters
        pass

    def add_callback(self, key: Any, func: Callable):
        self._additional_callbacks[key] = func
    
    @property
    def is_connected(self) -> bool:
        return self._leader_arm is not None and self._leader_arm.is_connected

    def advance(self) -> torch.Tensor:
        """
        Computes FK using Pinocchio and returns absolute pose.
        Output Shape: (8,)
        Format: [x, y, z, qw, qx, qy, qz, gripper_radian]
        """
        # 1. Connection Handling
        currently_connected = self._leader_arm is not None and self.is_connected
        
        if self._was_connected and not currently_connected:
            print("\n>> [DISCONNECT] Leader arm disconnected!")
            self._pending_reset = True
            self._was_connected = False
        
        if not currently_connected or self.pin_model is None:
            # Return a safe default pose if not connected or model failed
            # Default: x, y, z, qw, qx, qy, qz, gripper
            return torch.tensor([0.2, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0, 0.0], device=self._sim_device)

        if getattr(self, '_pending_reset', False) and currently_connected:
            print(">> [RECONNECT] Leader arm reconnected!")
            if "R" in self._additional_callbacks:
                self._additional_callbacks["R"]()
            self._pending_reset = False
        
        self._was_connected = True

        # 2. Get Raw Joint Data (Normalized 0-1)
        # Assuming joint_positions returns tensor of shape (5,)
        raw_joints = self._leader_arm.joint_positions # Tensor on device
        
        # Get gripper data
        _, raw_gripper, _ = self._leader_arm.update()
        
        # 3. Scale Joints to Radians (Input Correction)
        # Map input index to output index [0,1,2,3,4]
        ordered_joints = raw_joints  # Add mapping here if needed like raw_joints[JOINT_MAP]
        
        # Calculation: lower + norm * (upper - lower)
        # Result is in Radians
        q_arm_tensor = self.arm_lower + ordered_joints * (self.arm_upper - self.arm_lower)
        
        # Calculation: Gripper Radian
        q_gripper_scalar = self.gripper_lower + raw_gripper * (self.gripper_upper - self.gripper_lower)

        # 4. Pinocchio Forward Kinematics
        # Pinocchio requires numpy array (CPU) usually
        q_arm_np = q_arm_tensor.cpu().numpy().astype(np.float64)
        
        # Verify DoF match
        # Pinocchio model nq might include gripper joints depending on your URDF.
        # Assuming the URDF is the arm chain (5 or 6 DoF). 
        # If URDF expects more joints, you need to pad q_arm_np.
        if q_arm_np.shape[0] != self.pin_model.nq:
            # Simple padding if URDF has extra joints (like fixed gripper joints)
            # This is a fallback; ideally URDF nq should match arm joints
            padded_q = np.zeros(self.pin_model.nq)
            min_len = min(q_arm_np.shape[0], self.pin_model.nq)
            padded_q[:min_len] = q_arm_np[:min_len]
            q_in = padded_q
        else:
            q_in = q_arm_np

        # Run FK
        pin.forwardKinematics(self.pin_model, self.pin_data, q_in)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        
        # Get End-Effector SE3
        # oMf stands for "Object (Frame) wrt Model (World) Frame"
        M_ee = self.pin_data.oMf[self.ee_frame_id]
        
        # Extract Translation (x, y, z)
        trans = M_ee.translation # np.array shape (3,)
        
        # Extract Rotation as Quaternion
        # Pinocchio quaternion is usually created from rotation matrix
        rot_mat = M_ee.rotation
        quat_pin = pin.Quaternion(rot_mat) 
        # Pinocchio Quaternion object has .x, .y, .z, .w attributes
        # You requested: qw, qx, qy, qz
        quat_vec = np.array([quat_pin.w, quat_pin.x, quat_pin.y, quat_pin.z])
        
        # 5. Assemble Output Tensor
        # [x, y, z, qw, qx, qy, qz, gripper_radian]
        
        # Move back to torch device
        pos_t = torch.from_numpy(trans).to(dtype=torch.float32, device=self._sim_device)
        rot_t = torch.from_numpy(quat_vec).to(dtype=torch.float32, device=self._sim_device)
        grip_t = torch.tensor([q_gripper_scalar], dtype=torch.float32, device=self._sim_device)
        
        command = torch.cat([pos_t, rot_t, grip_t])
        
        return command