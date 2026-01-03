# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""
SO-ARM-101 Isaac Lab Mimic Environment Configuration (Joint Control)
"""

import os
import torch
from dataclasses import MISSING, field

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mimic_env_cfg import DataGenConfig, MimicEnvCfg, SubTaskConfig
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass

# Import Se3LeaderArmCfg for teleop device registration
from isaaclab_mimic.controll_scripts.input_devices.se3_leader_arm import Se3LeaderArmCfg


##
# Scene definition
##
@configclass
class SOArmSceneCfg(InteractiveSceneCfg):
    """Configuration for the SO-ARM-101 scene."""

    # World
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    
    # Light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    
    # Robot (configured in __post_init__)
    robot: ArticulationCfg = MISSING
    
    # End-effector frame (configured in __post_init__)
    ee_frame: FrameTransformerCfg = MISSING

    # Object (Cube)
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.8)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.0, 0.04)),
    )


##
# MDP settings
##

def ee_frame_pos(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """End-effector position in world frame."""
    return env.scene[asset_cfg.name].data.target_pos_w[..., 0, :]

def ee_frame_quat(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """End-effector orientation (quaternion) in world frame."""
    return env.scene[asset_cfg.name].data.target_quat_w[..., 0, :]

# Action history
def last_action(env) -> torch.Tensor:
    """Last action taken by the agent."""
    return env.action_manager.action

# Relative joint positions/velocities
def joint_pos_rel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint positions relative to default."""
    return env.scene[asset_cfg.name].data.joint_pos - env.scene[asset_cfg.name].data.default_joint_pos

def joint_vel_rel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint velocities."""
    return env.scene[asset_cfg.name].data.joint_vel

# Object state
def cube_pos(env) -> torch.Tensor:
    """Cube position."""
    return env.scene["cube"].data.root_pos_w

def cube_quat(env) -> torch.Tensor:
    """Cube orientation."""
    return env.scene["cube"].data.root_quat_w

def gripper_pos(env) -> torch.Tensor:
    """Gripper joint position."""
    # Assuming gripper is the last joint for simplicity or specific name
    # We can be more specific if needed
    return env.scene["robot"].data.joint_pos[:, -1:]

# Subtask signals (simplified)
def object_grasped(env, robot_cfg: SceneEntityCfg, ee_frame_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Check if object is grasped (close to EE and gripper closed)."""
    # This is a placeholder logic
    ee_pos = env.scene[ee_frame_cfg.name].data.target_pos_w[..., 0, :]
    obj_pos = env.scene[object_cfg.name].data.root_pos_w
    
    dist = torch.norm(ee_pos - obj_pos, dim=-1)
    is_close = dist < 0.05
    
    # Gripper state (closed)
    gripper_state = env.scene[robot_cfg.name].data.joint_pos[:, -1]
    is_gripping = gripper_state < 0.02 # Assuming 0 is closed
    
    return (is_close & is_gripping).float().unsqueeze(-1)


##
# Observation configuration
##
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        # Use lambda to defer import
        eef_pos = ObsTerm(func=ee_frame_pos, params={"asset_cfg": SceneEntityCfg("ee_frame")})
        eef_quat = ObsTerm(func=ee_frame_quat, params={"asset_cfg": SceneEntityCfg("ee_frame")})
        joint_pos = ObsTerm(func=joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        actions = ObsTerm(func=last_action)
        gripper_pos = ObsTerm(func=gripper_pos)
        cube_pos = ObsTerm(func=cube_pos)
        cube_quat = ObsTerm(func=cube_quat)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask signals."""
        
        grasp_1 = ObsTerm(
            func=object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


##
# Action configuration
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # Joint Position Control for Arm + Gripper
    # We combine them into a single action term if possible, or split them.
    # Mimic usually likes a single action tensor.
    # But JointPositionActionCfg applies to specific joints.
    
    # 5 Arm joints + 1 Gripper joint
    joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
        scale=1.0, 
        use_default_offset=False, # We want absolute control
    )


##
# Termination configuration
##
def time_out_termination(env) -> torch.Tensor:
    """Time out termination."""
    return env.episode_length_buf >= env.max_episode_length


def object_dropped(env, asset_cfg: SceneEntityCfg, minimum_height: float = -0.05) -> torch.Tensor:
    """Check if object dropped below minimum height."""
    obj = env.scene[asset_cfg.name]
    return obj.data.root_pos_w[:, 2] < minimum_height  # Returns bool tensor


def cube_lifted_success(env) -> torch.Tensor:
    """Success: cube is lifted above a threshold height."""
    cube = env.scene["cube"]
    # Success if cube is lifted above 0.1m
    return cube.data.root_pos_w[:, 2] > 0.1  # Returns bool tensor


@configclass
class TerminationsCfg:
    """Termination conditions."""
    
    time_out = DoneTerm(func=time_out_termination, time_out=True)
    cube_dropped = DoneTerm(
        func=object_dropped,
        params={"asset_cfg": SceneEntityCfg("cube"), "minimum_height": -0.05},
    )
    # Success condition for demo recording
    success = DoneTerm(func=cube_lifted_success)


@configclass
class RewardsCfg:
    """Empty rewards config for demo recording (no RL training)."""
    pass


@configclass
class EventCfg:
    """Empty events config."""
    pass


@configclass
class CommandsCfg:
    """Empty commands config."""
    pass


@configclass
class CurriculumCfg:
    """Empty curriculum config."""
    pass


##
# Main Environment Configuration
##
@configclass 
class SOArmStackJointMimicEnvCfg(ManagerBasedRLEnvCfg, MimicEnvCfg):
    """
    Isaac Lab Mimic environment configuration for SO-ARM-101 with Joint Control.
    """

    # Scene settings
    scene: SOArmSceneCfg = SOArmSceneCfg(num_envs=1, env_spacing=2.0)
    
    # Observations
    observations: ObservationsCfg = ObservationsCfg()
    
    # Actions (set in __post_init__)
    actions: ActionsCfg = ActionsCfg()
    
    # Terminations
    terminations: TerminationsCfg = TerminationsCfg()

    # Use empty configs instead of None
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization - configure robot and actions."""
        # === Basic settings ===
        self.decimation = 5
        self.episode_length_s = 60.0
        self.sim.dt = 0.01  # 100Hz physics
        self.sim.render_interval = 2
        
        # === Camera settings ===
        self.viewer.eye = [-0.4, -0.4, 0.4]
        self.viewer.lookat = [0.0, 0.0, 0.15]

        # === Teleop devices configuration ===
        # Register leader_arm as a teleop device
        self.teleop_devices = DevicesCfg(
            devices={
                "leader_arm": Se3LeaderArmCfg(
                    socket_host="0.0.0.0",
                    socket_port=5359,
                    server_mode=True,
                    pos_sensitivity=1.0,
                    rot_sensitivity=1.0,
                    sim_device="cuda:0",
                ),
            }
        )

        # === Robot USD path ===
        # The USD file is in isaaclab_mimic/controll_scripts/so_arm_101/SO-ARM101.usd
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        usd_path = os.path.join(script_dir, "controll_scripts", "so_arm_101", "SO-ARM101.usd")
        
        # === Robot configuration ===
        self.scene.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd_path,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    fix_root_link=True,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                joint_pos={
                    "shoulder_pan": 0.0,
                    "shoulder_lift": 0.0,
                    "elbow_flex": 0.0,
                    "wrist_flex": 0.0,
                    "wrist_roll": 0.0,
                    "gripper": 0.5,
                },
            ),
            actuators={
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
                    effort_limit=50.0,
                    stiffness=1000.0,
                    damping=100.0,
                ),
                "gripper": ImplicitActuatorCfg(
                    joint_names_expr=["gripper"],
                    effort_limit=2.0,
                    stiffness=100.0,
                    damping=10.0,
                ),
            },
        )

        # === End-effector frame ===
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.0),
                        rot=(1.0, 0.0, 0.0, 0.0),
                    ),
                ),
            ],
        )

        # === Mimic Data Generation Config ===
        self.datagen_config = DataGenConfig(
            name="so_arm_stack_datagen",
            max_num_failures=10,
            seed=42,
        )
        
        # Subtask configuration (required by MimicEnvCfg)
        self.subtask_configs = {
            "so_arm": [
                SubTaskConfig(
                    object_ref="cube",
                    subtask_term_signal="grasp_1",
                    subtask_term_offset_range=(5, 10),
                    selection_strategy="nearest_neighbor_object",
                    selection_strategy_kwargs={"nn_k": 1},
                    action_noise=0.01,
                    num_interpolation_steps=5,
                    num_fixed_steps=0,
                    apply_noise_during_interpolation=False,
                )
            ]
        }
