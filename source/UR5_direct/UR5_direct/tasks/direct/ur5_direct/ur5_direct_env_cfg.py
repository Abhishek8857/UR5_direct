# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg    
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from .ur5_direct_env import ASSET_DIR

OBS_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
}

STATE_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "joint_pos": 7,
    "held_pos": 3,
    "held_pos_rel_fixed": 3,
    "held_quat": 4,
    "fixed_pos": 3,
    "fixed_quat": 4,
    "task_prop_gains": 6,
    "ema_factor": 1,
    "pos_threshold": 3,
    "rot_threshold": 3,
}

@configclass
class ObsRandCfg:
    fixed_asset_pos = [0.001, 0.001, 0.001]


@configclass
class CtrlCfg:
    ema_factor = 0.2

    pos_action_bounds = [0.05, 0.05, 0.05]
    rot_action_bounds = [1.0, 1.0, 1.0]

    pos_action_threshold = [0.02, 0.02, 0.02]
    rot_action_threshold = [0.097, 0.097, 0.097]

    reset_joints = [1.5178e-03, -1.9651e-01, -1.4364e-03, -1.9761, -2.7717e-04, 1.7796, 7.8556e-01]
    reset_task_prop_gains = [300, 300, 300, 20, 20, 20]
    reset_rot_deriv_scale = 10.0
    default_task_prop_gains = [100, 100, 100, 30, 30, 30]

    # Null space parameters.
    default_dof_pos_tensor = [-1.3003, -0.4015, 1.1791, -2.1493, 0.4001, 1.9425, 0.4754]
    kp_null = 10.0
    kd_null = 6.3246
    
    
@configclass
class Ur5DirectEnvCfg(DirectRLEnvCfg):
    
    # env
    decimation = 8
    
    # - spaces definition
    action_space = 6
    observation_space = 21
    state_space = 72
    obs_order: list = ["fingertip_pos_rel_fixed",
                       "fingertip_quat", 
                       "ee_linvel", 
                       "ee_angvel"
    ]
    state_order: list = [
        "fingertip_pos",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
    ]
    episode_length_s = 5.0
    action_space = 1
    observation_space = 4
    state_space = 0

    episode_length_s = 10.0  # Probably need to override.
    
    # simulation
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1 / 120,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,  # Important to avoid interpenetration.
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_max_num_partitions=1,  # Important for stable simulation.
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # robot(s)
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="assets/UR5_robot.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyMaterialCfg(
                disable_gravity=True,
                max_penetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enable_self_collisions=False,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            collsion_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.05, 
                rest_offset=0.0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "base_joint": 0.0,
                "shoulder_pan_joint": -1.57,
                "shoulder_lift_joint": -1.57,
                "elbow_joint": 1.57,
                "wrist_1_joint": -1.57,
                "wrist_2_joint": 1.57,
                "wrist_3_joint": 0.0,
                "left_outer_knuckle": 0.0
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "ur_arm1": ImplicitActuatorCfg(
                joint_names_expr=[
                    "base_joint",
                    "shoulder_pan_joint",
                    "shoulder_lift_joint",
                    "elbow_joint"
                ],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit=150.0,
                velocity_limit=2.0,
            ),
            "ur5_arm2": ImplicitActuatorCfg(
                joint_names_expr=[
                    "wrist_1_joint",
                    "wrist_2_joint",
                    "wrist_3_joint"
                ],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit=75.0,
                velocity_limit=3.0,
            ),
            "ur5_gripper": ImplicitActuatorCfg(
                joint_names_expr=[
                    "left_outer_knuckle"
                ],
                stiffness=7500.0,
                damping=173.0,
                friction=0.1,
                armature=0.0,
                effort_limit=40.0,
                velocity_limit=0.04
            )
        }
    )

    # # custom parameters/scales
    # # - controllable joint
    # cart_dof_name = "slider_to_cart"
    # pole_dof_name = "cart_to_pole"
    # # - action scale
    # action_scale = 100.0  # [N]
    # # - reward scales
    # rew_scale_alive = 1.0
    # rew_scale_terminated = -2.0
    # rew_scale_pole_pos = -1.0
    # rew_scale_cart_vel = -0.01
    # rew_scale_pole_vel = -0.005
    # # - reset states/conditions
    # initial_pole_angle_range = [-0.25, 0.25]  # pole angle sample range on reset [rad]
    # max_cart_pos = 3.0  # reset if cart exceeds this position [m]
