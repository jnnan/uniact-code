"""
Humanoid Robot G1 Full-body Control System
Functions: Control G1 robot to perform full-body actions through reinforcement learning policy
Main Components:
1. G1 class: Robot environment and parameter configuration
2. DeployNode class: Policy deployment and real-time control loop
3. Utility functions: Coordinate transformation, angle processing, etc.
"""

import os
import argparse
import numpy as np
import torch
import faulthandler
import matplotlib.pyplot as plt
import time
from collections import deque
from typing import Optional
import json

from gamepad import Gamepad, parse_remote_data
from scipy.spatial.transform import Rotation as R
from motion_lib.skeleton import SkeletonTree
from motion_lib.motion_lib_h1 import MotionLibH1 as MotionLibRobot
from omegaconf import OmegaConf
import sys
import joblib

import mujoco
import mujoco.viewer

import torch_jit_utils as _rot

from proxy import MotionProxy
import threading


# ==================== Global Constant Configuration ====================
HW_DOF = 29  # Hardware degrees of freedom (29 joints of the robot)

# Feature toggle flags
WALK_STRAIGHT = False  # Whether to use straight walking mode
LOG_DATA = False  # Whether to log runtime data
USE_GRIPPPER = False  # Whether to use gripper
NO_MOTOR = False  # Whether to disable motors (for simulation testing only)

# Load motion configuration
motion_config = OmegaConf.load("configs/g1_ref_real.yaml")
HUMANOID_XML = motion_config['xml_path']  # Robot MJCF model file path
DEBUG = True  # Debug mode toggle

START_IDX_50FPS = 5196  # Motion start frame index at 50Hz frequency

SIM = DEBUG  # Simulation mode synchronized with debug mode

ONLINE_MOTION = False  # Whether to generate motions in real-time (offline loads pre-recorded motions)

# Control parameters
use_gamepad_stick = True  # Use gamepad stick input
hist_interval_step = None  # Historical observation downsampling step size
add_phase = False  # Whether to add phase information to observation
use_ref_motion = True  # Whether to use reference motion trajectory
use_future_ref = False  # Whether to use future reference frames

RPY = False  # Whether to use Roll-Pitch-Yaw Euler angle representation (otherwise use quaternion)
USE_TORQUE = False  # Whether to use torque control mode


# ==================== Joint Name Mapping ====================
# BYD robot joint naming (29 joints)
byd_joint_names = [
    'left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 
    'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_roll_joint', 
    'left_hip_yaw_joint', 'right_hip_yaw_joint', 'waist_pitch_joint', 
    'left_knee_joint', 'right_knee_joint', 
    'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 
    'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 
    'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 
    'left_ankle_roll_joint', 'right_ankle_roll_joint', 
    'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 
    'left_elbow_joint', 'right_elbow_joint', 
    'left_wrist_roll_joint', 'right_wrist_roll_joint', 
    'left_wrist_pitch_joint', 'right_wrist_pitch_joint', 
    'left_wrist_yaw_joint', 'right_wrist_yaw_joint'
]

# Joint naming order in MuJoCo simulator
mujoco_joint_names = [
    # Left leg (6 joints)
    'left_hip_pitch', 'left_hip_roll', 'left_hip_yaw', 'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
    # Right leg (6 joints)
    'right_hip_pitch', 'right_hip_roll', 'right_hip_yaw', 'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
    # Waist (3 joints)
    'waist_yaw', 'waist_roll', 'waist_pitch',
    # Left arm (7 joints)
    'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw', 'left_elbow', 
    'left_wrist_roll', 'left_wrist_pitch', 'left_wrist_yaw',
    # Right arm (7 joints)
    'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw', 'right_elbow', 
    'right_wrist_roll', 'right_wrist_pitch', 'right_wrist_yaw',
]

# Build joint index mapping table
# BYD joint order -> MuJoCo joint order
byd_joint_to_mujoco_joint = [byd_joint_names.index(joint_name+'_joint') for joint_name in mujoco_joint_names]
# MuJoCo joint order -> BYD joint order
mujoco_joint_to_byd_joint = [mujoco_joint_names.index(joint_name[:-6]) for joint_name in byd_joint_names]


# ==================== Utility Functions ====================

def my_subtract_frame_transforms(t01: torch.Tensor, q01: torch.Tensor, 
                                  t02: torch.Tensor, q02: torch.Tensor):
    """
    Compute relative coordinate transformation
    
    Function: Given transformations from coordinate system 0 to systems 1 and 2, compute the relative transformation from system 1 to system 2
    Application: Used to calculate relative differences between robot's current pose and reference motion pose
    
    Parameters:
        t01: Translation vector from coordinate system 0 to 1 [batch_size, 3]
        q01: Rotation quaternion from coordinate system 0 to 1 [batch_size, 4] (xyzw format)
        t02: Translation vector from coordinate system 0 to 2 [batch_size, 3]
        q02: Rotation quaternion from coordinate system 0 to 2 [batch_size, 4] (xyzw format)
    
    Returns:
        t12: Relative translation from coordinate system 1 to 2 [batch_size, 3]
        q12: Relative rotation quaternion from coordinate system 1 to 2 [batch_size, 4]
    """
    # Compute conjugate of q01 (equivalent to inverse rotation)
    q10 = _rot.quat_conjugate(q01)
    
    # Compute relative rotation: q12 = q10 * q02
    if q02 is not None:
        q12 = _rot.quat_mul(q10, q02)
    else:
        q12 = q10

    # Compute relative translation: first compute translation difference, then rotate to coordinate system 1
    if t02 is not None:
        t12 = _rot.quat_rotate(q10, t02 - t01)
    else:
        t12 = _rot.quat_rotate(q10, -t01)

    return t12, q12


def normalize_angle(angle):
    """
    Normalize angle to [-π, π] range
    
    Parameters:
        angle: Input angle (radians)
    
    Returns:
        Normalized angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


@torch.jit.script
def copysign(a, b):
    """
    Return |a| with the same sign as b
    Used to handle sign issues in Euler angle conversion
    
    Parameters:
        a: Numeric value (float)
        b: Tensor used to determine sign
    
    Returns:
        Tensor of |a| with the same sign as b
    """
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)


@torch.jit.script
def get_euler_xyz(q):
    """
    Convert quaternion to Euler angles (XYZ order, i.e., Roll-Pitch-Yaw)
    
    Quaternion format: [x, y, z, w]
    Euler angle order: First rotate around X-axis (Roll), then Y-axis (Pitch), finally Z-axis (Yaw)
    
    Parameters:
        q: Quaternion tensor [batch_size, 4] (xyzw format)
    
    Returns:
        Euler angle tensor [batch_size, 3] (roll, pitch, yaw)
    """
    qx, qy, qz, qw = 0, 1, 2, 3
    
    # Roll angle (rotation around X-axis)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch angle (rotation around Y-axis)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    # Handle gimbal lock issue
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

    # Yaw angle (rotation around Z-axis)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw), dim=-1)


def add_visual_capsule(scene, point1, point2, radius, rgba):
    """
    Add capsule to MuJoCo visualization scene
    Used for visualizing various information during debugging (e.g., trajectories, forces, etc.)
    
    Parameters:
        scene: MuJoCo scene object
        point1: Capsule start point [x, y, z]
        point2: Capsule end point [x, y, z]
        radius: Capsule radius
        rgba: Color [r, g, b, alpha], values 0-1
    """
    if scene.ngeom >= scene.maxgeom:
        return  # Scene geometry is full
    
    scene.ngeom += 1  # Increment geometry count
    
    # Initialize capsule geometry properties
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom-1],
        mujoco.mjtGeom.mjGEOM_CAPSULE,  # Geometry type: capsule
        np.zeros(3), np.zeros(3), np.zeros(9),
        rgba.astype(np.float32)
    )
    
    # Connect two points to create capsule
    mujoco.mjv_makeConnector(
        scene.geoms[scene.ngeom-1],
        mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
        point1[0], point1[1], point1[2],
        point2[0], point2[1], point2[2]
    )


def quat_rotate_inverse(q, v):
    """
    Rotate vector using inverse (conjugate) of quaternion
    
    Function: Transform vector from world coordinate system to robot body coordinate system
    Application: For example, transform gravity vector from world frame to robot body frame to get the gravity direction perceived by the robot
    
    Mathematical principle: v' = q* ⊗ v ⊗ q (where q* is the conjugate of q)
    
    Parameters:
        q: Quaternion [batch_size, 4] (xyzw format)
        v: Vector to rotate [batch_size, 3]
    
    Returns:
        Rotated vector [batch_size, 3]
    """
    shape = q.shape
    q_w = q[:, -1]  # Quaternion scalar part
    q_vec = q[:, :3]  # Quaternion vector part [x, y, z]
    
    # Fast computation of quaternion inverse rotation formula
    # v' = v*(2w²-1) - 2w*(q_vec × v) + 2*q_vec*(q_vec · v)
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    
    return a - b + c


# ==================== G1 Robot Environment Class ====================

class G1():
    """
    G1 Robot Environment Class
    
    Functions:
    1. Store robot physical parameters (joint limits, PD gains, etc.)
    2. Store observation normalization parameters from training (mean, standard deviation)
    3. Manage action scaling and offset
    4. Initialize MuJoCo simulator (debug mode)
    """
    
    def __init__(self, task='stand'):
        """
        Initialize G1 robot environment
        
        Parameters:
            task: Task name ('stand' indicates standing task)
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.task = task

        # ========== Observation Normalization Parameters ==========
        # These parameters come from observation statistics calculated during training, used to normalize observations to appropriate range
        # Standard deviation (shape: [151])
        self.obs_std = torch.tensor([
            0.3644, 0.3591, 0.0852, 0.2099, 0.2015, 0.0757, 0.2449, 0.2551, 0.1472,
            # ... (omitted for brevity)
        ], device='cpu')###

        # Mean (shape: [151])
        self.obs_mean = torch.tensor([
            -3.8448e-01, -3.8627e-01, -9.6048e-03,  9.8600e-02, -1.3668e-01,
            # ... (omitted for brevity)
        ], device='cpu')

        # ========== Environment Parameters ==========
        self.num_envs = 1  # Number of parallel environments (only 1 during deployment)
        
        # Observation dimensions
        if USE_TORQUE:
            self.num_observations = 121  # Observation dimension when torque information is included
        else:
            self.num_observations = 93  # Observation dimension when torque information is not included
        
        self.num_actions = 29  # Action dimension (29 joints)
        self.num_privileged_obs = None  # Privileged observations (not used during deployment)
        self.obs_context_len = 16  # Historical observation sequence length
        
        # ========== Observation Scaling Factors ==========
        # Used to scale different physical quantities to similar numerical ranges, facilitating neural network learning
        self.scale_base_lin_vel = 1.  # Base linear velocity scaling
        self.scale_base_ang_vel = 1.  # Base angular velocity scaling
        self.scale_project_gravity = 1.0  # Projected gravity scaling
        self.scale_dof_pos = 1.0  # Joint position scaling
        self.scale_dof_vel = 1.  # Joint velocity scaling
        self.scale_torque = 1.  # Torque scaling
        self.scale_base_force = 1.  # Base force scaling
        self.scaleref_motion_phase = 1.0  # Reference motion phase scaling

        # ========== Action Scaling Parameters ==========
        # Scale neural network output actions from [-1,1] range to actual joint angle variation range
        self.scale_actions = np.array([[
            0.5475, 0.5475, 0.5475, 0.3507, 0.3507, 0.4386, 0.5475, 0.5475, 0.4386,
            0.3507, 0.3507, 0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.4386,
            0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.0745, 0.0745,
            0.0745, 0.0745
        ]])
        
        # ========== Action Offset Parameters ==========
        # Action center offset to adjust actions around reference motion
        self.action_offset = np.array([[
            -3.1842e-01, -3.1324e-01, -5.9036e-03, -3.1133e-03, -6.0440e-03,
            6.0068e-03,  4.3117e-03,  6.2525e-03, -4.6817e-03,  6.7533e-01,
            6.6802e-01,  2.0053e-01,  2.0622e-01, -3.6718e-01, -3.5930e-01,
            1.9924e-01, -2.0916e-01, -4.0892e-03, -7.8090e-03,  3.4449e-03,
            5.0150e-03,  6.0501e-01,  6.0436e-01, -1.1171e-03,  6.5377e-03,
            -3.7684e-03, -5.6188e-03, -4.8534e-04, -8.4458e-03
        ]])
        
        self.action_offset_torch = torch.tensor(
            self.action_offset, 
            dtype=torch.float, 
            device=self.device, 
            requires_grad=False
        )

        # ========== PD Controller Gains ==========
        # P gain (proportional gain) - controls response strength to position error
        self.p_gains = np.array([
            40.1792, 99.0984, 40.1792, 99.0984, 28.5012, 28.5012,  # Left leg
            40.1792, 99.0984, 40.1792, 99.0984, 28.5012, 28.5012,  # Right leg
            40.1792, 28.5012, 28.5012,  # Waist
            14.2506, 14.2506, 14.2506, 14.2506, 14.2506, 16.7783, 16.7783,  # Left arm
            14.2506, 14.2506, 14.2506, 14.2506, 14.2506, 16.7783, 16.7783   # Right arm
        ])

        # D gain (derivative gain) - controls response strength to velocity error, provides damping
        self.d_gains = np.array([
            2.5579, 6.3088, 2.5579, 6.3088, 1.8144, 1.8144,  # Left leg
            2.5579, 6.3088, 2.5579, 6.3088, 1.8144, 1.8144,  # Right leg
            2.5579, 1.8144, 1.8144,  # Waist
            0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 1.0681, 1.0681,  # Left arm
            0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 1.0681, 1.0681   # Right arm
        ])

        # ========== Joint Limits ==========
        # Physical limits for each joint (radians)
        self.joint_limit_lo = [
            -2.5307, -0.5236, -2.7576, -0.087267, -np.inf, -np.inf,  # Left leg (note: ankle joints have no limits)
            -2.5307, -2.9671, -2.7576, -0.087267, -np.inf, -np.inf,  # Right leg
            -2.618, -0.52, -0.52,  # Waist
            -3.0892, -1.5882, -2.618, -1.0472, -1.972222054, -1.614429558, -1.614429558,  # Left arm
            -3.0892, -2.2515, -2.618, -1.0472, -1.972222054, -1.614429558, -1.614429558   # Right arm
        ]
        
        self.joint_limit_hi = [
            2.8798, 2.9671, 2.7576, 2.8798, np.inf, np.inf,  # Left leg
            2.8798, 0.5236, 2.7576, 2.8798, np.inf, np.inf,  # Right leg
            2.618, 0.52, 0.52,  # Waist
            2.6704, 2.2515, 2.618, 2.0944, 1.972222054, 1.614429558, 1.614429558,  # Left arm
            2.6704, 1.5882, 2.618, 2.0944, 1.972222054, 1.614429558, 1.614429558   # Right arm
        ]

        # Apply soft limits (reduce limit range by factor of 1.02)
        self.soft_dof_pos_limit = 1.02
        for i in range(len(self.joint_limit_lo)):
            # Skip ankle joints (indices 4, 5, 10, 11 are unlimited joints)
            if i != 5 and i != 11 and i != 4 and i != 10:
                m = (self.joint_limit_lo[i] + self.joint_limit_hi[i]) / 2  # Midpoint
                r = self.joint_limit_hi[i] - self.joint_limit_lo[i]  # Range
                self.joint_limit_lo[i] = m - 0.5 * r * self.soft_dof_pos_limit
                self.joint_limit_hi[i] = m + 0.5 * r * self.soft_dof_pos_limit
            
        # ========== Default Joint Positions ==========
        # Robot's initial pose (all zeros represent standard standing pose)
        self.default_dof_pos_np = np.zeros(29)
        
        # Convert to torch tensor
        default_dof_pos = torch.tensor(
            self.default_dof_pos_np, 
            dtype=torch.float, 
            device=self.device, 
            requires_grad=False
        )
        self.default_dof_pos = default_dof_pos.unsqueeze(0)  # [1, 29]

        print(f"default_dof_pos.shape: {self.default_dof_pos.shape}")

        # ========== Observation Buffer ==========
        # Tensor storing current observation
        self.obs_tensor = torch.zeros(1, 151, dtype=torch.float, device=self.device, requires_grad=False)

        # Numpy array storing historical observations
        self.hist_obs = np.zeros(self.num_observations * (self.obs_context_len - 1), dtype=np.float32)

    def init_mujoco_viewer(self):
        """
        Initialize MuJoCo simulator and visualization window
        Only used in DEBUG mode
        """
        # Load MuJoCo model from XML file
        self.mj_model = mujoco.MjModel.from_xml_path(HUMANOID_XML)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = 0.001  # Set simulation timestep to 1ms

        # Launch passive window (does not take over main loop)
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

        # Add 28 visualization capsules (for debugging visualization)
        for _ in range(28):
            add_visual_capsule(
                self.viewer.user_scn, 
                np.zeros(3), 
                np.array([0.001, 0, 0]), 
                0.05, 
                np.array([0, 1, 0, 1])  # Green
            )
        self.viewer.user_scn.geoms[27].pos = [0, 0, 0]


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """
    PD Controller: Calculate torque required to reach target position and velocity
    
    Function: Implement position-velocity PD (proportional-derivative) control
    Formula: τ = Kp*(q_target - q_current) + Kd*(dq_target - dq_current)
    
    Parameters:
        target_q: Target joint position [n_joints]
        q: Current joint position [n_joints]
        kp: Proportional gain [n_joints]
        target_dq: Target joint velocity [n_joints]
        dq: Current joint velocity [n_joints]
        kd: Derivative gain [n_joints]
    
    Returns:
        Calculated joint torque [n_joints]
    """
    return (target_q - q) * kp + (target_dq - dq) * kd


# ==================== Deployment Node Class ====================

class DeployNode():
    """
    Deployment Node Class
    
    Functions:
    1. Load reinforcement learning policy network
    2. Load reference motion library
    3. Read robot state (IMU, joint position/velocity, etc.)
    4. Compute observation vector
    5. Run policy inference
    6. Send joint control commands
    7. Implement real-time control loop (50Hz)
    """

    class WirelessButtons:
        """
        Wireless gamepad button mapping
        Each button corresponds to one bit
        """
        R1 =     0b00000001       # 1
        L1 =     0b00000010       # 2
        start =  0b00000100       # 4
        select = 0b00001000       # 8
        R2 =     0b00010000       # 16
        L2 =     0b00100000       # 32
        F1 =     0b01000000       # 64
        F2 =     0b10000000       # 128
        A =      0b100000000      # 256
        B =      0b1000000000     # 512
        X =      0b10000000000    # 1024
        Y =      0b100000000000   # 2048
        up =     0b1000000000000  # 4096
        right =  0b10000000000000 # 8192
        down =   0b100000000000000 # 16384
        left =   0b1000000000000000 # 32768

    def __init__(self, task='stand', motion_queue=None, config=None):
        """
        Initialize deployment node
        Parameters:
            task: Task name
            motion_queue: Motion queue (optional, for online motion generation)
            config: proxy configuration
        """
        super().__init__() 
        self.config = config
        self.proxy = MotionProxy(config)
        self.running = False
        # ========== State Variable Initialization ==========
        self.joint_pos = np.zeros(HW_DOF)  # Current joint position
        self.joint_vel = np.zeros(HW_DOF)  # Current joint velocity
        self.joint_tau = np.zeros(HW_DOF)  # Current joint torque

        self.initial_yaw = 0.0  # Initial yaw angle

        # ========== Control Frequency Settings ==========
        self.motor_pub_freq = 50  # Motor command publishing frequency 50Hz
        self.dt = 1 / self.motor_pub_freq  # Timestep 0.02s

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # ========== Motion Library Related ==========
        self.motion_ids = torch.arange(1).to(self.device)  # Motion ID
        self.motion_start_times = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=False)
        self.motion_len = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=False)
        
        # ========== Load Configuration and Policy ==========
        self.motion_config = OmegaConf.load("configs/g1_ref_real.yaml")
        self.motion_config["policy_encoder_path"] = self.motion_config["policy_path"].replace('.pt', '_encoder.pt')
        
        # Initialize policy network
        self.init_policy()
        self.prev_action = np.zeros(self.env.num_actions)  # Previous step's action
        self.start_policy = False  # Whether policy has started running

        # Initialize motion library
        self._init_motion_lib()

        # Reference motion total length (set to 1 hour, will actually stop when reaching end of motion library)
        self._ref_motion_length = torch.tensor([36000000000.], dtype=torch.float32, device=self.env.device)
        
        if not use_ref_motion:
            # If not using reference motion, load pre-recorded motion data
            with open(self.motion_config["motion_file"], 'rb') as f:
                self.motion_data = joblib.load(f)['000']
            self.stick_input = self.motion_data['stick_input_xy'] if 'stick_input_xy' in self.motion_data else np.zeros((1, 2), dtype=np.float32)
            self.stick_input = torch.from_numpy(self.stick_input).to(self.device, dtype=torch.float64)

        # ========== Debug Mode Initialization ==========
        if DEBUG:
            # Initialize MuJoCo visualization
            self.env.init_mujoco_viewer()
            self.env.mj_data.qpos[7:] = self.angles  # Set initial joint angles
            self.env.mj_data.qpos[:3] = [0, 0, 0.78]  # Set initial position (height 0.78m)
            mujoco.mj_forward(self.env.mj_model, self.env.mj_data)  # Forward kinematics computation

            # Compute initial PD control torque
            tau = pd_control(
                self.motion_res['dof_pos'][0].cpu().numpy(), 
                self.env.mj_data.qpos[7:], 
                self.env.p_gains,
                np.zeros(self.env.num_actions), 
                self.env.mj_data.qvel[6:], 
                self.env.d_gains
            )
            self.env.mj_data.ctrl[:] = tau

            # Execute one simulation step
            mujoco.mj_step(self.env.mj_model, self.env.mj_data)
            self.joint_tau = tau
            self.env.viewer.sync()################################################3

        # ========== Status Flags ==========
        self.stand_up = True  # Whether in standing-up phase

        # Record start time
        self.start_time = time.monotonic()

        # ========== Data Recording Buffers ==========
        self.init_buffer = 0
        self.foot_contact_buffer = []
        self.time_hist = []
        self.obs_time_hist = []
        self.angle_hist = []
        self.action_hist = []
        self.dof_pos_hist = []
        self.dof_vel_hist = []
        self.imu_hist = []
        self.ang_vel_hist = []
        self.foot_contact_hist = []
        self.tau_hist = []
        self.obs_hist = []

        # ========== Observation Related Variables ==========
        self.up_axis_idx = 2  # Up axis index (2 represents Z-axis)
        self.gravity_vec = torch.zeros((1, 3), device=self.device, dtype=torch.float32)
        self.gravity_vec[:, self.up_axis_idx] = -1  # Gravity downward [0, 0, -1]
        
        self.episode_length_buf = torch.zeros(1, device=self.device, dtype=torch.long)  # Current episode length
        self.phase = torch.zeros(1, device=self.device, dtype=torch.float)  # Motion phase

        # ========== Control Flags ==========
        self.Emergency_stop = False  # Emergency stop flag
        self.stop = False  # Normal stop flag

        # ========== Gamepad Control ==========
        self.gamepad = Gamepad(smooth=1.)  # Initialize gamepad
        self.lx = 0.0  # Left stick X-axis
        self.lr = 0.0  # Left stick Y-axis (typically used for forward/backward movement)

        self.walking_time = torch.zeros(1, device=self.device, dtype=torch.float64)  # Walking time

        self.motion_queue = motion_queue  # Motion queue (optional)

        # ========== Motion Recording ==========
        self.motion_records_pos: list[np.ndarray] = []
        self.motion_records_vel: list[np.ndarray] = []
        self.motion_records_path = "motion_records.npy"
        self.motion_records_interval = 200
        self.text_schedule: list[dict] = []
        self.text_schedule_idx = 0
        
        # Whether to use text file commands (if False, only use command line input)
        self.use_text_file = config.get('use_text_file', True) if config else True
        if self.use_text_file:
            self._init_text_schedule()
        else:
            print("[text] Using command line input mode, not reading text commands from file")

        time.sleep(1)  # Wait 1 second to ensure all components are initialized

    def _init_text_schedule(self, path: Optional[str] = None):
        """
        Read text command schedule

        Parameters:
            path: Text command file path (JSON Lines)
        """
        if path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(base_dir, "text.jsonl")

        self.text_schedule_path = path
        self.text_schedule = []
        self.text_schedule_idx = 0

        try:
            with open(path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        print(f"[text] JSON parsing failed (line {line_num}): {exc}")
                        continue

                    frame = record.get("frame")
                    text = record.get("text")
                    if frame is None or text is None:
                        print(f"[text] Line {line_num} missing frame or text field")
                        continue

                    try:
                        frame = int(frame)
                    except (TypeError, ValueError):
                        print(f"[text] Line {line_num} frame value cannot be converted to integer: {frame}")
                        continue

                    self.text_schedule.append({"frame": frame, "text": text})

            self.text_schedule.sort(key=lambda item: item["frame"])

            if self.text_schedule:
                print(f"[text] Loaded {len(self.text_schedule)} text commands from: {path}")
            else:
                print(f"[text] Text command file is empty: {path}")

        except FileNotFoundError:
            print(f"[text] Text command file not found: {path}")
        except Exception as exc:
            print(f"[text] Error reading text command file: {exc}")

    def _maybe_send_scheduled_prompt(self, frame_count: int):
        """
        Send text command to proxy when counter reaches specified frame
        Only executed in file mode
        """
        if not self.use_text_file:
            return  # Not using file mode, skip
        
        while self.text_schedule_idx < len(self.text_schedule):
            entry = self.text_schedule[self.text_schedule_idx]
            target_frame = entry["frame"]

            if frame_count < target_frame:
                break

            text = entry["text"]
            if frame_count > target_frame:
                print(f"[text] cnt={frame_count} has exceeded target frame={target_frame}, auto-skipping.")
                self.text_schedule_idx += 1
                continue

            try:
                print(f"[text] frame={target_frame} sending text command: {text}")
                self.proxy.send_start_command(text)
            except Exception as exc:
                print(f"[text] Failed to send text command: {exc}")
            finally:
                self.text_schedule_idx += 1

    def connect(self):
        """Connect to proxy and start input listener"""
        if not self.proxy.connect():
            return False
            
        # Start input listener thread (start regardless of file mode to allow manual input)
        # Token request thread will automatically start after receiving start command
        self.proxy.start_input_listener()
        
        # If using file mode, notify user
        if self.use_text_file:
            print("[text] File mode: Will automatically read and send commands from text.jsonl")
        else:
            print("[text] Command line mode: Please use command line input 'start <prompt>' to send commands")
        
        return True

    def _init_motion_lib(self):
        """
        Initialize motion library
        
        Functions:
        1. Load reference motion file
        2. Create skeleton tree structure
        3. Load motion data
        4. Set motion start time
        """
        xml_path = self.motion_config["xml_path"]
        motion_file = self.motion_config["motion_file"]
        self.motion_config.step_dt = self.dt

        if not ONLINE_MOTION:
            # Offline mode: load pre-recorded motions from file
            self._motion_lib = MotionLibRobot(
                motion_file=motion_file, 
                device="cpu", 
                masterfoot_conifg=None,
                fix_height=False, 
                multi_thread=False, 
                mjcf_file=xml_path, 
                extend_hand=False,
                extend_head=False
            )
            
            # Create skeleton tree from MJCF file
            sk_tree = SkeletonTree.from_mjcf(xml_path)
            skeleton_trees = [sk_tree] * 1
            
            # Load motion data
            self._motion_lib.load_motions(
                skeleton_trees=skeleton_trees,
                gender_betas=[torch.zeros(17)] * 1,  # SMPL body shape parameters
                limb_weights=[np.zeros(10)] * 1,  # Limb weights
                random_sample=False  # Do not random sample
            )

            # Get initial motion state (time=0)
            self.motion_res = self._motion_lib.get_motion_state(
                self.motion_ids, 
                torch.tensor([0.], device=self.device)
            )

        # Set motion start time (starting from frame 5196, i.e., 103.92 seconds)
        self.motion_start_times[0] = torch.ones(
            len(torch.arange(self.env.num_envs)), 
            dtype=torch.float32,
            device=self.device
        ) * START_IDX_50FPS / 50

        self.motion_start_idx = 0

    def lowlevel_state_mujoco(self):
        """
        Get low-level state from MuJoCo simulator
        Only used in DEBUG and SIM modes
        
        Functions:
        1. Read IMU data (pose quaternion, angular velocity)
        2. Read joint position and velocity
        3. Compute projected gravity vector
        4. Compute PD control torque
        """
        if DEBUG and SIM:
            # ========== IMU Data ==========
            quat = self.env.mj_data.qpos[3:7]  # Quaternion [w, x, y, z] (MuJoCo format)
            self.obs_ang_vel = np.array(self.env.mj_data.qvel[3:6])  # Angular velocity

            # Convert to xyzw format (PyTorch convention)
            quat_xyzw = torch.tensor([
                quat[1],  # x
                quat[2],  # y
                quat[3],  # z
                quat[0],  # w
            ], device=self.device, dtype=torch.float32).unsqueeze(0)

            # ========== Compute Euler Angles ==========
            rpy = R.from_quat(quat_xyzw.cpu().numpy())
            self.roll, self.pitch, self.yaw = rpy.as_euler('xyz', degrees=False)[0]
            self.obs_imu = np.array([self.roll, self.pitch, self.yaw]) * self.env.scale_project_gravity
            
            # Reconstruct quaternion (ensure normalization)
            rpy = R.from_euler('xyz', [self.roll, self.pitch, self.yaw], degrees=False)
            self.quat_xyzw = torch.tensor(rpy.as_quat(), device=self.device, dtype=torch.float32).unsqueeze(0)
            
            # ========== Compute Projected Gravity ==========
            if RPY:
                # Use Euler angle representation
                self.obs_projected_gravity = get_euler_xyz(self.quat_xyzw)[:, :]
            else:
                # Use quaternion inverse rotation to transform gravity from world frame to body frame
                self.obs_projected_gravity = quat_rotate_inverse(self.quat_xyzw, self.gravity_vec)

            # ========== Joint Data ==========
            self.joint_pos = np.array(self.env.mj_data.qpos[7:])  # Joint position (skip first 7 floating base DOFs)
            self.joint_vel = np.array(self.env.mj_data.qvel[6:])  # Joint velocity (skip first 6 base velocities)
            
            # ========== Compute PD Control Torque ==========
            tau = pd_control(
                self.angles,  # Target angles
                self.env.mj_data.qpos[7:],  # Current angles
                self.env.p_gains,  # P gain
                np.zeros(self.env.num_actions),  # Target velocity (0)
                self.env.mj_data.qvel[6:],  # Current velocity
                self.env.d_gains  # D gain
            )
            self.joint_tau = tau

    def set_gains(self, kp: np.ndarray, kd: np.ndarray):
        """
        Set PD controller gains
        
        Parameters:
            kp: Proportional gain array [29]
            kd: Derivative gain array [29]
        """
        self.kp = kp
        self.kd = kd

    def set_motor_position(self, q: np.ndarray):
        """
        Set motor target position
        On real hardware, will send commands to robot
        
        Parameters:
            q: Target joint position [29]
        """
        pass  # Need to implement command sending logic on actual hardware

    def init_policy(self):
        """
        Initialize policy network
        
        Functions:
        1. Create environment object
        2. Load trained policy model (TorchScript format)
        3. Execute one forward inference (JIT compilation warm-up)
        """
        faulthandler.enable()  # Enable error handling

        # Create G1 environment
        self.env = G1()

        # Load policy network (TorchScript format)
        self.policy = torch.jit.load(self.motion_config["policy_path"], map_location=self.env.device)
        self.policy.to(self.env.device)
        
        # Warm-up: First inference is usually slow (JIT compilation)
        actions = self.policy(self.env.obs_tensor.reshape(1, -1))

        # Initialize target angles to default standing pose
        self.angles = self.env.default_dof_pos_np

    def compute_observations(self):
        """
        Compute observation vector for policy network
        
        Process:
        1. Get reference motion for current time from motion library
        2. Combine current state and reference motion to form observation
        3. Update historical observation buffer
        
        Observation composition (93 dimensions):
        - Reference joint position (29 dimensions)
        - Reference joint velocity (29 dimensions)
        - Body angular velocity (3 dimensions)
        - Current joint position - offset (29 dimensions)
        - Current joint velocity (29 dimensions)
        - Projected gravity (3 dimensions)
        - Previous step action (29 dimensions)
        - Historical observations (optional)
        """
        # ========== Get Reference Motion ==========
        #motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times
        #motion_res_cur = self._motion_lib.get_motion_state([0], motion_times)
        motion_res_cur = self.proxy.get_motion_state()
        
        # Check if valid data is obtained, if None then wait or use previous frame data
        if motion_res_cur is None:
            print(f"Warning: motion_state is None for frame {self.episode_length_buf}")
            # Wait a short time then retry, maximum 3 retries
            max_retries = 3
            retry_count = 0
            while motion_res_cur is None and retry_count < max_retries:
                time.sleep(0.001)  # Wait 1ms
                motion_res_cur = self.proxy.get_motion_state()
                retry_count += 1
            
            if motion_res_cur is None:
                # If still None after retry, use previous frame data (if exists)
                if len(self.motion_records_pos) > 0:
                    print(f"Retry failed, using previous frame data (recorded {len(self.motion_records_pos)} frames), skipping record")
                    # If there were previous records, use the last frame's data as current frame (for control, but don't record)
                    ref_qj = torch.from_numpy(self.motion_records_pos[-1]).unsqueeze(0).to(self.device)
                    ref_dqj = torch.from_numpy(self.motion_records_vel[-1]).unsqueeze(0).to(self.device) * self.env.scale_dof_vel
                    # Set flag to not record this frame
                    if not hasattr(self, '_skip_record_this_frame'):
                        self._skip_record_this_frame = False
                    self._skip_record_this_frame = True
                else:
                    # If no records exist, use default values
                    print("Error: No available reference motion data and no historical records, using default values, skipping record")
                    ref_qj = torch.zeros((1, 29), dtype=torch.float32, device=self.device)
                    ref_dqj = torch.zeros((1, 29), dtype=torch.float32, device=self.device)
                    if not hasattr(self, '_skip_record_this_frame'):
                        self._skip_record_this_frame = False
                    self._skip_record_this_frame = True
        
        # If valid data is obtained (initial or after retry), process data
        if motion_res_cur is not None:
            if not hasattr(self, '_skip_record_this_frame'):
                self._skip_record_this_frame = False
            self._skip_record_this_frame = False
            # ========== Reference Motion Data ==========
            device = self.device
            ref_qj = torch.tensor(
                motion_res_cur['dof_pos'],
                dtype=torch.float32,
                device=device,
            )  # Reference joint position
            ref_dqj = torch.tensor(
                motion_res_cur['dof_vel'],
                dtype=torch.float32,
                device=device,
            ) * self.env.scale_dof_vel  # Reference joint velocity

        
        num_records = len(self.motion_records_pos)
        if (
            self.motion_records_interval > 0
            and num_records >= self.motion_records_interval
            and num_records % self.motion_records_interval == 0
        ):
            self.save_motion_records()

        # Update default joint position to current reference motion
        self.env.default_dof_pos_np = ref_qj[0].detach().cpu().numpy()

        # ========== Convert Current State to Tensor ==========
        qj = torch.from_numpy(self.joint_pos).unsqueeze(0).to(device)  # Joint position
        dqj = torch.from_numpy(self.joint_vel).unsqueeze(0).to(device) * self.env.scale_dof_vel  # Joint velocity (scaled)
        omega = torch.from_numpy(self.obs_ang_vel).unsqueeze(0).to(device)  # Angular velocity
        #ref_root_rot = motion_res_cur['root_rot']  # Reference root rotation
        
        # Previous step action
        action = torch.from_numpy(self.prev_action).unsqueeze(0).to(self.device)
        
        # ========== Compute Relative Pose ==========
        # motion_anchor_pos_b, motion_anchor_ori_b_quat = my_subtract_frame_transforms(
        #     torch.zeros(1, 3, device=self.device),  # Current position (body frame origin)
        #     self.quat_xyzw,  # Current pose
        #     torch.zeros(1, 3, device=self.device),  # Reference position (body frame origin)
        #     ref_root_rot,  # Reference pose
        # )
        
        # Convert quaternion to 6D rotation representation (more suitable for neural networks)
        #tmp_matrix = _rot.quaternion_to_matrix(motion_anchor_ori_b_quat[..., [3, 0, 1, 2]])  # [1, 3, 3]
        #anchor_ori_6d = tmp_matrix[..., :2].reshape(-1, 6)  # Take first two columns [1, 6]

        # ========== Combine Observation Vector ==========
        if use_ref_motion:
            # Convert joint order from MuJoCo to BYD
            ref_qj = ref_qj[:, mujoco_joint_to_byd_joint]
            ref_dqj = ref_dqj[:, mujoco_joint_to_byd_joint]
            qj = qj[:, mujoco_joint_to_byd_joint]
            dqj = dqj[:, mujoco_joint_to_byd_joint]

            # Record reference motion data (only record when valid data is obtained)
            if not hasattr(self, '_skip_record_this_frame') or not self._skip_record_this_frame:
                self.motion_records_pos.append(ref_qj.detach().cpu().numpy().squeeze(0))
                self.motion_records_vel.append(ref_dqj.detach().cpu().numpy().squeeze(0))
                # Output statistics every 100 frames
                if len(self.motion_records_pos) % 100 == 0:
                    print(f"[Recording Stats] Currently recorded {len(self.motion_records_pos)} frames, current frame counter: {self.episode_length_buf}")
            else:
                # Skip recording, but output debug information
                if len(self.motion_records_pos) > 0:
                    print(f"[Skip Record] Frame {self.episode_length_buf} (recorded {len(self.motion_records_pos)} frames, possible discontinuity)")
            #print("ref_dqj:", ref_dqj)
            # Concatenate observations (version without 6D pose representation)
            cur_obs = torch.cat([
                ref_qj,  # Reference joint position [29]
                ref_dqj,  # Reference joint velocity [29]
                omega,  # Angular velocity [3]
                qj - self.env.action_offset_torch,  # Current joint position minus offset [29]
                dqj,  # Current joint velocity [29]
                self.obs_projected_gravity,  # Projected gravity [3]
                action  # Previous step action [29]
            ], dim=-1).float()  # Total: 29+29+3+29+29+3+29 = 151 dimensions
        else:
            pass  # Case when not using reference motion

        # ========== Process Historical Observations ==========
        hist_obs = torch.from_numpy(self.env.hist_obs).unsqueeze(0).to(self.device)

        if hist_interval_step is not None:
            # Downsample historical observations
            hist_obs_partial = hist_obs.reshape(1, self.env.obs_context_len-1, self.env.num_observations)
            hist_obs_partial = hist_obs_partial[:, hist_interval_step::hist_interval_step]
            hist_obs_partial = hist_obs_partial.reshape(1, -1)
        else:
            hist_obs_partial = hist_obs.clone()

        # Save current observation
        self.env.obs_tensor = cur_obs.to(self.device)

        # ========== Update Historical Observation Buffer ==========
        # Observation for policy (does not include reference motion)
        current_obs_a = torch.cat((qj, dqj, omega, self.obs_projected_gravity, action), dim=1)
        # Rolling update history
        hist_obs[:, 1 * self.env.num_observations:] = hist_obs[:, :-1 * self.env.num_observations].clone()
        hist_obs[:, 0 * self.env.num_observations: 1 * self.env.num_observations] = current_obs_a.clone()
        self.env.hist_obs = hist_obs.squeeze(0).cpu().numpy()

    def save_motion_records(self, file_path: Optional[str] = None):
        """
        Save collected reference motion data as npy file.

        Parameters:
            file_path: Save path, defaults to self.motion_records_path
        """
        if file_path is None:
            file_path = self.motion_records_path

        if len(self.motion_records_pos) == 0:
            print("[motion_records] No data to save.")
            return

        try:
            data = {
                'dof_pos': np.stack(self.motion_records_pos, axis=0),
                'dof_vel': np.stack(self.motion_records_vel, axis=0),
            }
            np.save(file_path, data, allow_pickle=True)
            print(f"[motion_records] Saved {data['dof_pos'].shape[0]} records to {file_path}")
        except Exception as exc:
            print(f"[motion_records] Save failed: {exc}")

    @torch.no_grad()
    def main_loop(self):
        """
        Main control loop
        
        Divided into two phases:
        1. Stand-up phase: Smooth interpolation to initial standing pose
        2. Policy running phase: Execute reinforcement learning policy
        """
        # ========== Phase 1: Stand-up Phase ==========
        _percent_1 = 0  # Interpolation progress
        _duration_1 = 500  # Interpolation duration steps
        firstRun = True
        init_success = False
        
        while self.stand_up and not self.start_policy:
            if firstRun:
                firstRun = False
                start_pos = self.joint_pos  # Record starting position
            else:
                self.set_gains(kp=self.env.p_gains, kd=self.env.d_gains)
                
                if _percent_1 < 1:
                    # Linear interpolation to target pose
                    target_pos = (1 - _percent_1) * np.array(start_pos) + _percent_1 * np.array(self.env.default_dof_pos_np)
                    self.set_motor_position(q=target_pos)
                    _percent_1 += 1 / _duration_1
                    _percent_1 = min(1, _percent_1)
                
                if _percent_1 == 1 and not init_success:
                    init_success = True
                    print("---Initialized---")
                    self.start_policy = True  # Start policy running

                # Debug: Compare real torque and simulated torque
                debug_torque = False
                if debug_torque:
                    tau = pd_control(
                        self.angles, self.joint_pos, self.env.p_gains,
                        np.zeros(self.env.num_actions), self.joint_vel, self.env.d_gains
                    )
                    print("real_tau:", self.joint_tau)
                    print("sim_tau:", tau)
                
                current_time = self.episode_length_buf * self.dt + self.motion_start_times

        # ========== Phase 2: Policy Running Phase ==========
        cnt = 0
        self._maybe_send_scheduled_prompt(cnt)
        fps_ckt = time.monotonic()
        while not self.proxy.ready():
            pass
        try:
            while True:
                #print(cnt)
                loop_start_time = time.monotonic()
                #print(cnt)
                self._maybe_send_scheduled_prompt(cnt)
                
                # ========== Safety Check ==========
                if self.Emergency_stop:
                    breakpoint()  # Emergency stop enters debug
                if self.stop:
                    while True:
                        pass  # Normal stop suspends
                if DEBUG and SIM:
                    self.lowlevel_state_mujoco()  # Get state from simulator


                if self.start_policy:
                    # ========== 1. Get State ==========
                    # if DEBUG and SIM:
                        # self.lowlevel_state_mujoco()  # Get state from simulator
                    
                    # ========== 2. Compute Observation ==========
                    self.compute_observations()
                    self.episode_length_buf += 1  # Increment timestep counter
                    
                    # ========== 3. Policy Inference ==========
                    raw_actions = self.policy(self.env.obs_tensor).detach()
                    
                    # Check for NaN (numerical instability)
                    if torch.any(torch.isnan(raw_actions)):
                        print("NaN action detected! Stopping control...")
                        self.set_gains(np.array([0.0]*HW_DOF), self.env.d_gains)  # Turn off P gain
                        self.set_motor_position(q=self.env.default_dof_pos_np)
                        raise SystemExit
                    
                    # Save action
                    self.prev_action = raw_actions.clone().detach().cpu().numpy().squeeze(0)
                    whole_body_action = raw_actions.clone().detach().cpu().numpy().squeeze(0)
                    
                    # ========== 4. Action Scaling and Mapping ==========
                    # Apply scaling and offset
                    actions_scaled = whole_body_action * self.env.scale_actions + self.env.action_offset
                    # Joint order conversion: BYD -> MuJoCo
                    actions_scaled = actions_scaled[:, byd_joint_to_mujoco_joint]

                    # Compute final target angles (this is target DOF mode)
                    self.angles = actions_scaled.astype(np.float64)

                    #print(actions_scaled)  # Print target angles

                    # inference_time = time.monotonic() - loop_start_time
                    # print(f"Inference time: {inference_time:.4f} seconds")

                    # ========== 5. Send Commands ==========
                    if LOG_DATA:
                        self.action_hist.append(self.prev_action)
                    
                    self.set_motor_position(self.angles)
                    
                    if not NO_MOTOR and not DEBUG:
                        # Real hardware mode: send motor commands
                        # self.motor_pub.publish(self.cmd_msg)
                        pass
                    else:
                        # Debug mode: execute in simulator
                        if not SIM:
                            # Only set joint positions, do not run physics
                            self.env.mj_data.qpos[7:] = self.angles
                            mujoco.mj_forward(self.env.mj_model, self.env.mj_data)
                            self.env.viewer.sync()
                        else:
                            #pass
                            #Run PD-controlled physics simulation
                            for i in range(20):  # Simulate 20 times per control step (20ms = 0.02s)
                                self.env.viewer.sync()
                                
                                # Compute PD control torque
                                tau = pd_control(
                                    self.angles, 
                                    self.env.mj_data.qpos[7:], 
                                    self.env.p_gains,
                                    np.zeros(self.env.num_actions), 
                                    self.env.mj_data.qvel[6:], 
                                    self.env.d_gains
                                )

                                self.env.mj_data.ctrl[:] = tau
                                # Execute physics simulation step
                                mujoco.mj_step(self.env.mj_model, self.env.mj_data)
                    
                    
                    current_time = self.episode_length_buf * self.dt + self.motion_start_times
                    
                    # Check if motion length is exceeded
                    if current_time > self._ref_motion_length:
                        breakpoint()
                    
                    # Draw progress bar
                    bar_length = 50
                    progress = current_time / self._ref_motion_length
                    filled_length = int(bar_length * progress)
                    bar = '=' * filled_length + '-' * (bar_length - filled_length)
                    
                    # Output progress bar (no newline)
                    #sys.stdout.write(f"\rProgress: [{bar}] {int(progress * 100)}%")
                    #sys.stdout.flush()
                    #before_token_time = time.monotonic()
                    # while not self.proxy.ready():
                    #     pass
                    #     #print("Proxy no longer ready, waiting...")
                    #     # time.sleep(0.1)
                    #     # continue
                    # # else:
                    # token = self.proxy.get_motion_state()
                    # if token is not None:
                    #     print(f"Received token: {token}")
                    # else:
                    #     print("No token available")
                    #after_token_time = time.monotonic()-before_token_time
                    #print(f"Token time: {after_token_time:.4f} seconds")

                    # if 0.02-time.monotonic()+loop_start_time>0:
                    #     print(f"Sleep time: {0.02-time.monotonic()+loop_start_time:.4f} seconds")
                    #     time.sleep(0.02-time.monotonic()+loop_start_time)

                    #time.sleep(0.01)
                # ========== 7. Calculate and Display FPS ==========
                cnt += 1
                # if cnt == 50:
                #     dt = (time.monotonic() - fps_ckt) / cnt
                #     cnt = 0
                #     fps_ckt = time.monotonic()
                #     print(f"\nFPS: {1/dt:.2f}")
        except KeyboardInterrupt:
            #self.save_motion_records()
            raise


# ==================== Main Program Entry ====================

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task_name', 
        action='store', 
        type=str, 
        help='Task name: stand, stand_w_waist, wb, squat', 
        required=False, 
        default='stand'
    )
    parser.add_argument(
        '--use_text_file',
        action='store_true',
        help='Automatically read text commands from text.jsonl file (default: use file mode)',
        default=False
    )
    parser.add_argument(
        '--use_commandline',
        action='store_true',
        help='Use command line input mode, do not read from file (default: False)',
        default=False
    )
    args = parser.parse_args()
    
    # Determine which mode to use:
    # If both parameters are specified, command line mode takes priority
    # If neither is specified, default to file mode
    if args.use_commandline:
        use_text_file = False
        print("[Config] Using command line input mode")
    elif args.use_text_file:
        use_text_file = True
        print("[Config] Using file input mode (read from text.jsonl)")
    else:
        # Default to file mode
        use_text_file = True
        print("[Config] Default to file input mode (read from text.jsonl)")
    
    config = {
        'server_host': 'localhost',
        'server_port': 8000,
        'frequency': 50,  # 50Hz
        'ready_threshold': 30,  # ready is true when number of tokens in queue reaches this value
        'buffer_threshold': 50,
        'keep_tokens_on_new_instruction': 30,
        'keep_tokens_for_generate': 48,
        'read_batch_size': 20,
        'use_text_file': use_text_file  # Add text file mode configuration
    }
    # Create deployment node
    dp_node = DeployNode(args.task_name,config=config)
    try:
        # Connect to proxy
        if not dp_node.connect():
            print("Failed to connect to proxy")
            raise Exception(f"Failed to connect to proxy")
    except Exception as e:
        print(f"Client error: {e}")

    # Run main loop
    dp_node.main_loop()

