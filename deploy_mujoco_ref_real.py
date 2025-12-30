import os
import argparse

import numpy as np
import torch
import faulthandler
import matplotlib.pyplot as plt

import time


# from motion_lib.motion_lib_robot import MotionLibRobot
from motion_lib.skeleton import SkeletonTree
from motion_lib.motion_lib_h1 import MotionLibH1 as MotionLibRobot
from omegaconf import OmegaConf
import sys

import mujoco
import mujoco.viewer

HW_DOF = 29

WALK_STRAIGHT = False
LOG_DATA = False
USE_GRIPPPER = False
NO_MOTOR = False

HUMANOID_XML = "assets/robots/g1/g5_yes.xml"
DEBUG = True
SIM = True
USE_TORQUE = False

def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)

@torch.jit.script
def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(
        torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw), dim=-1)

def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
                            mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                            point1[0], point1[1], point1[2],
                            point2[0], point2[1], point2[2])
    
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

class G1():
    def __init__(self,task='stand'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.task = task


        self.num_envs = 1 
        if USE_TORQUE:
            self.num_observations = 121
        else:
            self.num_observations = 93
        self.num_actions = 29
        self.num_privileged_obs = None
        self.obs_context_len=16
        
        self.scale_base_lin_vel = 2.0
        self.scale_base_ang_vel = 0.25
        self.scale_project_gravity = 1.0
        self.scale_dof_pos = 1.0
        self.scale_dof_vel = 0.05
        self.scale_torque = 0.05
        self.scale_actions = 0.25
        self.scale_base_force = 0.01
        self.scaleref_motion_phase = 1.0

        self.p_gains = np.array([160., 250., 160., 250., 90., 90., 
                                 160., 250., 160., 250., 90., 90., 
                                 400., 400., 400., 
                                 45., 45., 45., 45., 20., 20., 20., 
                                 45., 45., 45., 45., 20., 20., 20.])
        self.d_gains = np.array([0.5, 0.5, 0.5, 1.0, 0.2, 0.1,
                                 0.5, 0.5, 0.5, 1.0, 0.2, 0.1, 
                                 5.0, 5.0, 5.0, 
                                 0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2,
                                 0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2])
        # self.joint_limit_lo = [-2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618, -2.5307,-2.9671,-2.7576,-0.087267,-0.87267,-0.2618,-2.618,-0.52,-0.52,-3.0892,-1.5882,-2.618,-1.0472, -1.972222054,-1.614429558,-1.614429558,-3.0892,-2.2515,-2.618,-1.0472,-1.972222054,-1.614429558,-1.614429558]
        # self.joint_limit_hi = [2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618, 2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618, 2.618, 0.52, 0.52,2.6704,2.2515,2.618,2.0944,1.972222054,1.614429558,1.614429558,2.6704,1.5882,2.618,2.0944, 1.972222054,1.614429558,1.614429558]
        self.joint_limit_lo = [-2.5307, -0.5236, -2.7576, -0.087267, -np.inf, -np.inf, -2.5307,-2.9671,-2.7576,-0.087267,-np.inf,-np.inf,-2.618,-0.52,-0.52,-3.0892,-1.5882,-2.618,-1.0472, -1.972222054,-1.614429558,-1.614429558,-3.0892,-2.2515,-2.618,-1.0472,-1.972222054,-1.614429558,-1.614429558]
        self.joint_limit_hi = [2.8798, 2.9671, 2.7576, 2.8798, np.inf, np.inf, 2.8798, 0.5236, 2.7576, 2.8798, np.inf, np.inf, 2.618, 0.52, 0.52,2.6704,2.2515,2.618,2.0944,1.972222054,1.614429558,1.614429558,2.6704,1.5882,2.618,2.0944, 1.972222054,1.614429558,1.614429558]
        self.torque_limits = [ 88., 139.,  88., 139.,  50.,  50.,  88., 139.,  88., 139.,  50.,  50.,
         88.,  50.,  50.,  25.,  25.,  25.,  25.,  25.,   5.,   5.,  25.,  25.,
         25.,  25.,  25.,   5.,   5.]
        self.soft_dof_pos_limit = 0.98
        for i in range(len(self.joint_limit_lo)):
            # soft limits
            if i != 5 and i != 11 and i !=4 and i != 10:
                m = (self.joint_limit_lo[i] + self.joint_limit_hi[i]) / 2
                r = self.joint_limit_hi[i] - self.joint_limit_lo[i]
                self.joint_limit_lo[i] = m - 0.5 * r * self.soft_dof_pos_limit
                self.joint_limit_hi[i] = m + 0.5 * r * self.soft_dof_pos_limit
            
        self.default_dof_pos_np = np.zeros(29)
        
        self.default_dof_pos_np = np.array([
                -0.1,  0.0,  0.0,  0.3, -0.2, 0.0, 
                -0.1,  0.0,  0.0,  0.3, -0.2, 0.0,
                0.0, 0.0, 0.0,
                0.4, 0.1, 0.0, 0.3, 0, 0, 0,
                0.4, -0.1, 0.0, 0.3, 0, 0, 0,])
        
        default_dof_pos = torch.tensor(self.default_dof_pos_np, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos = default_dof_pos.unsqueeze(0)

        print(f"default_dof_pos.shape: {self.default_dof_pos.shape}")

        # prepare osbervations buffer
        self.obs_tensor = torch.zeros(1, self.num_observations*(self.obs_context_len) + 96, dtype=torch.float, device=self.device, requires_grad=False)
        self.obs_buf = np.zeros(self.num_observations*self.obs_context_len, dtype=np.float32)
        # self.hist_obs = torch.zeros(1, self.num_observations * (self.obs_context_len - 1), dtype=torch.float, device=self.device, requires_grad=False)
        self.hist_obs = np.zeros(self.num_observations*(self.obs_context_len-1), dtype=np.float32)
        # self.hist_dict = {
        #     "dof_pos": np.zeros(self.num_actions*(self.obs_context_len-1), dtype=np.float32),
        #     "dof_vel": np.zeros(self.num_actions*(self.obs_context_len-1), dtype=np.float32),
        #     "base_ang_vel": np.zeros(3 * (self.obs_context_len - 1), dtype=np.float32),
        #     "projected_gravity": np.zeros(2*(self.obs_context_len-1), dtype=np.float32),
        #     "actions": np.zeros(self.num_actions * (self.obs_context_len - 1), dtype=np.float32),
        #     "torques": np.zeros(self.num_actions * (self.obs_context_len - 1), dtype=np.float32),

            # "ref_joint_angles": np.zeros(self.num_actions*(self.obs_context_len-1), dtype=np.float32),
            # "ref_joint_velocities": np.zeros(self.num_actions*(self.obs_context_len-1), dtype=np.float32),
            # "ref_motion_phase": np.zeros(1*(self.obs_context_len-1), dtype=np.float32),
        # }

    def init_mujoco_viewer(self):

        self.mj_model = mujoco.MjModel.from_xml_path(HUMANOID_XML)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = 0.001
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)


        for _ in range(28):
            add_visual_capsule(self.viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([0, 1, 0, 1]))
        self.viewer.user_scn.geoms[27].pos = [0,0,0]

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

class DeployNode():

    class WirelessButtons:
        R1 =            0b00000001 # 1
        L1 =            0b00000010 # 2
        start =         0b00000100 # 4
        select =        0b00001000 # 8
        R2 =            0b00010000 # 16
        L2 =            0b00100000 # 32
        F1 =            0b01000000 # 64
        F2 =            0b10000000 # 128
        A =             0b100000000 # 256
        B =             0b1000000000 # 512
        X =             0b10000000000 # 1024
        Y =             0b100000000000 # 2048
        up =            0b1000000000000 # 4096
        right =         0b10000000000000 # 8192
        down =          0b100000000000000 # 16384
        left =          0b1000000000000000 # 32768

    def __init__(self,task='stand'):
        
        # init subcribers & publishers
        # self.joy_stick_sub = self.create_subscription(WirelessController, "wirelesscontroller", self._joy_stick_callback, 1)
        # self.joy_stick_sub  # prevent unused variable warning
        
        self.joint_pos = np.zeros(HW_DOF)
        self.joint_vel = np.zeros(HW_DOF)
        self.joint_tau = np.zeros(HW_DOF)

        self.motor_pub_freq = 50
        self.dt = 1/self.motor_pub_freq


        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # motion
        self.motion_ids = torch.arange(1).to(self.device)
        self.motion_start_times = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=False)
        self.motion_len = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=False)
        

        self.motion_config = OmegaConf.load("configs/g1_ref_real.yaml")
        # init policy
        self.init_policy()
        self.prev_action = np.zeros(self.env.num_actions)
        self.start_policy = True


        # init motion library
        self._init_motion_lib()
        self._ref_motion_length = self._motion_lib.get_motion_length(self.motion_ids)
        
        if DEBUG:
            self.env.init_mujoco_viewer()
            self.env.mj_data.qpos[7:] = self.angles
            self.env.mj_data.qpos[:3] = [0, 0, 0.78]
            mujoco.mj_forward(self.env.mj_model, self.env.mj_data)

            # motion_res_cur = self._motion_lib.get_motion_state([0], torch.tensor([0.], device=self.device))
            # ref_body_pos_extend = motion_res_cur["rg_pos_t"][0]
            #
            # for i in range(ref_body_pos_extend.shape[0]):
            # # if i in [0, 1, 4, 7, 2, 5, 8, 16, 18, 22, 17, 19, 23, 15]:  # joint for matching
            #     self.env.viewer.user_scn.geoms[i].pos = ref_body_pos_extend[i].cpu() + torch.tensor([1., 0., 0.])

            # tau = pd_control(np.concatenate((self.angles[:19], self.angles[22:26]), axis=0),
            #                             self.env.mj_data.qpos[7:],
            #                             np.concatenate((self.env.p_gains[:19], self.env.p_gains[22:26]), axis=0),
            #                             np.zeros(self.env.num_actions),
            #                             self.env.mj_data.qvel[6:],
            #                             np.concatenate((self.env.d_gains[:19], self.env.d_gains[22:26]), axis=0))
            tau = pd_control(self.motion_res['dof_pos'][0].cpu().numpy(), self.env.mj_data.qpos[7:], self.env.p_gains,
                             np.zeros(self.env.num_actions), self.env.mj_data.qvel[6:], self.env.d_gains)
            self.env.mj_data.ctrl[:] = tau
                        # mj_step can be replaced with code that also evaluates
                        # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(self.env.mj_model, self.env.mj_data)
            self.joint_tau = tau
            self.env.viewer.sync()
            # for i, p in enumerate([self.ref_left_wrist_pos, self.ref_right_wrist_pos, self.ref_head_pos]):
            #     self.env.viewer.user_scn.geoms[i].pos = p

        # standing up
        self.stand_up = False
        self.stand_up = True

        # start
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

        # cmd and observation
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.gravity_vec = torch.zeros((1, 3), device= self.device, dtype= torch.float32)
        self.gravity_vec[:, self.up_axis_idx] = -1
        
        self.episode_length_buf = torch.zeros(1, device=self.device, dtype=torch.long)
        self.phase = torch.zeros(1, device=self.device, dtype=torch.float)

        self.Emergency_stop = False
        self.stop = False


        time.sleep(1)

        
    ##############################
    # subscriber callbacks
    ##############################

    def _init_motion_lib(self):
        xml_path = self.motion_config["xml_path"]
        motion_file = self.motion_config["motion_file"]
        self.motion_config.step_dt = self.dt
        # self._motion_lib = MotionLibRobot(self.motion_config["motion"], num_envs=self.env.num_envs, device=self.device)
        # self._motion_lib.load_motions(random_sample=False)
        self._motion_lib = MotionLibRobot(motion_file=motion_file, device="cuda", masterfoot_conifg=None,
                                          fix_height=False, multi_thread=False, mjcf_file=xml_path, extend_hand=False,
                                          extend_head=False)
        sk_tree = SkeletonTree.from_mjcf(xml_path)
        skeleton_trees = [sk_tree] * 1
        self._motion_lib.load_motions(skeleton_trees=skeleton_trees,
                                      gender_betas=[torch.zeros(17)] * 1,
                                      limb_weights=[np.zeros(10)] * 1, random_sample=False)

        self.motion_res = self._motion_lib.get_motion_state(self.motion_ids, torch.tensor([0.], device=self.device))
        self.motion_len[0] = self._motion_lib.get_motion_length(self.motion_ids[torch.arange(self.env.num_envs)])
        self.motion_start_times[0] = torch.zeros(len(torch.arange(self.env.num_envs)), dtype=torch.float32,
                                                 device=self.device)
        self.motion_dt = self._motion_lib._motion_dt
        self.motion_start_idx = 0
        self.num_motions = self._motion_lib._num_unique_motions


    def lowlevel_state_mujoco(self):
        if DEBUG and self.start_policy and SIM:
            # imu data
            quat = self.env.mj_data.qpos[3:7]
            self.obs_ang_vel = np.array(self.env.mj_data.qvel[3:6]) * self.env.scale_base_ang_vel
            
            quat_xyzw = torch.tensor([
                quat[1],
                quat[2],
                quat[3],
                quat[0],
            ], device=self.device, dtype=torch.float32).unsqueeze(0)
            self.obs_projected_gravity = get_euler_xyz(quat_xyzw)[:, :]
            # self.obs_projected_gravity = quat_rotate_inverse(quat_xyzw, self.gravity_vec)


            # motor data
            self.joint_pos = np.array(self.env.mj_data.qpos[7:])
            self.joint_vel = np.array(self.env.mj_data.qvel[6:])
            
            tau = pd_control(self.angles, self.env.mj_data.qpos[7:], self.env.p_gains,
                                np.zeros(self.env.num_actions), self.env.mj_data.qvel[6:], self.env.d_gains)
            self.joint_tau = tau



            
    ##############################
    # motor commands
    ##############################

    ##############################
    # deploy policy
    ##############################
    def init_policy(self):
        faulthandler.enable()

        # prepare environment
        self.env = G1()

        # load policy
        file_pth = os.path.dirname(os.path.realpath(__file__))
        self.policy = torch.jit.load(self.motion_config["policy_path"], map_location=self.env.device)
        self.policy.to(self.env.device)
        self.policy_encoder = torch.jit.load(self.motion_config["policy_encoder_path"], map_location=self.env.device)
        self.policy_encoder.to(self.env.device)
        # actions = self.policy(self.env.obs_buf.detach().reshape(1, -1))  # first inference takes longer time
        # self.policy = None
        # init p_gains, d_gains, torque_limits
        self.angles = self.env.default_dof_pos_np
    
    def compute_observations(self):
        """ Computes observations
        """
        motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times + 8
        self.ref_motion_phase = motion_times / self._ref_motion_length
        motion_res_cur = self._motion_lib.get_motion_state([0], motion_times)

        self.env.default_dof_pos_np = np.array((motion_res_cur["dof_pos"][0]).cpu())

        # ref_body_pos_extend = motion_res_cur["rg_pos_t"][0]
        qj = torch.from_numpy(self.joint_pos).unsqueeze(0).to('cuda')
        dqj = torch.from_numpy(self.joint_vel).unsqueeze(0).to('cuda') * self.env.scale_dof_vel
        quat = self.obs_projected_gravity
        omega = torch.from_numpy(self.obs_ang_vel).unsqueeze(0).to('cuda')

        ref_qj = motion_res_cur['dof_pos']
        ref_dqj = motion_res_cur['dof_vel'] * self.env.scale_dof_vel
        diff_qj = motion_res_cur['dof_pos'] - qj
        ref_root_rot = motion_res_cur['root_rot']
        ref_rp = get_euler_xyz(ref_root_rot)[:, :]
        ref_root_ang_vel = motion_res_cur['root_ang_vel']
        ref_root_vel = motion_res_cur['root_vel']
        ref_omega = quat_rotate_inverse(ref_root_rot, ref_root_ang_vel) * self.env.scale_base_ang_vel
        ref_root_vel = quat_rotate_inverse(ref_root_rot, ref_root_vel) * self.env.scale_base_lin_vel
        torque = torch.from_numpy(self.joint_tau[:]).unsqueeze(0).to(self.device) * self.env.scale_torque
        action = torch.from_numpy(self.prev_action).unsqueeze(0).to(self.device)
        if USE_TORQUE:
            cur_obs = torch.cat([qj, dqj, omega, quat, action, torque, ref_qj, diff_qj, ref_root_vel,
                             ref_omega, ref_dqj, ref_rp], dim=-1).float()
        else:
            cur_obs = torch.cat([qj, dqj, omega, quat, action, ref_qj, diff_qj, ref_root_vel,
                             ref_omega, ref_dqj, ref_rp], dim=-1).float()
        hist_obs = torch.from_numpy(self.env.hist_obs).unsqueeze(0).to(self.device)
        self.env.obs_tensor = torch.cat((cur_obs, hist_obs), dim=1).to('cuda')

        if USE_TORQUE:
            current_obs_a = torch.cat((qj, dqj, omega, quat, action, torque), dim=1)
        else:
            current_obs_a = torch.cat((qj, dqj, omega, quat, action), dim=1)
        hist_obs[:, 1 * self.env.num_observations:] = hist_obs[:, :-1 * self.env.num_observations].clone()
        hist_obs[:, 0 * self.env.num_observations: 1 * self.env.num_observations] = current_obs_a.clone()
        self.env.hist_obs = hist_obs.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def main_loop(self):
        # keep stand up pose first
        _percent_1 = 0
        _duration_1 = 500
        firstRun = True
        init_success = False
        while self.stand_up and not self.start_policy:
        # while True:
            if firstRun:
                firstRun = False
                start_pos = self.joint_pos
            else:
                if _percent_1 < 1:
                    # init_joint_pos = np.concatenate([self.motion_res['dof_pos'][0][:19].cpu(), np.zeros(3), self.motion_res['dof_pos'][0][19:23].cpu(), np.zeros(3)])
                    # init_joint_pos = np.concatenate([self.env.default_dof_pos_np[:15], self.motion_res['dof_pos'][0][15:19].cpu(), np.zeros(3), self.motion_res['dof_pos'][0][19:23].cpu(), np.zeros(3)])
                    # self.set_motor_position(q=(1 - _percent_1) * np.array(start_pos) + _percent_1 * init_joint_pos)
                    self.set_motor_position(q=(1 - _percent_1) * np.array(start_pos) + _percent_1 * np.array(self.env.default_dof_pos_np))
                    _percent_1 += 1 / _duration_1
                    _percent_1 = min(1, _percent_1)
                if _percent_1 == 1 and not init_success:
                    init_success = True
                    print("---Initialized---")
                if not NO_MOTOR:
                    self.motor_pub.publish(self.cmd_msg)
                    pass

        cnt = 0
        fps_ckt = time.monotonic()

        
        while True:
            loop_start_time = time.monotonic()
            
            if self.Emergency_stop:
                breakpoint()
            if self.stop:
                _percent_1 = 0
                _duration_1 = 1000
                start_pos = self.joint_pos
                while _percent_1 < 1:
                    self.set_motor_position(q=(1 - _percent_1) * np.array(start_pos) + _percent_1 * np.array(self.env.default_dof_pos_np))
                    _percent_1 += 1 / _duration_1
                    _percent_1 = min(1, _percent_1)
                break
                    

            # spin stuff
            # if self.msg_tick == self.obs_tick:
            #     rclpy.spin_once(self,timeout_sec=0.005)
            # self.obs_tick = self.msg_tick

            if self.start_policy:
                
                if DEBUG and SIM:
                    self.lowlevel_state_mujoco()
                self.compute_observations()
                self.episode_length_buf += 1
                encoder_input = torch.cat([self.env.obs_tensor[:, :self.env.num_observations], self.env.obs_tensor[:, self.env.num_observations+96:]], dim=-1)
                priv = self.policy_encoder(encoder_input).detach()
                raw_actions = self.policy(torch.cat([self.env.obs_tensor, priv], dim=-1)).detach()
                if torch.any(torch.isnan(raw_actions)):
                    self.set_gains(np.array([0.0]*HW_DOF),self.env.d_gains)
                    self.set_motor_position(q=self.env.default_dof_pos_np)
                    raise SystemExit
                self.prev_action = raw_actions.clone().detach().cpu().numpy().squeeze(0)
                whole_body_action = raw_actions.clone().detach().cpu().numpy().squeeze(0)
                
                # whole_body_action = np.pad(whole_body_action, pad_width=padding, mode='constant', constant_values=0)
                # whole_body_action  = np.concatenate((whole_body_action[:19], np.zeros(3), whole_body_action[19:23], np.zeros(3)))
                # angles = whole_body_action * self.env.scale_actions + self.env.default_dof_pos_np
                # self.angles = np.clip(angles, self.env.joint_limit_lo, self.env.joint_limit_hi)
                actions_scaled = whole_body_action * self.env.scale_actions
                p_limits_low = (-np.array(self.env.torque_limits)) + self.env.d_gains*self.joint_vel
                p_limits_high = (np.array(self.env.torque_limits)) + self.env.d_gains*self.joint_vel
                actions_low = (p_limits_low/self.env.p_gains) - self.env.default_dof_pos_np + self.joint_pos
                actions_high = (p_limits_high/self.env.p_gains) - self.env.default_dof_pos_np + self.joint_pos
                self.angles = np.clip(actions_scaled, actions_low, actions_high) + self.env.default_dof_pos_np
                # self.angles = angles
                # print("raw_actions:", raw_actions)
                # print("angles:", self.angles)
                inference_time=time.monotonic()-loop_start_time
                # while 0.009-time.monotonic()+loop_start_time > 0:
                #     pass
                if LOG_DATA:
                    self.action_hist.append(self.prev_action)
                else:
                    if not SIM:
                        # self.env.mj_data.qpos[7:] = np.concatenate((self.angles[:19], self.angles[22:26]), axis=0)
                        self.env.mj_data.qpos[7:] = self.angles
                        mujoco.mj_forward(self.env.mj_model, self.env.mj_data)
                        self.env.viewer.sync()
                    else:
                        for i in range(20):
                            self.env.viewer.sync()
                            tau = pd_control(self.angles, self.env.mj_data.qpos[7:], self.env.p_gains,
                                             np.zeros(self.env.num_actions), self.env.mj_data.qvel[6:], self.env.d_gains)

                            self.env.mj_data.ctrl[:] = tau
                            # mj_step can be replaced with code that also evaluates
                            # a policy and applies a control signal before stepping the physics.
                            mujoco.mj_step(self.env.mj_model, self.env.mj_data)
                    self.joint_tau = tau
                current_time = self.episode_length_buf * self.dt + self.motion_start_times
                if current_time > self._ref_motion_length:
                    breakpoint()
                
                bar_length = 50
                progress = current_time / self._ref_motion_length
                filled_length = int(bar_length * progress)
                bar = '=' * filled_length + '-' * (bar_length - filled_length)
                
                # 输出不换行的进度条，并刷新输出
                sys.stdout.write(f"\rProgress: [{bar}] {int(progress * 100)}%")
                sys.stdout.flush()

            while 0.02-time.monotonic()+loop_start_time>0:  #0.012473  0.019963
                pass
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='Task name: stand, stand_w_waist, wb, squat', required=False, default='stand')
    args = parser.parse_args()
    
    dp_node = DeployNode(args.task_name)

    dp_node.main_loop()
