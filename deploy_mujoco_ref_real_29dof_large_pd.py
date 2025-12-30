import sys
import time
import torch
import mujoco
import mujoco.viewer
import numpy as np
import faulthandler
from omegaconf import OmegaConf
from motion_lib.motion_lib_robot import MotionLibRobot

HW_DOF = 29

DEBUG = True
SIM = True
VISUAL = False
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

def copysign(a, b):
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)

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

class G1():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.num_envs = 1 
        self.num_observations = 161
        self.num_actions = 29
        self.obs_context_len=15
        
        self.scale_project_gravity = 1.0
        self.scale_dof_pos = 1.0
        self.scale_dof_vel = 0.05
        self.scale_ref_body_ang_vel_root = 0.25
        self.scale_ref_body_vel_root = 2.0

        self.scale_actions = 0.25

        self.p_gains = np.array([150., 200., 150., 250., 40., 40.,
                                 150., 200., 150., 250., 40., 40.,
                                 200., 200., 200.,
                                 90., 60., 20., 60., 40., 40., 40.,
                                 90., 60., 20., 60., 40., 40., 40.,])
        self.p_gains_tensor = torch.tensor(self.p_gains, dtype=torch.float32, device=self.device, requires_grad=False)
        self.d_gains = np.array([1.5, 2.0, 1.5, 2.5, 0.4, 0.4,
                                 1.5, 2.0, 1.5, 2.5, 0.4, 0.4,
                                 2.0, 2.0, 2.0,
                                 0.9, 0.6, 0.2, 0.6, 0.4, 0.4, 0.4,
                                 0.9, 0.6, 0.2, 0.6, 0.4, 0.4, 0.4,])
        self.d_gains_tensor = torch.tensor(self.d_gains, dtype=torch.float32, device=self.device, requires_grad=False)
        
        # self.joint_limit_lo = [-2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618, -2.5307,-2.9671,-2.7576,-0.087267,-0.87267,-0.2618,-2.618,-0.52,-0.52,-3.0892,-1.5882,-2.618,-1.0472, -1.972222054,-1.614429558,-1.614429558,-3.0892,-2.2515,-2.618,-1.0472,-1.972222054,-1.614429558,-1.614429558]
        # self.joint_limit_hi = [2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618, 2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618, 2.618, 0.52, 0.52,2.6704,2.2515,2.618,2.0944,1.972222054,1.614429558,1.614429558,2.6704,1.5882,2.618,2.0944, 1.972222054,1.614429558,1.614429558]
        self.joint_limit_lo = [-2.5307, -0.5236, -2.7576, -0.087267, -np.inf, -np.inf, -2.5307,-2.9671,-2.7576,-0.087267,-np.inf,-np.inf,-2.618,-0.52,-0.52,-3.0892,-1.5882,-2.618,-1.0472, -1.972222054,-1.614429558,-1.614429558,-3.0892,-2.2515,-2.618,-1.0472,-1.972222054,-1.614429558,-1.614429558]
        self.joint_limit_hi = [2.8798, 2.9671, 2.7576, 2.8798, np.inf, np.inf, 2.8798, 0.5236, 2.7576, 2.8798, np.inf, np.inf, 2.618, 0.52, 0.52,2.6704,2.2515,2.618,2.0944,1.972222054,1.614429558,1.614429558,2.6704,1.5882,2.618,2.0944, 1.972222054,1.614429558,1.614429558]
        self.torque_limits = torch.tensor([88.0, 139.0, 88.0, 139.0, 50.0, 50.0, 
                                88.0, 139.0, 88.0, 139.0, 50.0, 50.0, 
                                88.0, 50.0, 50.0,
                                25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
                                25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,], dtype=torch.float32, device=self.device)
        self.soft_dof_pos_limit = 0.98
        for i in range(len(self.joint_limit_lo)):
            # soft limits
            if i != 5 and i != 11 and i !=4 and i != 10:
                m = (self.joint_limit_lo[i] + self.joint_limit_hi[i]) / 2
                r = self.joint_limit_hi[i] - self.joint_limit_lo[i]
                self.joint_limit_lo[i] = m - 0.5 * r * self.soft_dof_pos_limit
                self.joint_limit_hi[i] = m + 0.5 * r * self.soft_dof_pos_limit
            
        self.default_dof_pos_np = np.array([
                -0.1,  0.0,  0.0,  0.3, -0.2, 0.0, 
                -0.1,  0.0,  0.0,  0.3, -0.2, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0, 0, 0,
                0.0, 0.0, 0.0, 0.0, 0, 0, 0,])
        
        default_dof_pos = torch.tensor(self.default_dof_pos_np, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos = default_dof_pos.unsqueeze(0)

        print(f"default_dof_pos.shape: {self.default_dof_pos.shape}")

        # prepare osbervations buffer
        self.obs_tensor = torch.zeros(1, self.num_observations*self.obs_context_len, dtype=torch.float, device=self.device, requires_grad=False)
        self.hist_obs = torch.zeros(self.num_observations*(self.obs_context_len-1), dtype=torch.float32)
        self.hist_dict = {
            "actions": torch.zeros(self.num_actions*(self.obs_context_len-1), dtype=torch.float32, device=self.device),
            "base_ang_vel": torch.zeros(3*(self.obs_context_len-1), dtype=torch.float32, device=self.device),
            "base_euler_xyz": torch.zeros(3*(self.obs_context_len-1), dtype=torch.float32, device=self.device),
            "dof_pos": torch.zeros(self.num_actions*(self.obs_context_len-1), dtype=torch.float32, device=self.device),
            "dof_vel": torch.zeros(self.num_actions*(self.obs_context_len-1), dtype=torch.float32, device=self.device),
            "ref_body_ang_vel_root": torch.zeros(3*(self.obs_context_len-1), dtype=torch.float32, device=self.device),
            "ref_body_rot_root": torch.zeros(4*(self.obs_context_len-1), dtype=torch.float32, device=self.device),
            "ref_body_vel_root": torch.zeros(3*(self.obs_context_len-1), dtype=torch.float32, device=self.device),
            "ref_joint_angles": torch.zeros(self.num_actions*(self.obs_context_len-1), dtype=torch.float32, device=self.device),
            "ref_joint_velocities": torch.zeros(self.num_actions*(self.obs_context_len-1), dtype=torch.float32, device=self.device),
        }
        
    def init_mujoco_viewer(self, robot_xml):
        self.mj_model = mujoco.MjModel.from_xml_path(robot_xml)
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

    def __init__(self):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.motor_pub_freq = 50
        self.dt = 1/self.motor_pub_freq

        self.joint_pos = torch.zeros(HW_DOF, device=self.device)
        self.joint_vel = torch.zeros(HW_DOF, device=self.device)

        # motion
        self.motion_ids = torch.arange(1).to(self.device)
        self.motion_start_times = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=False)
        self.motion_len = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=False)
        
        self.config = OmegaConf.load("sim2sim/configs/g1_ref_real_29dof_large_pd.yaml")
        # init policy
        self.init_policy()
        self.prev_action = torch.zeros(self.env.num_actions, device=self.device)
        self.start_policy = True

        # init motion library
        self._init_motion_lib()
        self._ref_motion_length = self._motion_lib.get_motion_length(self.motion_ids)
        
        if DEBUG:
            self.env.init_mujoco_viewer(robot_xml=self.config["xml_path"])
            
            motion_res_cur = self._motion_lib.get_motion_state([0], torch.tensor([0.], device=self.device))
            ref_body_pos_extend = motion_res_cur["rg_pos_t"][0]
            self.env.mj_data.qpos[7:] = self.angles
            self.env.mj_data.qpos[:3] = [0, 0, 0.78]
            mujoco.mj_forward(self.env.mj_model, self.env.mj_data)

            if VISUAL:
                for i in range(ref_body_pos_extend.shape[0]):
                    self.env.viewer.user_scn.geoms[i].pos = ref_body_pos_extend[i].cpu() + torch.tensor([1., 0., 0.])

            tau = pd_control(self.angles, 
                            self.env.mj_data.qpos[7:], 
                            self.env.p_gains, 
                            np.zeros(self.env.num_actions), 
                            self.env.mj_data.qvel[6:], 
                            self.env.d_gains)
            self.env.mj_data.ctrl[:] = tau
            mujoco.mj_step(self.env.mj_model, self.env.mj_data)
            
            self.env.viewer.sync()

        # cmd and observation
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.gravity_vec = torch.zeros((1, 3), device= self.device, dtype= torch.float32)
        self.gravity_vec[:, self.up_axis_idx] = -1
        
        self.episode_length_buf = torch.zeros(1, device=self.device, dtype=torch.long)
        self.phase = torch.zeros(1, device=self.device, dtype=torch.float)

        self.Emergency_stop = False
        self.stop = False

        time.sleep(1)

    def _init_motion_lib(self):
        self.config.step_dt = self.dt
        self._motion_lib = MotionLibRobot(self.config["motion"], num_envs=self.env.num_envs, device=self.device)
        self._motion_lib.load_motions(random_sample=False)
            
        self.motion_res = self._motion_lib.get_motion_state(self.motion_ids, torch.tensor([0.], device=self.device))
        self.motion_len[0] = self._motion_lib.get_motion_length(self.motion_ids[torch.arange(self.env.num_envs)])
        self.motion_start_times[0] = torch.zeros(len(torch.arange(self.env.num_envs)), dtype=torch.float32, device=self.device)
        self.motion_dt = self._motion_lib._motion_dt
        self.motion_start_idx = 0
        self.num_motions = self._motion_lib._num_unique_motions

    def lowlevel_state_mujoco(self):
        if DEBUG and self.start_policy and SIM:
            # imu data
            quat = self.env.mj_data.qpos[3:7]
            obs_ang_vel = torch.from_numpy(self.env.mj_data.qvel[3:6]).to(self.device) 
            self.obs_ang_vel = (obs_ang_vel + (torch.rand_like(obs_ang_vel)* 2. - 1.) * self.config.noise_scales.base_ang_vel)* self.config.obs_scales.base_ang_vel
            
            quat_xyzw = torch.tensor([
                quat[1],
                quat[2],
                quat[3],
                quat[0],
            ], device= self.device, dtype= torch.float32).unsqueeze(0)
            self.obs_projected_gravity = quat_rotate_inverse(quat_xyzw, self.gravity_vec).squeeze(0)
            obs_base_euler_xyz = get_euler_xyz(quat_xyzw)[:, :].squeeze(0)
            self.obs_base_euler_xyz = (obs_base_euler_xyz + (torch.rand_like(obs_base_euler_xyz)* 2. - 1.) * self.config.noise_scales.base_euler_xyz)* self.config.obs_scales.base_euler_xyz
            
            # motor data
            self.joint_pos = torch.from_numpy(self.env.mj_data.qpos[7:]).to(self.device)
            
            self.joint_vel = torch.from_numpy(self.env.mj_data.qvel[6:]).to(self.device)
            obs_joint_vel = self.joint_vel
            self.obs_joint_vel = (obs_joint_vel + (torch.rand_like(obs_joint_vel)* 2. - 1.) * self.config.noise_scales.dof_vel)* self.config.obs_scales.dof_vel

    def init_policy(self):
        faulthandler.enable()

        # prepare environment
        self.env = G1()

        # load policy
        self.policy = torch.jit.load(self.config["policy_path"], map_location=self.env.device)
        self.policy.to(self.env.device)

        self.angles = np.zeros(HW_DOF, dtype=np.float32)
        self.angles_last = np.zeros(HW_DOF, dtype=np.float32)
    
    def compute_observations(self):
        """ Computes observations
        """
        motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times
        self.ref_motion_phase = motion_times / self._ref_motion_length
        motion_res_cur = self._motion_lib.get_motion_state([0], motion_times)

        ref_body_pos_extend = motion_res_cur["rg_pos_t"][0]
        self.ref_joint_pos = motion_res_cur["dof_pos"][0]
        self.ref_joint_vel = motion_res_cur["dof_vel"][0]

        ref_body_ang_vel_root = (motion_res_cur["body_ang_vel_t"][0][0]) * self.config.obs_scales.ref_body_ang_vel_root
        ref_body_rot_root = motion_res_cur["rg_rot_t"][0][0] * self.config.obs_scales.ref_body_rot_root
        ref_body_vel_root = motion_res_cur["body_vel_t"][0][0] * self.config.obs_scales.ref_body_vel_root

        # reference motion
        ref_joint_angles = self.ref_joint_pos
        ref_joint_angles = (ref_joint_angles + (torch.rand_like(ref_joint_angles)* 2. - 1.) * self.config.noise_scales.ref_joint_angles)* self.config.obs_scales.ref_joint_angles

        ref_joint_velocities = self.ref_joint_vel
        ref_joint_velocities = (ref_joint_velocities + (torch.rand_like(ref_joint_velocities)* 2. - 1.) * self.config.noise_scales.ref_joint_velocities)* self.config.obs_scales.ref_joint_velocities

        dof_pos = self.joint_pos - self.env.default_dof_pos.squeeze(0)
        dof_pos = (dof_pos + (torch.rand_like(dof_pos)* 2. - 1.) * self.config.noise_scales.dof_pos)* self.config.obs_scales.dof_pos

        self.env.obs_tensor[:,:self.env.num_actions] = self.prev_action
        self.env.obs_tensor[:,self.env.num_actions:self.env.num_actions+3] = self.obs_ang_vel
        self.env.obs_tensor[:,self.env.num_actions+3:self.env.num_actions+6] = self.obs_base_euler_xyz
        self.env.obs_tensor[:,self.env.num_actions+6 : self.env.num_actions*2+6] = dof_pos
        self.env.obs_tensor[:,self.env.num_actions*2+6 : self.env.num_actions*3+6] = self.obs_joint_vel
        history_array = []
        for key in sorted(self.env.hist_dict.keys()):
            history_array.append(self.env.hist_dict[key])
        self.env.obs_tensor[:,self.env.num_actions*3+6 : self.env.num_actions*3+6+self.env.num_observations*(self.env.obs_context_len-1)] = torch.concatenate(history_array, axis=-1)
        self.env.obs_tensor[:,self.env.num_actions*3+6+self.env.num_observations*(self.env.obs_context_len-1): self.env.num_actions*3+9+self.env.num_observations*(self.env.obs_context_len-1)] = ref_body_ang_vel_root
        self.env.obs_tensor[:,self.env.num_actions*3+9+self.env.num_observations*(self.env.obs_context_len-1): self.env.num_actions*3+13+self.env.num_observations*(self.env.obs_context_len-1)] = ref_body_rot_root
        self.env.obs_tensor[:,self.env.num_actions*3+13+self.env.num_observations*(self.env.obs_context_len-1): self.env.num_actions*3+16+self.env.num_observations*(self.env.obs_context_len-1)] = ref_body_vel_root
        self.env.obs_tensor[:,self.env.num_actions*3+16+self.env.num_observations*(self.env.obs_context_len-1): self.env.num_actions*4+16+self.env.num_observations*(self.env.obs_context_len-1)] = ref_joint_angles
        self.env.obs_tensor[:,self.env.num_actions*4+16+self.env.num_observations*(self.env.obs_context_len-1): self.env.num_actions*5+16+self.env.num_observations*(self.env.obs_context_len-1)] = ref_joint_velocities

        self.env.hist_dict["actions"] = torch.concatenate([self.prev_action, self.env.hist_dict["actions"][:-self.env.num_actions]])
        self.env.hist_dict["base_ang_vel"] = torch.concatenate([self.obs_ang_vel, self.env.hist_dict["base_ang_vel"][:-3]])
        self.env.hist_dict["base_euler_xyz"] = torch.concatenate([self.obs_base_euler_xyz, self.env.hist_dict["base_euler_xyz"][:-3]])
        self.env.hist_dict["dof_pos"] = torch.concatenate([dof_pos, self.env.hist_dict["dof_pos"][:-self.env.num_actions]])
        self.env.hist_dict["dof_vel"] = torch.concatenate([self.obs_joint_vel, self.env.hist_dict["dof_vel"][:-self.env.num_actions]])
        self.env.hist_dict["ref_body_ang_vel_root"] = torch.concatenate([ref_body_ang_vel_root, self.env.hist_dict["ref_body_ang_vel_root"][:-3]])
        self.env.hist_dict["ref_body_rot_root"] = torch.concatenate([ref_body_rot_root, self.env.hist_dict["ref_body_rot_root"][:-4]])
        self.env.hist_dict["ref_body_vel_root"] = torch.concatenate([ref_body_vel_root, self.env.hist_dict["ref_body_vel_root"][:-3]])
        self.env.hist_dict["ref_joint_angles"] = torch.concatenate([ref_joint_angles, self.env.hist_dict["ref_joint_angles"][:-self.env.num_actions]])
        self.env.hist_dict["ref_joint_velocities"] = torch.concatenate([ref_joint_velocities, self.env.hist_dict["ref_joint_velocities"][:-self.env.num_actions]])

    @torch.no_grad()
    def main_loop(self):
        # keep stand up pose first
        _percent_1 = 0
        _duration_1 = 500
        firstRun = True
        init_success = False
        while not init_success and not self.start_policy:
            if firstRun:
                firstRun = False
                start_pos = self.joint_pos
            else:
                if _percent_1 < 1:
                    self.set_motor_position(q=(1 - _percent_1) * np.array(start_pos) + _percent_1 * np.array(self.env.default_dof_pos_np))
                    _percent_1 += 1 / _duration_1
                    _percent_1 = min(1, _percent_1)
                if _percent_1 == 1 and not init_success:
                    init_success = True
                    print("---Initialized---")
                self.motor_pub.publish(self.cmd_msg)

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

            if self.start_policy:
                if DEBUG and SIM:
                    self.lowlevel_state_mujoco()
                self.compute_observations()
                self.episode_length_buf += 1
                raw_actions = self.policy(self.env.obs_tensor.detach().reshape(1, -1))
                if torch.any(torch.isnan(raw_actions)):
                    self.set_gains(np.array([0.0]*HW_DOF),self.env.d_gains)
                    self.set_motor_position(q=self.env.default_dof_pos_np)
                    raise SystemExit
                self.prev_action = raw_actions.squeeze(0)
                whole_body_action = raw_actions.squeeze(0)
                
                actions_scaled = whole_body_action * self.env.scale_actions + self.env.default_dof_pos
                # p_limits_low = (-self.env.torque_limits) + self.env.d_gains_tensor*self.joint_vel
                # p_limits_high = (self.env.torque_limits) + self.env.d_gains_tensor*self.joint_vel
                # actions_low = (p_limits_low/self.env.p_gains_tensor) - self.env.default_dof_pos + self.joint_pos
                # actions_high = (p_limits_high/self.env.p_gains_tensor) - self.env.default_dof_pos + self.joint_pos
                # angles = torch.clip(actions_scaled, actions_low, actions_high) + self.env.default_dof_pos
                # self.angles = angles.cpu().numpy()
                self.angles = actions_scaled.cpu().numpy()
                inference_time=time.monotonic()-loop_start_time
                if DEBUG:

                    motion_res_cur = self._motion_lib.get_motion_state([0], (self.episode_length_buf + 1) * self.dt + self.motion_start_times)
                    ref_body_pos_extend = motion_res_cur["rg_pos_t"][0]

                    if VISUAL:
                        for i in range(ref_body_pos_extend.shape[0]):
                            self.env.viewer.user_scn.geoms[i].pos = ref_body_pos_extend[i].cpu() # + torch.tensor([1., 0., 0.])

                    action_delay_decimation = np.random.randint(self.config["action_depaly_decimation"][0], self.config["action_depaly_decimation"][1]+1)
                    for i in range(20):
                        if i == action_delay_decimation:
                            self.angles_last = self.angles.copy()
                        self.env.viewer.sync()
                        tau = pd_control(self.angles_last, 
                                        self.env.mj_data.qpos[7:], 
                                        self.env.p_gains, 
                                        np.zeros(self.env.num_actions), 
                                        self.env.mj_data.qvel[6:], 
                                        self.env.d_gains)
                        self.env.mj_data.ctrl[:] = tau
                        mujoco.mj_step(self.env.mj_model, self.env.mj_data)
                current_time = self.episode_length_buf * self.dt + self.motion_start_times
                if current_time > self._ref_motion_length:
                    break
                
                bar_length = 50
                progress = current_time / self._ref_motion_length
                filled_length = int(bar_length * progress)
                bar = '=' * filled_length + '-' * (bar_length - filled_length)
                
                sys.stdout.write(f"\rProgress: [{bar}] {int(progress * 100)}%, latency {action_delay_decimation}ms")
                sys.stdout.flush()

            while 0.02-time.monotonic()+loop_start_time>0:  #0.012473  0.019963
                pass

if __name__ == "__main__":
    dp_node = DeployNode()
    dp_node.main_loop()
