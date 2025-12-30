
# for all bodies
for i in range(self.env.mj_model.nbody):
    body_name = mujoco.mj_id2name(self.env.mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
    if body_name:  # Skip unnamed bodies
        print(f"Body {i}: {body_name}")


# for all joints
for i in range(self.env.mj_model.njnt):
    joint_name = mujoco.mj_id2name(self.env.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
    joint_type = self.env.mj_model.jnt_type[i]

    if joint_type == 0:  # mujoco.mjtJoint.mjJNT_FREE
        print(f"Free joint '{joint_name}")




body_id = mujoco.mj_name2id(self.env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

linear_vel = self.env.mj_data.cvel[body_id][0:3]   # Linear velocity
angular_vel = self.env.mj_data.cvel[body_id][3:6]  # Angular velocity
torso_quat = self.env.mj_data.xquat[body_id][:]