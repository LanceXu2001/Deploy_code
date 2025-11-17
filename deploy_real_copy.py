from typing import Union
import numpy as np
import time
import torch

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import (
        create_damping_cmd, 
        create_zero_cmd, 
        init_cmd_hg, 
        init_cmd_go,  
        MotorMode
    )
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from common.motion_lib_helper import get_motion_len
# from config import Config
from collections import deque
import onnxruntime as ort
from pynput import keyboard

import hydra
from omegaconf import OmegaConf, ListConfig
from hydra.utils import get_original_cwd
from pathlib import Path

class Controller:
    def __init__(self, config: OmegaConf) -> None:
        self.config = config
        self.current_mode = "stance"
        self.target_mode = "locomotion"
        self.mode_cfg = OmegaConf.select(self.config, "locomotion")
        self.frame_stack = int(config.frame_stack)

        self.remote_controller = RemoteController()

        # Initialize the policy network
        self.policy_path = None
        self.policy = None
        
        # Initializing process variables
        self.num_actions = len(config.dof29_joint2motor_idx)
        self.qj = np.zeros(self.num_actions, dtype=np.float32)
        self.dqj = np.zeros(self.num_actions, dtype=np.float32)
        self.all_action = np.zeros(self.num_actions, dtype=np.float32)
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = np.array(config.default_angles, dtype=np.float32)
        # self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.counter = 0
        self.kps = np.array(config.kps, dtype=np.float32)
        self.kds = np.array(config.kds, dtype=np.float32)

        self.mode_switch_requested = False
        self.is_transitioning = False
        self.transition_counter = 0
        self.mimic_finish = False

        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            # h1 uses the go msg type
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        else:
            raise ValueError("Invalid msg_type")

        # wait for the subscriber to receive data
        print("Waiting for the first low state message...")
        self.wait_for_low_state()

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

        self.start_time = time.time()

        # Initialize histories for each observation type
        # self.init_history()
        
        self.max_steps = 10  # ← 固定步数
        self.data_buffer = []

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def build_cmd(self,actions,kps,kds):
        if len(actions) != len(self.config.dof29_joint2motor_idx) or\
            len(kps) != len(self.config.dof29_joint2motor_idx) or\
            len(kds) != len(self.config.dof29_joint2motor_idx):
            raise ValueError("Length of actions, kps, kds must be equal to dof29_joint2motor_idx length")

        for i in range(len(self.config.dof29_joint2motor_idx)):
            motor_idx = self.config.dof29_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = actions[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def send_interpolate_action(self, start, end, alpha):
        return start * (1 - alpha) + end * alpha

    def move_to_target_pos(self,target_pos):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        
        kps = self.config.kps
        kds = self.config.kds
        
        # record the current pos
        init_dof_pos = np.zeros(self.num_actions, dtype=np.float32)
        for i in range(self.num_actions):
            init_dof_pos[i] = self.low_state.motor_state[self.config.dof29_joint2motor_idx[i]].q

        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            interpolated_actions = self.send_interpolate_action(init_dof_pos, target_pos, alpha)
            self.build_cmd(interpolated_actions,kps,kds)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            self.build_cmd(np.array(self.config.default_angles),self.config.kps,self.config.kds)
            self.send_cmd(self.low_cmd)
            # self.data_buffer.append({
            #     'qj': self.qj.copy(),
            #     'dqj': self.dqj.copy(),
            #     'target_dof_pos': self.stand_default_angles.copy(),
            #     'action': self.action.copy(),
            #     # 'tau': self.low_state.motor_state[0:len(self.config.leg_joint2motor_idx)].tau_est.copy(),
            # })
            time.sleep(self.config.control_dt)
        self.start_switch_mode()

    def init_history(self):
        # Initialize histories for each observation type
        self.history = {
            "action": deque(maxlen=self.frame_stack-1),
            "omega": deque(maxlen=self.frame_stack-1),
            "qj": deque(maxlen=self.frame_stack-1),
            "dqj": deque(maxlen=self.frame_stack-1),
            "gravity_orientation": deque(maxlen=self.frame_stack-1),
            "ref_motion_phase": deque(maxlen=self.frame_stack-1),
            "command_lin_vel": deque(maxlen=self.frame_stack-1),
            "command_ang_vel": deque(maxlen=self.frame_stack-1),
            "command_base_height": deque(maxlen=self.frame_stack-1),
            "command_stand": deque(maxlen=self.frame_stack-1),
            "ref_upper_body_pose": deque(maxlen=self.frame_stack-1),
            "sin_phase": deque(maxlen=self.frame_stack-1),
            "cos_phase": deque(maxlen=self.frame_stack-1),
        }

        for _ in range(self.frame_stack - 1):
            for key in self.history:
                if key in ["qj", "dqj"]:
                    self.history[key].append(torch.zeros(1, self.mode_cfg.num_joints_obs, dtype=torch.float))
                elif key in ["omega", "gravity_orientation"]:
                    self.history[key].append(torch.zeros(1, 3, dtype=torch.float))
                elif key in ["ref_motion_phase", "command_base_height", "command_stand","command_ang_vel" ,"sin_phase", "cos_phase"]:
                    self.history[key].append(torch.zeros(1, 1, dtype=torch.float))
                elif key == "command_lin_vel":
                    self.history[key].append(torch.zeros(1, 2, dtype=torch.float))
                elif key == "ref_upper_body_pose":
                    self.history[key].append(torch.zeros(1, 17, dtype=torch.float))
                elif key == "action":
                    self.history[key].append(torch.zeros(1, self.mode_cfg.num_output_actions, dtype=torch.float))
                else:
                    raise ValueError(f"Not Implement: {key}")

    def check_mode_switch(self):
        if self.remote_controller.button[KeyMap.B] == 1 and not self.mode_switch_requested:
            self.mode_switch_requested = True
            self.start_switch_mode()
        # elif (self.current_mode == "mimic" and self.mimic_finish) == True and not self.mode_switch_requested:
        #     self.mode_switch_requested = True
        #     self.start_switch_mode()
        elif self.remote_controller.button[KeyMap.B] == 0:
            self.mode_switch_requested = False  # 重置标志，等待下次按下

    def start_switch_mode(self):
        # record the current pos
        self.switch_mode_start_pos = np.zeros(len(self.config.dof29_joint2motor_idx), dtype=np.float32)
        for i in range(len(self.config.dof29_joint2motor_idx)):
            self.switch_mode_start_pos[i] = self.low_state.motor_state[self.config.dof29_joint2motor_idx[i]].q

        if self.current_mode == "mimic" or self.current_mode == "stance":
            self.target_mode = "locomotion"
            self.current_mode = self.target_mode
            self.target_cfg = OmegaConf.select(self.config, "locomotion")
            self.mode_cfg = self.target_cfg
        elif self.current_mode == "locomotion":
            self.target_mode = "mimic"
            self.target_cfg = OmegaConf.select(self.config, "mimic")

        policy_path = str(Path(get_original_cwd()) / self.mode_cfg.policy_path)
        self.policy = ort.InferenceSession(policy_path)
        self.action = np.zeros(self.mode_cfg.num_output_actions, dtype=np.float32)
        self.default_angles = np.array(list(self.mode_cfg.default_angles), dtype=np.float32)
        self.init_history()

        self.mimic_finish = False
        self.is_transitioning = True   
        self.counter = 0
        self.transition_counter = 0

    def complete_switch_mode(self):
        self.current_mode = "mimic"
        self.mode_cfg = OmegaConf.select(self.config, "mimic")
        self.motion_file = str(Path(get_original_cwd()) / self.mode_cfg.motion_file)
        self.motion_len = get_motion_len(self.motion_file)
        
        policy_path = str(Path(get_original_cwd()) / self.mode_cfg.policy_path)
        self.policy = ort.InferenceSession(policy_path)
        self.action = np.zeros(self.mode_cfg.num_output_actions, dtype=np.float32)
        self.default_angles = np.array(list(self.mode_cfg.default_angles), dtype=np.float32)
        self.init_history()
        self.mimic_finish = False
        self.is_transitioning = False
        self.counter = 0
    
    def get_body_interpolation(self):
        total_time = 5.0
        num_steps = int(total_time / self.config.control_dt)
        alpha = min(self.transition_counter / num_steps, 1.0)

        target_pos = np.array(self.target_cfg.init_pos, dtype=np.float32)
        interpolated_pos = self.switch_mode_start_pos * (1 - alpha) + target_pos * alpha
        interpolated_action = (interpolated_pos - self.mode_cfg.default_angles) / self.config.action_scale
        
        self.transition_counter += 1
        
        # 检查是否完成
        if self.transition_counter >= num_steps:
            if self.target_mode == "mimic" and self.current_mode == "locomotion":
                self.complete_switch_mode()
            # else:
            #     self.is_transitioning = False
            #     print("\n[Switch] Transition complete")
        
        return interpolated_action


    def get_obs(self):
        # Get the current joint position and velocity
        for i in range(len(self.config.dof29_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.dof29_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.dof29_joint2motor_idx[i]].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)
        # if self.config.imu_type == "torso":
            # h1 and h1_2 imu is on the torso
            # imu data needs to be transformed to the pelvis frame
            # waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            # waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            # quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)

        # Build history tensors
        action_hist_tensor = torch.cat([self.history["action"][i] for i in range(self.frame_stack-1)], dim=1)
        omega_hist_tensor = torch.cat([self.history["omega"][i] for i in range(self.frame_stack-1)], dim=1)
        qj_hist_tensor = torch.cat([self.history["qj"][i] for i in range(self.frame_stack-1)], dim=1)
        dqj_hist_tensor = torch.cat([self.history["dqj"][i] for i in range(self.frame_stack-1)], dim=1)
        action_mask = np.array(self.mode_cfg.output_action_mask)
        # obs_joint_mask = np.array(self.mode_cfg.obs_joint_mask)
        # action_hist_tensor = torch.cat([self.history["action"][i][:, action_mask] for i in range(self.frame_stack-1)], dim=1)
        # omega_hist_tensor = torch.cat([self.history["omega"][i] for i in range(self.frame_stack-1)], dim=1)
        # qj_hist_tensor = torch.cat([self.history["qj"][i][:, obs_joint_mask] for i in range(self.frame_stack-1)], dim=1)
        # dqj_hist_tensor = torch.cat([self.history["dqj"][i][:, obs_joint_mask] for i in range(self.frame_stack-1)], dim=1)
        gravity_orientation_hist_tensor = torch.cat([self.history["gravity_orientation"][i] for i in range(self.frame_stack-1)], dim=1)
        ref_motion_phase_hist_tensor = torch.cat([self.history["ref_motion_phase"][i] for i in range(self.frame_stack-1)], dim=1)
        command_lin_vel_hist_tensor = torch.cat([self.history["command_lin_vel"][i] for i in range(self.frame_stack-1)], dim=1)
        command_ang_vel_hist_tensor = torch.cat([self.history["command_ang_vel"][i] for i in range(self.frame_stack-1)], dim=1)
        command_base_height_hist_tensor = torch.cat([self.history["command_base_height"][i] for i in range(self.frame_stack-1)], dim=1)
        command_stand_hist_tensor = torch.cat([self.history["command_stand"][i] for i in range(self.frame_stack-1)], dim=1)
        ref_upper_body_pose_hist_tensor = torch.cat([self.history["ref_upper_body_pose"][i] for i in range(self.frame_stack-1)], dim=1)
        sin_phase_hist_tensor = torch.cat([self.history["sin_phase"][i] for i in range(self.frame_stack-1)], dim=1)
        cos_phase_hist_tensor = torch.cat([self.history["cos_phase"][i] for i in range(self.frame_stack-1)], dim=1)

        # 2. Concatenate all parts into a single observation tensor
        if self.current_mode == "mimic":
            obs_hist = torch.cat([
                action_hist_tensor, #23*4
                omega_hist_tensor,  #3*4
                qj_hist_tensor,     #23*4
                dqj_hist_tensor,    #23*4
                gravity_orientation_hist_tensor, #3*4
                ref_motion_phase_hist_tensor #1*4
            ], dim=1)
        elif self.current_mode == "locomotion":
            obs_hist = torch.cat([
                action_hist_tensor,
                omega_hist_tensor,
                command_ang_vel_hist_tensor,
                command_base_height_hist_tensor,
                command_lin_vel_hist_tensor,
                command_stand_hist_tensor,
                cos_phase_hist_tensor,
                qj_hist_tensor,
                dqj_hist_tensor,
                gravity_orientation_hist_tensor,
                ref_upper_body_pose_hist_tensor,
                sin_phase_hist_tensor,
            ], dim=1)
    
        # create observation #TODO: adapt to different policy obs scale
        gravity_orientation = get_gravity_orientation(quat)
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        qj_obs = (qj_obs - self.default_angles) * self.config.dof_pos_scale
        dqj_obs = dqj_obs * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale

        #select observed joints if input joint dim is less than 29
        if self.mode_cfg.num_joints_obs < len(self.config.dof29_joint2motor_idx):
            qj_obs = qj_obs[np.array(self.mode_cfg.obs_joint_mask)]
            dqj_obs = dqj_obs[np.array(self.mode_cfg.obs_joint_mask)]

        num_actions = self.mode_cfg.num_output_actions
        # put into observation buffer
        if self.current_mode == "mimic" and self.mimic_finish == False:
            ref_motion_phase = ((self.counter * self.config.control_dt) % self.motion_len) / self.motion_len
            # ref_motion_phase = (self.counter * self.config.control_dt) / self.motion_len
            # if ref_motion_phase >= 1:
            #     self.mimic_finish = True
            curr_obs = np.zeros(self.mode_cfg.num_obs, dtype=np.float32)
            curr_obs[: num_actions] = self.action
            curr_obs[num_actions: num_actions + 3] = ang_vel
            curr_obs[num_actions + 3: 2 * num_actions + 3] = qj_obs
            curr_obs[2 * num_actions + 3: 3 * num_actions + 3] = dqj_obs
            curr_obs[3 * num_actions + 3: 3 * num_actions + 6] = gravity_orientation
            curr_obs[6 + 3 * num_actions] = ref_motion_phase

            curr_obs_tensor = torch.from_numpy(curr_obs).unsqueeze(0)

            self.obs_buf = torch.cat([
                curr_obs_tensor[:, :3 * num_actions + 3], 
                obs_hist, 
                curr_obs_tensor[:, 3 * num_actions + 3:]], 
                dim=1
                )
            

            self.history["action"].appendleft(curr_obs_tensor[:, :num_actions])
            self.history["omega"].appendleft(curr_obs_tensor[:, num_actions: num_actions + 3])
            self.history["qj"].appendleft(curr_obs_tensor[:, num_actions + 3: 2 * num_actions + 3])
            self.history["dqj"].appendleft(curr_obs_tensor[:, 2 * num_actions + 3: 3 * num_actions + 3])
            self.history["gravity_orientation"].appendleft(curr_obs_tensor[:, 3 * num_actions + 3: 3 * num_actions + 6])
            self.history["ref_motion_phase"].appendleft(curr_obs_tensor[:, -1].unsqueeze(0))


        elif self.current_mode == "locomotion":
            ang_vel_command = np.array([[0.0]])
            base_height_command = np.array([[0.78]]) * 2
            lin_vel_command = np.array([[0.0, 0.0]])
            stand_command = np.array([[0]])  # default stand command
            cos_phase = np.array([[1.]])
            ref_upper_dof_pose = np.array([0.,0.,0.,0.,0.3,0.,1.,0.,0.,0.,0.,-0.3,0.,1.,0.,0.,0.])
            sin_phase = np.array([[0.0]])
            curr_obs = np.zeros(self.mode_cfg.num_obs, dtype=np.float32)
            curr_obs[:num_actions] = self.action
            curr_obs[num_actions: num_actions + 3] = ang_vel
            curr_obs[num_actions + 3] = ang_vel_command #default ang_vel_command
            curr_obs[num_actions + 4] = base_height_command #default base_height_command
            curr_obs[num_actions + 5:num_actions + 7] = lin_vel_command #default base_vel_command
            curr_obs[num_actions + 7] = stand_command #default stand_command
            curr_obs[num_actions + 8] = cos_phase
            curr_obs[num_actions + 9:num_actions + 9 + 29] = qj_obs
            curr_obs[num_actions + 9 + 29:num_actions + 9 + 2*29] = dqj_obs
            curr_obs[num_actions + 9 + 2*29:num_actions + 12 + 2*29] = gravity_orientation
            curr_obs[num_actions + 12 + 2*29:num_actions + 29 + 2*29] = ref_upper_dof_pose
            curr_obs[num_actions + 29 + 2*29] = sin_phase
            curr_obs_tensor = torch.from_numpy(curr_obs).unsqueeze(0)

            self.obs_buf = torch.cat([
                curr_obs_tensor[:, :num_actions + 9 + 2*29], 
                obs_hist, 
                curr_obs_tensor[:, num_actions + 9 + 2*29:]], 
                dim=1
                )
            self.history["action"].appendleft(curr_obs_tensor[:, :num_actions])
            self.history["omega"].appendleft(curr_obs_tensor[:, num_actions: num_actions + 3])
            self.history["command_ang_vel"].appendleft(curr_obs_tensor[:, num_actions + 3].unsqueeze(0))
            self.history["command_base_height"].appendleft(curr_obs_tensor[:, num_actions + 4].unsqueeze(0))
            self.history["command_lin_vel"].appendleft(curr_obs_tensor[:,num_actions + 5:num_actions + 7])
            self.history["command_stand"].appendleft(curr_obs_tensor[:, num_actions + 7].unsqueeze(0))
            self.history["cos_phase"].appendleft(curr_obs_tensor[:, num_actions + 8].unsqueeze(0))
            self.history["qj"].appendleft(curr_obs_tensor[:, num_actions + 9:num_actions + 9 + 29])
            self.history["dqj"].appendleft(curr_obs_tensor[:, num_actions + 9 + 29:num_actions + 9 + 2*29])
            self.history["gravity_orientation"].appendleft(curr_obs_tensor[:, num_actions + 9 + 2*29:num_actions + 12 + 2*29])
            self.history["ref_upper_body_pose"].appendleft(curr_obs_tensor[:, num_actions + 12 + 2*29:num_actions + 29 + 2*29])
            self.history["sin_phase"].appendleft(curr_obs_tensor[:, num_actions + 29 + 2*29].unsqueeze(0))
        
        # self.history["action"].appendleft(torch.from_numpy(self.all_action).unsqueeze(0))
        # self.history["omega"].appendleft(torch.from_numpy(ang_vel).unsqueeze(0))
        # self.history["qj"].appendleft(torch.from_numpy(self.qj).unsqueeze(0))
        # self.history["dqj"].appendleft(torch.from_numpy(self.dqj).unsqueeze(0))
        # self.history["gravity_orientation"].appendleft(torch.from_numpy(gravity_orientation).unsqueeze(0))

    def run(self):

        self.counter += 1

        # Check mode switch
        self.check_mode_switch()
        # update obs
        self.get_obs()

        # Get policy's infered action
        input_name = self.policy.get_inputs()[0].name
        outputs = self.policy.run(None, {input_name: self.obs_buf.numpy()})
        self.action = outputs[0].squeeze()
        
        # full_action = np.zeros(len(self.config.dof29_joint2motor_idx), dtype=np.float32)
        # mask = np.array(self.mode_cfg.output_action_mask)
        # full_action[mask] = self.action
        # target_dof_pos = self.default_angles + full_action * self.config.action_scale
        # if self.need_interpolation == True and self.mimic_finish == True:

        # Get interpolated actions during mode switch
        self.all_action = np.zeros(len(self.config.dof29_joint2motor_idx), dtype=np.float32)
        mask = np.array(self.mode_cfg.output_action_mask)
        self.all_action[mask] = self.action
        if self.is_transitioning:
            interpolated_actions = self.get_body_interpolation()
            self.all_action[~mask] = interpolated_actions[~mask]    
        
        # if self.need_interpolation and not self.transition_is_complete:
        #     # interpolate to the default pos
        #     total_time = 5
        #     num_step = int(total_time / self.config.control_dt)
        #     alpha = self.counter / num_step
        #     target_pos = np.array(self.mode_cfg.init_pos)
        #     interpolated_actions = self.send_interpolate_action(self.switch_mode_start_pos, target_pos, alpha)
        #     interpolated_actions -= target_pos
        #     interpolated_actions /= self.config.action_scale
        #     self.all_action[~mask] = interpolated_actions[~mask]
        #     if self.counter >= num_step:
        #         self.need_interpolation = False
        #         self.transition_is_complete = True

        target_dof_pos = self.default_angles + self.all_action * self.config.action_scale 

        # Build low cmd
        self.build_cmd(target_dof_pos,self.mode_cfg.kps,self.mode_cfg.kds) #TODO: adapt to different policy

        # send the command
        self.send_cmd(self.low_cmd)

        time.sleep(self.config.control_dt)

@hydra.main(config_path="configs", config_name="g1_29", version_base="1.1")
def main(cfg: OmegaConf):
    # Initialize DDS communication
    print("Initializing DDS communication...")
    net = cfg.get("net_interface", "lo")
    ChannelFactoryInitialize(1, net)
    print("DDS communication initialized.")

    controller = Controller(cfg)

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_target_pos(np.array(controller.config.default_angles))

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()

    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")

if __name__ == "__main__":
    main()