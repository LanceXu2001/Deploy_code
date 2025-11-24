from typing import Union, Dict
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
from common.rotation_helper import (
    get_gravity_orientation, 
    quat_mul, 
    matrix_from_quat, 
    quat_inverse, 
    get_euler_xyz,
    quat_rotate_inverse_np,
    axis_angle_to_quaternion,
    quaternion_to_matrix,
    matrix_to_quaternion,
    wxyz_to_xyzw,
)
from common.remote_controller import RemoteController, KeyMap
from common.motion_lib_helper import (
    get_motion_len, 
    load_pkl_motion,
    from_mjcf,
    setup_init_frame_from_pkl,
    get_robot_frame_anchor_from_pkl,
    interpolate_motion_at_times,
    compute_local_key_body_positions,
    _compute_velocity,
    _compute_angular_velocity)
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
        self.old_action = np.zeros(self.num_actions, dtype=np.float32)
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
        self.init_history(first_init=True)
        
        self.max_steps = 10  # ← 固定步数
        self.data_buffer = []
    
        self.motion_data = load_pkl_motion(str(Path(get_original_cwd()) / self.config.mimic.motion_file))
        self.xml_data = from_mjcf(str(Path(get_original_cwd()) / self.config.mimic.mjcf_xml_path))
        self._parents = self.xml_data["parent_indices"]
        self._offsets = self.xml_data["local_translation"][None, ]
        self._local_rotation = self.xml_data["local_rotation"][None, ]
        self.body_names = self.xml_data['node_names']
        self.extend_config()
        self.prepare_fk()
        self.robot_quat_init = np.array([0, 0, 0, 1])  # 机器人初始旋转 (xyzw)
        self.fn_ref_to_robot_frame = setup_init_frame_from_pkl(self.robot_quat_init, self.motion_data)

    def prepare_fk(self):
        print("Computing forward kinematics for all frames during initialization...")
        from common.motion_lib_helper import forward_kinematics
        pose_aa = torch.from_numpy(self.motion_data['pose_aa'][None,]).clone()  # [num_frames, num_joints, 3]
        root_trans = torch.from_numpy(self.motion_data['root_trans_offset']).clone()  # [num_frames, 3]
        root_rot = torch.from_numpy(self.motion_data['root_rot']).clone()  # [num_frames, 4]
        num_bodies = 27  # 默认body数量
        B,seq_len = pose_aa.shape[:2]
        pose = pose_aa[..., :len(self._parents), :]
        pose_quat = axis_angle_to_quaternion(pose.clone())
        pose_mat = quaternion_to_matrix(pose_quat)

        if pose_mat.shape != 5:
            pose_mat = pose_mat.reshape(B, seq_len, -1, 3, 3)
        
        pose_trans_global, pose_rot_mat_global = forward_kinematics(pose_mat[:, :, 1:], pose_mat[:, :, 0:1], root_trans, self._offsets, self._parents, self._local_rotation_mat)
        
        pose_quat_global = wxyz_to_xyzw(matrix_to_quaternion(pose_rot_mat_global))

        root_vel_global = _compute_velocity(pose_trans_global[:, :,0:1,:], 1.0/30.0)
        root_ang_vel_global = _compute_angular_velocity(pose_quat_global[:, :,0:1,:], 1.0/30.0)
        # 将计算结果存储到motion_data中，后续直接使用
        self.motion_data['pose_trans_global'] = pose_trans_global
        self.motion_data['pose_quat_global'] = pose_quat_global
        self.motion_data['root_vel_global'] = root_vel_global
        self.motion_data['root_ang_vel_global'] = root_ang_vel_global
        # print(f"Forward kinematics computed. Shape: {pose_trans_global.shape}, {pose_quat_global.shape}")

    def extend_config(self):
        for extend_config in self.config.mimic.extend_config:
        # self.body_names_augment += [extend_config.joint_name]
            self._parents = torch.cat([self._parents, torch.tensor([self.body_names.index(extend_config.parent_name)])], dim = 0)
            self._offsets = torch.cat([self._offsets, torch.tensor([[extend_config.pos]])], dim = 1)
            self._local_rotation = torch.cat([self._local_rotation, torch.tensor([[extend_config.rot]])], dim = 1)
            # self.num_extend_dof += 1
        self._local_rotation_mat = quaternion_to_matrix(self._local_rotation).float()

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

    def init_history(self,first_init=False):
        # Initialize histories for each observation type
        if first_init:
            self.history = {
                "ref_motion_phase": deque(maxlen=10),
                "command_lin_vel": deque(maxlen=10),
                "command_ang_vel": deque(maxlen=10),
                "command_base_height": deque(maxlen=10),
                "command_stand": deque(maxlen=10),
                "ref_upper_body_pose": deque(maxlen=10),
                "sin_phase": deque(maxlen=10),
                "cos_phase": deque(maxlen=10),
            }

            self.abs_history = {
                "action": deque(maxlen=10),
                "omega": deque(maxlen=10),
                "qj": deque(maxlen=10),
                "dqj": deque(maxlen=10),
                "gravity_orientation": deque(maxlen=10),
                "roll_pitch": deque(maxlen=10),
            }

            for _ in range(self.mode_cfg.history_len):
                for key in self.abs_history:
                    if key in ["qj", "dqj"]:
                        self.abs_history[key].append(torch.zeros(1, 29, dtype=torch.float))
                    elif key in ["omega", "gravity_orientation"]:
                        self.abs_history[key].append(torch.zeros(1, 3, dtype=torch.float))
                    elif key == "action":
                        self.abs_history[key].append(torch.zeros(1, 29, dtype=torch.float))
                    elif key == "roll_pitch":
                        self.abs_history[key].append(torch.zeros(1, 2, dtype=torch.float))
                    else:
                        raise ValueError(f"Not Implement: {key}")

        for _ in range(self.mode_cfg.history_len):
            for key in self.history:
                if key in ["ref_motion_phase", "command_base_height", "command_stand","command_ang_vel" ,"sin_phase", "cos_phase"]:
                    self.history[key].append(torch.zeros(1, 1, dtype=torch.float))
                elif key == "command_lin_vel":
                    self.history[key].append(torch.zeros(1, 2, dtype=torch.float))
                elif key == "ref_upper_body_pose":
                    self.history[key].append(torch.zeros(1, 17, dtype=torch.float))
                else:
                    raise ValueError(f"Not Implement: {key}")

    def check_mode_switch(self):
        if self.remote_controller.button[KeyMap.B] == 1 and not self.mode_switch_requested:
            self.mode_switch_requested = True
            self.start_switch_mode()
        elif (self.current_mode == "mimic" and self.mimic_finish == True) and not self.mode_switch_requested:
            self.mode_switch_requested = True
            self.start_switch_mode()
        elif self.remote_controller.button[KeyMap.B] == 0:
            self.mode_switch_requested = False  # 重置标志，等待下次按下

    def start_switch_mode(self):
        # record the current pos
        self.switch_mode_start_pos = np.zeros(len(self.config.dof29_joint2motor_idx), dtype=np.float32)
        for i in range(len(self.config.dof29_joint2motor_idx)):
            self.switch_mode_start_pos[i] = self.low_state.motor_state[self.config.dof29_joint2motor_idx[i]].q

        if self.current_mode == "mimic" or self.current_mode == "stance":
            # directly change 'mimic' -> 'locomotion' to keep balance
            print("switching to locomotion mode")
            self.target_mode = "locomotion"
            self.mode_cfg = OmegaConf.select(self.config, "locomotion")
            self.target_cfg = self.mode_cfg # target_cfg defines target init pos
            
            #directly load locomotion policy
            policy_path = str(Path(get_original_cwd()) / self.mode_cfg.policy_path)
            self.policy = ort.InferenceSession(policy_path)
            self.action = np.zeros(self.mode_cfg.num_output_actions, dtype=np.float32)
            self.default_angles = np.array(list(self.mode_cfg.default_angles), dtype=np.float32)
            
            self.mimic_finish = False
        elif self.current_mode == "locomotion":
            # still in 'locomotion' until finish interpolation
            self.target_mode = "mimic"
            self.target_cfg = OmegaConf.select(self.config, "mimic")

        self.is_transitioning = True   
        self.transition_counter = 0

    def complete_switch_mode(self):
        """
        only get in when interpolation finished, set current_mode to target_mode
        """
        self.current_mode = self.target_mode
        self.mode_cfg = OmegaConf.select(self.config, self.current_mode)

        if self.current_mode == "mimic":
            self.motion_file = str(Path(get_original_cwd()) / self.mode_cfg.motion_file)
            self.motion_len = get_motion_len(self.motion_file)
        
        policy_path = str(Path(get_original_cwd()) / self.mode_cfg.policy_path)
        self.policy = ort.InferenceSession(policy_path)
        self.action = np.zeros(self.mode_cfg.num_output_actions, dtype=np.float32)
        self.default_angles = np.array(list(self.mode_cfg.default_angles), dtype=np.float32)
        self.init_history()
        self.is_transitioning = False
        self.counter = 0
    
    def get_body_interpolation(self):
        total_time = 2.0
        num_steps = int(total_time / self.config.control_dt)
        alpha = self.transition_counter / num_steps
        target_pos = np.array(self.target_cfg.init_pos, dtype=np.float32)
        interpolated_pos = self.switch_mode_start_pos * (1 - alpha) + target_pos * alpha
        action_scale = np.array(self.mode_cfg.action_scale, dtype=np.float32)
        interpolated_action = (interpolated_pos - self.mode_cfg.default_angles) / action_scale

        self.transition_counter += 1
        
        if self.transition_counter > num_steps:
            print("Interpolation complete")
            self.complete_switch_mode()
        # print("transition_counter:", self.transition_counter)
        
        return interpolated_action

    def get_obs_anchor_ref_rot(
        self,
        robot_quat: np.ndarray,  # 机器人当前旋转 (xyzw)
        robot_frame_anchor_rot: torch.Tensor  # 从 get_robot_frame_anchor_from_pkl 获取的旋转 [4]
    ) -> torch.Tensor:
        """
        计算 anchor_ref_rot 观测值（6维）
        
        Args:
            robot_quat: 机器人当前旋转 (numpy array, xyzw格式)
            robot_frame_anchor_rot: 参考锚点旋转，在机器人坐标系中 [4] (xyzw)
        
        Returns:
            anchor_ref_rot: [6] 旋转矩阵的前两列
        """
        root_quat = torch.from_numpy(robot_quat).float()  # [4]
        
        # 计算相对旋转
        relative_rot = quat_mul(
            quat_inverse(root_quat, w_last=True).unsqueeze(0),
            robot_frame_anchor_rot.unsqueeze(0),
            w_last=True,
        )
        
        # 转换为旋转矩阵并取前两列
        rot_matrix = matrix_from_quat(relative_rot)  # [1, 3, 3]
        anchor_ref_rot = rot_matrix[..., :2].reshape(-1)  # [6]
        
        return anchor_ref_rot

    def get_future_motion_and_next_step_from_pkl(
        self,
        motion_data: Dict,
        current_motion_time: float,
        dt: float = 0.02,
        future_max_steps: int = 95,
        future_num_steps: int = 20,
        anchor_index: int = 0,
        key_body_id: list = [4, 6, 10, 12, 19, 23, 24, 25, 26],
        num_dofs: int = 23,
        num_bodies: int = 27,
        device: str = 'cpu'
    ):
        """
        从pkl文件同时获取 future_motion_targets 和 next_step_ref_motion 的最小实现
        
        Args:
            motion_data: 从 load_pkl_motion 加载的数据
            current_motion_time: 当前动作时间
            dt: 时间步长
            future_max_steps: 未来最大步数
            future_num_steps: 未来采样步数（通常是20）
            anchor_index: anchor body的索引（通常是0，即root）
            key_body_id: 关键body的索引列表
            num_dofs: 关节数量（通常是23）
            num_bodies: 总body数量（通常是27）
            device: 设备 ('cpu' 或 'cuda')
        
        Returns:
            future_motion_targets: torch.Tensor, shape [1, 600]
                包含: root_height(20) + roll_pitch(40) + base_lin_vel(60) + 
                    base_yaw_vel(20) + dof_pos(460) = 600维
            next_step_ref_motion: torch.Tensor, shape [1, 57]
                包含: root_height(1) + roll_pitch(2) + base_lin_vel(3) + 
                    base_yaw_vel(1) + dof_pos(23) + key_body_pos(27) = 57维
        """
        # 1. 计算未来时间步
        start_time = time.perf_counter()
        tar_obs_steps = np.linspace(1, future_max_steps, future_num_steps, dtype=np.int64)
        obs_motion_times = tar_obs_steps * dt + current_motion_time
        
        interpolated_data = interpolate_motion_at_times(
            motion_data, obs_motion_times, num_dofs, num_bodies
        )
        
        root_pos = interpolated_data["root_pos"]  # [num_steps, 3]
        root_rot = interpolated_data["root_rot"]  # [num_steps, 4]
        root_vel_world = interpolated_data["root_vel_world"]  # [num_steps, 3]
        root_ang_vel_world = interpolated_data["root_ang_vel_world"]  # [num_steps, 3]
        dof_pos = interpolated_data["dof_pos"]  # [num_steps, num_dofs]
        ref_body_pos = interpolated_data["ref_body_pos"]  # [num_steps, num_bodies_actual, 3]
        ref_body_rot = interpolated_data["ref_body_rot"]  # [num_steps, num_bodies_actual, 4]
        num_steps = root_pos.shape[0]
        t1 = time.perf_counter()

        # 3. 计算roll_pitch
        rpy = np.array([get_euler_xyz(q) for q in root_rot])  # [num_steps, 3]
        roll_pitch = rpy[:, :2]  # [num_steps, 2]
        
        # print("root_vel_world:", root_vel_world)
        # print("root_ang_vel_world:", root_ang_vel_world)
        # 4. 将速度转换到局部坐标系
        root_vel_local = np.array([quat_rotate_inverse_np(root_rot[i], root_vel_world[i]) 
                                for i in range(num_steps)])  # [num_steps, 3]
        root_ang_vel_local = np.array([quat_rotate_inverse_np(root_rot[i], root_ang_vel_world[i]) 
                                    for i in range(num_steps)])  # [num_steps, 3]
        t2 = time.perf_counter()
        # 5. 计算关键body的局部位置（相对于anchor）
        local_ref_key_body_pos = compute_local_key_body_positions(
            ref_body_pos, ref_body_rot, anchor_index, key_body_id, num_bodies
        )  # [num_steps, key_bodies*3]
        t3 = time.perf_counter()
        # 6. 组装 future_motion_targets
        future_motion_root_height = root_pos[:, 2:3]  # [num_steps, 1]
        future_motion_roll_pitch = roll_pitch  # [num_steps, 2]
        future_motion_base_lin_vel = root_vel_local  # [num_steps, 3]
        future_motion_base_yaw_vel = root_ang_vel_local[:, 2:3]  # [num_steps, 1]
        future_motion_dof_pos = dof_pos  # [num_steps, num_dofs]
        
        # 展平并拼接
        future_motion_targets = np.concatenate([
            future_motion_base_lin_vel.flatten(),     # 60维
            future_motion_base_yaw_vel.flatten(),     # 20维
            future_motion_dof_pos.flatten(),          # 460维 (20 * 23)
            future_motion_root_height.flatten(),      # 20维
            future_motion_roll_pitch.flatten(),       # 40维
        ], axis=0)
        # print("future_motion_targets:", future_motion_targets)
        # print("future_motion_targets shape:", future_motion_targets.shape)
        # print("future_motion_root_height:", future_motion_root_height)
        # print("future_motion_roll_pitch:", future_motion_roll_pitch)
        # print("future_motion_base_lin_vel:", future_motion_base_lin_vel)
        # print("future_motion_base_yaw_vel:", future_motion_base_yaw_vel)
        # print("future_motion_dof_pos:", future_motion_dof_pos)
        
        # 7. 组装 next_step_ref_motion（只取第一步的数据）
        next_step_root_height = root_pos[0, 2:3]  # [1]
        next_step_roll_pitch = roll_pitch[0, :]  # [2]
        next_step_root_vel = root_vel_local[0, :]  # [3]
        next_step_root_ang_vel_yaw = root_ang_vel_local[0, 2:3]  # [1]
        next_step_dof_pos = dof_pos[0, :]  # [num_dofs]
        next_step_local_key_body_pos = local_ref_key_body_pos[0, :]  # [key_bodies*3]
        
        next_step_ref_motion = np.concatenate([
            next_step_root_height,           # 1维
            next_step_roll_pitch,           # 2维
            next_step_root_vel,             # 3维
            next_step_root_ang_vel_yaw,     # 1维
            next_step_dof_pos,               # 23维
            next_step_local_key_body_pos,    # 27维 (9个body * 3)
        ], axis=0)
        
        # 8. 转换为torch tensor
        future_motion_targets = torch.from_numpy(future_motion_targets).float().reshape(1, -1)
        next_step_ref_motion = torch.from_numpy(next_step_ref_motion).float().reshape(1, -1)
        
        if device == 'cuda':
            future_motion_targets = future_motion_targets.cuda()
            next_step_ref_motion = next_step_ref_motion.cuda()
        print("interpolate time:", t1 - start_time)
        # print("rotation time:", t2 - t1)
        # print("compute local pos time:", t3 - t2)
        return future_motion_targets, next_step_ref_motion

    def get_obs(self):
        # Get the current joint position and velocity
        for i in range(len(self.config.dof29_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.dof29_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.dof29_joint2motor_idx[i]].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        quat_z_last = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float32)  # 转换为 x,y,z,w 格式
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)


        motion_time = self.counter * self.config.control_dt
        robot_frame_anchor = get_robot_frame_anchor_from_pkl(
            self.motion_data,
            motion_time,
            quat_z_last,
            self.fn_ref_to_robot_frame
        )

        anchor_ref_rot = self.get_obs_anchor_ref_rot(
            quat_z_last,
            robot_frame_anchor[1]  # 旋转部分
        )

        dt = 0.02  # 控制时间步长
        num_dofs = 23  # 关节数量
        key_body_id = [4, 6, 10, 12, 19, 23, 24, 25, 26]  # 关键身体部位ID

        # if self.config.imu_type == "torso":
            # h1 and h1_2 imu is on the torso
            # imu data needs to be transformed to the pelvis frame
            # waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            # waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            # quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)

        # Build history tensors
        action_mask = np.array(self.mode_cfg.output_action_mask)
        obs_joint_mask = np.array(self.mode_cfg.obs_joint_mask)
        default_pos = torch.from_numpy(self.default_angles).unsqueeze(0).float()
        action_scale = np.array(self.mode_cfg.action_scale, dtype=np.float32)
        action_mask_idx = np.where(action_mask)[0]
        action_hist_tensor = torch.cat([
            ((self.abs_history["action"][i] - default_pos)/ action_scale)[:, action_mask_idx] 
            for i in range(self.mode_cfg.history_len)], dim=1)
        qj_hist_tensor = torch.cat([
            (self.abs_history["qj"][i] - default_pos)[:, obs_joint_mask] * self.config.dof_pos_scale
            for i in range(self.mode_cfg.history_len)], dim=1)
        dqj_hist_tensor = torch.cat([
            self.abs_history["dqj"][i][:, obs_joint_mask] * self.config.dof_vel_scale 
            for i in range(self.mode_cfg.history_len)], dim=1)
        
        # action_hist_tensor = torch.cat([self.abs_history["action"][i][:, action_mask_idx] for i in range(self.frame_stack-1)], dim=1)
        omega_hist_tensor = torch.cat([self.abs_history["omega"][i] for i in range(self.mode_cfg.history_len)], dim=1)
        # qj_hist_tensor = torch.cat([self.abs_history["qj"][i][:, obs_joint_mask] for i in range(self.frame_stack-1)], dim=1)
        # dqj_hist_tensor = torch.cat([self.abs_history["dqj"][i][:, obs_joint_mask] for i in range(self.frame_stack-1)], dim=1)
        gravity_orientation_hist_tensor = torch.cat([self.abs_history["gravity_orientation"][i] for i in range(self.mode_cfg.history_len)], dim=1)
        # ref_motion_phase_hist_tensor = torch.cat([self.history["ref_motion_phase"][i] for i in range(self.mode_cfg.history_len)], dim=1)
        command_lin_vel_hist_tensor = torch.cat([self.history["command_lin_vel"][i] for i in range(self.mode_cfg.history_len)], dim=1)
        command_ang_vel_hist_tensor = torch.cat([self.history["command_ang_vel"][i] for i in range(self.mode_cfg.history_len)], dim=1)
        command_base_height_hist_tensor = torch.cat([self.history["command_base_height"][i] for i in range(self.mode_cfg.history_len)], dim=1)
        command_stand_hist_tensor = torch.cat([self.history["command_stand"][i] for i in range(self.mode_cfg.history_len)], dim=1)
        ref_upper_body_pose_hist_tensor = torch.cat([self.history["ref_upper_body_pose"][i] for i in range(self.mode_cfg.history_len)], dim=1)
        sin_phase_hist_tensor = torch.cat([self.history["sin_phase"][i] for i in range(self.mode_cfg.history_len)], dim=1)
        cos_phase_hist_tensor = torch.cat([self.history["cos_phase"][i] for i in range(self.mode_cfg.history_len)], dim=1)
        roll_pitch_hist_tensor = torch.cat([self.abs_history["roll_pitch"][i] for i in range(self.mode_cfg.history_len)], dim=1)
    
        # create observation #TODO: adapt to different policy obs scale
        gravity_orientation = get_gravity_orientation(quat)
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        scale_qj = (qj_obs - self.default_angles) * self.config.dof_pos_scale
        scale_dqj = dqj_obs * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale
        roll_pitch = gravity_orientation[0:2]
        #select observed joints
        qj_obs = scale_qj[np.array(self.mode_cfg.obs_joint_mask)]
        dqj_obs = scale_dqj[np.array(self.mode_cfg.obs_joint_mask)]
        num_actions = self.mode_cfg.num_output_actions

        self.obs_buf_dict = {}
        
        if self.current_mode == "mimic" and self.target_mode == "mimic":
            future_motion_targets, next_step_ref_motion = self.get_future_motion_and_next_step_from_pkl(
                self.motion_data,
                motion_time,
                dt=dt,
                num_dofs=num_dofs,
                key_body_id=key_body_id,
                anchor_index=0
            )

            obs_hist = torch.cat([
                action_hist_tensor, #23*4
                omega_hist_tensor,  #3*4
                qj_hist_tensor,     #23*4
                dqj_hist_tensor,    #23*4
                roll_pitch_hist_tensor, #2*4
            ], dim=1)

            ref_motion_phase = (self.counter * self.config.control_dt) / self.motion_len

            #after 95% of the motion, finish mimic
            if ref_motion_phase >= 0.95:
                self.mimic_finish = True

            # put into observation buffer
            curr_obs = np.zeros(137, dtype=np.float32)
            curr_obs[: num_actions] = self.all_action[action_mask]
            curr_obs[num_actions: num_actions + 6] = anchor_ref_rot.numpy()
            curr_obs[num_actions + 6: num_actions + 9] = ang_vel
            curr_obs[num_actions + 9: 2*num_actions + 9] = qj_obs
            curr_obs[2*num_actions + 9: 3*num_actions + 9] = dqj_obs
            curr_obs[3*num_actions + 9: 3*num_actions + 9 + 57] = next_step_ref_motion.squeeze(0).numpy()
            curr_obs[3*num_actions + 9 + 57:3*num_actions + 9 + 57 +2]= roll_pitch
            curr_obs_tensor = torch.from_numpy(curr_obs).unsqueeze(0)

            self.obs_buf = torch.cat([
                curr_obs_tensor[:, :3 * num_actions + 9], 
                obs_hist, 
                curr_obs_tensor[:, 3 * num_actions + 9:]], 
                dim=1
                )
            # print("action_obs:", self.all_action[action_mask])
            # print("ang_vel_obs:", ang_vel)
            # print("qj_obs:", qj_obs)
            # print("dqj_obs:", dqj_obs)
            # print("gravity_orientation:", gravity_orientation)
            # print("ref_motion_phase:", ref_motion_phase)
            # print("obs_hist:", obs_hist)
            # print("obs_hist shape:", obs_hist.shape)
            # print("obs_buf shape:", self.obs_buf.shape)

            self.obs_buf_dict["actor_obs"] = self.obs_buf
            self.obs_buf_dict["future_motion_targets"] = future_motion_targets
            self.obs_buf_dict["prop_history"] = obs_hist

            # self.history["ref_motion_phase"].appendleft(curr_obs_tensor[:, -1].unsqueeze(0))

        elif self.current_mode == "locomotion" or self.target_mode == "locomotion":
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

            ang_vel_command = np.array([[0.0]])
            base_height_command = np.array([[1.0]]) * 2
            lin_vel_command = np.array([[0.0, 0.0]])
            stand_command = np.array([[0]])  # default stand command
            cos_phase = np.array([[1.]])
            # ref_upper_dof_pose = np.array([0.,0.,0.,0.,0.3,0.,1.,0.,0.,0.,0.,-0.3,0.,1.,0.,0.,0.])
            ref_upper_dof_pose = np.array([self.qj[~action_mask]])
            sin_phase = np.array([[0.0]])
            curr_obs = np.zeros(self.mode_cfg.num_obs, dtype=np.float32)
            curr_obs[:num_actions] = self.all_action[action_mask]
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
            
            self.obs_buf_dict["actor_obs"] = self.obs_buf

            self.history["command_ang_vel"].appendleft(curr_obs_tensor[:, num_actions + 3].unsqueeze(0))
            self.history["command_base_height"].appendleft(curr_obs_tensor[:, num_actions + 4].unsqueeze(0))
            self.history["command_lin_vel"].appendleft(curr_obs_tensor[:,num_actions + 5:num_actions + 7])
            self.history["command_stand"].appendleft(curr_obs_tensor[:, num_actions + 7].unsqueeze(0))
            self.history["cos_phase"].appendleft(curr_obs_tensor[:, num_actions + 8].unsqueeze(0))
            self.history["ref_upper_body_pose"].appendleft(curr_obs_tensor[:, num_actions + 12 + 2*29:num_actions + 29 + 2*29])
            self.history["sin_phase"].appendleft(curr_obs_tensor[:, num_actions + 29 + 2*29].unsqueeze(0))
        
        self.abs_history["omega"].appendleft(torch.from_numpy(ang_vel))
        self.abs_history["qj"].appendleft(torch.from_numpy(self.qj.copy()).unsqueeze(0))
        self.abs_history["dqj"].appendleft(torch.from_numpy(self.dqj.copy()).unsqueeze(0))
        self.abs_history["gravity_orientation"].appendleft(torch.from_numpy(gravity_orientation).unsqueeze(0))
        self.abs_history["roll_pitch"].appendleft(torch.from_numpy(roll_pitch).unsqueeze(0))

    def run(self):
        start_time = time.perf_counter()
        self.counter += 1

        # Check mode switch
        self.check_mode_switch()
        # update obs
        t1 = time.perf_counter()
        self.get_obs()
        t2 = time.perf_counter()

        # Get policy's infered action
        input_names = [input.name for input in self.policy.get_inputs()]
        output_name = self.policy.get_outputs()[0].name

        # 构建输入字典（根据你的观测数据）
        inputs_dict = {}

        # 输入1: actor_obs
        if len(input_names) > 0:
            inputs_dict[input_names[0]] = self.obs_buf_dict["actor_obs"].numpy().astype(np.float32)  # [1, 877]

        # 输入2: future_motion_targets
        if len(input_names) > 1 and "future_motion_targets" in self.obs_buf_dict:
            inputs_dict[input_names[1]] = self.obs_buf_dict["future_motion_targets"].numpy().astype(np.float32)  # [1, 600]

        # 输入3: prop_history
        if len(input_names) > 2 and "prop_history" in self.obs_buf_dict:
            inputs_dict[input_names[2]] = self.obs_buf_dict["prop_history"].numpy().astype(np.float32)  # [1, 740]

        # 运行推理
        outputs = self.policy.run([output_name], inputs_dict)
        t3 = time.perf_counter()
        action = outputs[0].squeeze()

        # Get interpolated actions during mode switch
        self.all_action = np.zeros(len(self.config.dof29_joint2motor_idx), dtype=np.float32)
        mask = np.array(self.mode_cfg.output_action_mask)
        self.all_action[mask] = action
        if self.is_transitioning:
            interpolated_actions = self.get_body_interpolation()
            self.all_action[~mask] = interpolated_actions[~mask]

        if self.current_mode=="mimic" and self.counter < 10:
            self.old_action = self.old_action * 0.9 + self.all_action * 0.1
            self.all_action = self.old_action

        action_scale = np.array(self.mode_cfg.action_scale, dtype=np.float32)
        target_dof_pos = self.default_angles + self.all_action * action_scale 
        self.abs_history["action"].appendleft(torch.from_numpy(target_dof_pos.copy()).unsqueeze(0))

        # Build low cmd
        self.build_cmd(target_dof_pos,self.mode_cfg.kps,self.mode_cfg.kds) #TODO: adapt to different policy

        # send the command
        self.send_cmd(self.low_cmd)

        end_time = time.perf_counter()
        # print("get_obs time percentage:", (t2 - t1) / (end_time - start_time))
        # print("policy time percentage:", (t3 - t2) / (end_time - start_time))
        sleep_duration = max(0, self.config.control_dt - (end_time - start_time))
        time.sleep(sleep_duration)

        # end_log_time = time.perf_counter()
        # print("run frequency: {:.2f} Hz".format(1.0 / (end_log_time - start_time)))

@hydra.main(config_path="configs", config_name="g1_29_actor", version_base="1.1")
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