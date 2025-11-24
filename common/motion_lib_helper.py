import joblib
import torch
import numpy as np
from typing import Dict, Tuple, List, Optional
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import mujoco
import re
from common.rotation_helper import (
    quat_inverse, quat_mul, quat_apply, calc_heading_quat, slerp,
    quat_slerp,
    quat_identity_like,
    quat_mul_norm,
    quat_angle_axis,
)
import xml.etree.ElementTree as ETree
from collections import OrderedDict
import scipy.ndimage.filters as filters

def get_motion_len(motion_file_path: str):
    """
    Get the length of the motion from a motion file.
    
    Args:
        motion_file_path (str): Path to the motion file.
        
    Returns:
        int: Length of the motion.
    """
    motion_data = joblib.load(motion_file_path)
    motion_data = motion_data[list(motion_data.keys())[0]]
    fps = motion_data["fps"]
    dt = 1.0 / fps
    num_frames = motion_data["root_rot"].shape[0]
    motion_len = dt * (num_frames - 1)
    return motion_len

def load_pkl_motion(pkl_path: str) -> Dict:
    """
    加载 .pkl 文件中的运动数据
    
    Args:
        pkl_path: .pkl 文件路径
    
    Returns:
        运动数据字典
    """
    motion_data = joblib.load(pkl_path)
    # 如果只有一个动作，取第一个
    if isinstance(motion_data, dict) and len(motion_data) == 1:
        motion_data = motion_data[next(iter(motion_data))]
    
    return motion_data
    
def from_mjcf(path):
        # function from Poselib: 
        tree = ETree.parse(path)
        xml_doc_root = tree.getroot()
        xml_world_body = xml_doc_root.find("worldbody")
        if xml_world_body is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")
        # assume this is the root
        xml_body_root = xml_world_body.find("body")
        if xml_body_root is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")
            
        xml_joint_root = xml_body_root.find("joint")
        
        node_names = []
        parent_indices = []
        local_translation = []
        local_rotation = []
        joints_range = []
        body_to_joint = OrderedDict()

        # recursively adding all nodes into the skel_tree
        def _add_xml_node(xml_node, parent_index, node_index):
            node_name = xml_node.attrib.get("name")
            # parse the local translation into float list
            pos = np.fromstring(xml_node.attrib.get("pos", "0 0 0"), dtype=float, sep=" ")
            quat = np.fromstring(xml_node.attrib.get("quat", "1 0 0 0"), dtype=float, sep=" ") # wxyz
            node_names.append(node_name)
            parent_indices.append(parent_index)
            local_translation.append(pos)
            local_rotation.append(quat)
            curr_index = node_index
            node_index += 1
            all_joints = xml_node.findall("joint") # joints need to remove the first 6 joints
            if len(all_joints) == 6:
                all_joints = all_joints[6:]
            
            for joint in all_joints:
                if not joint.attrib.get("range") is None: 
                    joints_range.append(np.fromstring(joint.attrib.get("range"), dtype=float, sep=" "))
                else:
                    if not joint.attrib.get("type") == "free":
                        joints_range.append([-np.pi, np.pi])
            for joint_node in xml_node.findall("joint"):
                body_to_joint[node_name] = joint_node.attrib.get("name")
                
            for next_node in xml_node.findall("body"):
                node_index = _add_xml_node(next_node, curr_index, node_index)

            
            return node_index
        
        _add_xml_node(xml_body_root, -1, 0)
        assert(len(joints_range) == 23) 
        return {
            "node_names": node_names,
            "parent_indices": torch.from_numpy(np.array(parent_indices, dtype=np.int32)),
            "local_translation": torch.from_numpy(np.array(local_translation, dtype=np.float32)),
            "local_rotation": torch.from_numpy(np.array(local_rotation, dtype=np.float32)),
            "joints_range": torch.from_numpy(np.array(joints_range)),
            "body_to_joint": body_to_joint
        }

def forward_kinematics(rotations,root_rotations,root_positions,offsets,parents,local_rotation_mat):
    B, seq_len = rotations.size()[0:2]
    J = rotations.size()[2] + 1
    positions_world = []
    rotations_world = []
    expanded_offsets = (offsets[:, None].expand(B, seq_len, J, 3))

    for i in range(J):
            if parents[i] == -1:
                positions_world.append(root_positions.unsqueeze(0))
                rotations_world.append(root_rotations)

            else:
                jpos = (torch.matmul(rotations_world[parents[i]][:, :, 0], expanded_offsets[:, :, i, :, None]).squeeze(-1) + positions_world[parents[i]])
                rot_mat = torch.matmul(rotations_world[parents[i]], torch.matmul(local_rotation_mat[:,  (i):(i + 1)], rotations[:, :, (i - 1):i, :]))
                
                positions_world.append(jpos)
                rotations_world.append(rot_mat)
    
    positions_world = torch.stack(positions_world, dim=2)
    rotations_world = torch.cat(rotations_world, dim=2)
    return positions_world, rotations_world


def calc_frame_blend(time: torch.Tensor, motion_len: float, num_frames: int, dt: float):
    """
    计算帧索引和插值权重
    
    Args:
        time: 时间点（秒）
        motion_len: 运动总长度（秒）
        num_frames: 总帧数
        dt: 每帧时间间隔（秒）
    
    Returns:
        frame_idx0, frame_idx1, blend
    """
    time = time.clone()
    phase = time / motion_len
    phase = torch.clip(phase, 0.0, 1.0)
    time[time < 0] = 0
    
    frame_idx0 = (phase * (num_frames - 1)).long()
    frame_idx1 = torch.min(frame_idx0 + 1, torch.tensor(num_frames - 1))
    blend = torch.clip((time - frame_idx0 * dt) / dt, 0.0, 1.0)
    
    return frame_idx0, frame_idx1, blend

def get_motion_state_from_pkl(motion_data: Dict, motion_time: float) -> Dict[str, torch.Tensor]:
    """
    从 .pkl 数据中获取指定时刻的运动状态（替代 motion_lib.get_motion_state）
    
    Args:
        motion_data: 从 load_pkl_motion 加载的数据
        motion_time: 时间点（秒）
    
    Returns:
        包含运动状态的字典
    """
    # 获取基本信息
    fps = motion_data['fps']
    dt = 1.0 / fps
    num_frames = motion_data['dof'].shape[0]
    motion_len = num_frames / fps
    
    # 转换为 torch tensor
    motion_times = torch.tensor([motion_time], dtype=torch.float32)
    
    # 计算帧索引和插值权重
    frame_idx0, frame_idx1, blend = calc_frame_blend(
        motion_times, motion_len, num_frames, dt
    )
    
    # 获取前后两帧的数据
    idx0 = frame_idx0[0].item()
    idx1 = frame_idx1[0].item()
    blend_val = blend[0].item()
    
    # 根节点位置和旋转
    root_pos0 = torch.from_numpy(motion_data['root_trans_offset'][idx0]).float()
    root_pos1 = torch.from_numpy(motion_data['root_trans_offset'][idx1]).float()
    root_rot0 = torch.from_numpy(motion_data['root_rot'][idx0]).float()  # xyzw
    root_rot1 = torch.from_numpy(motion_data['root_rot'][idx1]).float()  # xyzw
    
    # 关节位置
    dof_pos0 = torch.from_numpy(motion_data['dof'][idx0]).float()
    dof_pos1 = torch.from_numpy(motion_data['dof'][idx1]).float()
    
    # 插值
    blend_exp = torch.tensor(blend_val).unsqueeze(-1)
    
    # 位置线性插值
    root_pos = (1.0 - blend_exp) * root_pos0 + blend_exp * root_pos1
    
    # 旋转球面插值
    root_rot = slerp(
        root_rot0.unsqueeze(0), 
        root_rot1.unsqueeze(0), 
        blend_exp.unsqueeze(0)
    )[0]
    
    # 关节位置线性插值
    dof_pos = (1.0 - blend_val) * dof_pos0 + blend_val * dof_pos1
    
    # 如果有速度数据
    if 'dof_vel' in motion_data:
        dof_vel0 = torch.from_numpy(motion_data['dof_vel'][idx0]).float()
        dof_vel1 = torch.from_numpy(motion_data['dof_vel'][idx1]).float()
        dof_vel = (1.0 - blend_val) * dof_vel0 + blend_val * dof_vel1
    else:
        dof_vel = torch.zeros_like(dof_pos)
    
    # 如果有根节点速度
    if 'root_lin_vel' in motion_data:
        root_vel0 = torch.from_numpy(motion_data['root_lin_vel'][idx0]).float()
        root_vel1 = torch.from_numpy(motion_data['root_lin_vel'][idx1]).float()
        root_vel = (1.0 - blend_val) * root_vel0 + blend_val * root_vel1
    else:
        root_vel = torch.zeros(3, dtype=torch.float32)
    
    if 'root_ang_vel' in motion_data:
        root_ang_vel0 = torch.from_numpy(motion_data['root_ang_vel'][idx0]).float()
        root_ang_vel1 = torch.from_numpy(motion_data['root_ang_vel'][idx1]).float()
        root_ang_vel = (1.0 - blend_val) * root_ang_vel0 + blend_val * root_ang_vel1
    else:
        root_ang_vel = torch.zeros(3, dtype=torch.float32)
    
    return {
        "root_pos": root_pos.unsqueeze(0),  # [1, 3]
        "root_rot": root_rot.unsqueeze(0),  # [1, 4] (xyzw)
        "dof_pos": dof_pos.unsqueeze(0),    # [1, num_dofs]
        "dof_vel": dof_vel.unsqueeze(0),    # [1, num_dofs]
        "root_vel": root_vel.unsqueeze(0),  # [1, 3]
        "root_ang_vel": root_ang_vel.unsqueeze(0),  # [1, 3]
    }

def setup_init_frame_from_pkl(robot_quat: np.ndarray, motion_data: Dict):
    """
    从 .pkl 数据初始化坐标系转换函数
    
    Args:
        robot_quat: 机器人初始旋转 (numpy array, xyzw)
        motion_data: 从 load_pkl_motion 加载的数据
    
    Returns:
        fn_ref_to_robot_frame: 转换函数
    """
    # 获取初始时刻的参考运动状态
    motion_res_init = get_motion_state_from_pkl(motion_data, 0.0)
    
    # 机器人初始帧
    robot_init_pos = torch.zeros(3, dtype=torch.float32)
    robot_init_rot = calc_heading_quat(
        torch.from_numpy(robot_quat).reshape(1, 4).float(),
        w_last=True
    )[0]
    
    # 参考运动初始帧
    ref_init_pos = motion_res_init["root_pos"][0]
    ref_init_rot = calc_heading_quat(
        motion_res_init["root_rot"],
        w_last=True
    )[0]
    
    # 计算相对旋转
    ref_init_inv = quat_inverse(ref_init_rot, w_last=True)
    q_rel = quat_mul(robot_init_rot, ref_init_inv, w_last=True)

    # 定义转换函数
    def fn_ref_to_robot_frame(ref_frame_anchor: Tuple[torch.Tensor, torch.Tensor]):
        ref_frame_pos, ref_frame_quat = ref_frame_anchor
        
        # 位置转换
        p_rel = quat_apply(ref_init_inv, ref_frame_pos - ref_init_pos, w_last=True)
        p_new = robot_init_pos + quat_apply(robot_init_rot, p_rel, w_last=True)
        
        # 旋转转换
        q_new = quat_mul(q_rel, ref_frame_quat, w_last=True)
        
        return (p_new, q_new)
    
    return fn_ref_to_robot_frame


def get_robot_frame_anchor_from_pkl(
    motion_data: Dict,
    motion_time: float,
    robot_quat: np.ndarray,
    fn_ref_to_robot_frame
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从 .pkl 文件获取 robot_frame_anchor
    
    Args:
        motion_data: 从 load_pkl_motion 加载的数据
        motion_time: 当前时间点（秒）
        robot_quat: 机器人当前旋转（numpy array, xyzw）
        fn_ref_to_robot_frame: 从 setup_init_frame_from_pkl 返回的转换函数
    
    Returns:
        robot_frame_anchor: (位置, 旋转) 元组
    """
    # 获取当前时刻的参考状态
    motion_res = get_motion_state_from_pkl(motion_data, motion_time)
    
    # 提取参考运动的根节点位置和旋转
    ref_root_pos = motion_res["root_pos"][0]  # [3]
    ref_root_rot = motion_res["root_rot"][0]   # [4] (xyzw)
    ref_frame_anchor = (ref_root_pos, ref_root_rot)
    
    # 转换到机器人坐标系
    robot_frame_anchor = fn_ref_to_robot_frame(ref_frame_anchor)
    
    return robot_frame_anchor

def remove_mesh_from_xml(xml_string: str) -> str:
    """
    从XML字符串中移除所有mesh相关的标签，用于前向运动学计算时不需要mesh文件
    
    Args:
        xml_string: 原始XML字符串
    
    Returns:
        清理后的XML字符串（移除了mesh相关标签）
    """
    # 1. 移除meshdir属性（如果存在）
    xml_string = re.sub(r'meshdir="[^"]*"', '', xml_string)
    
    # 2. 移除所有<mesh>标签及其内容（包括多行）
    xml_string = re.sub(r'<mesh[^>]*/>', '', xml_string)  # 自闭合标签
    xml_string = re.sub(r'<mesh[^>]*>.*?</mesh>', '', xml_string, flags=re.DOTALL)  # 有内容的标签
    
    # 3. 移除所有使用mesh属性的geom标签（特别是visual类型的）
    # 匹配包含mesh属性的geom标签（单行或多行）
    xml_string = re.sub(r'<geom[^>]*mesh="[^"]*"[^>]*/>', '', xml_string)  # 自闭合
    xml_string = re.sub(r'<geom[^>]*mesh="[^"]*"[^>]*>.*?</geom>', '', xml_string, flags=re.DOTALL)  # 有内容
    
    # 4. 移除geom标签中的mesh属性（如果还有其他属性）
    xml_string = re.sub(r'\s+mesh="[^"]*"', '', xml_string)
    
    # 5. 清理可能产生的多余空格和空行（但保留必要的格式）
    xml_string = re.sub(r'\n\s*\n+', '\n', xml_string)  # 移除多个连续空行，保留单个换行
    xml_string = re.sub(r'>\s{2,}<', '><', xml_string)  # 只移除标签间的多个空格，保留单个空格
    
    return xml_string

def interpolate_motion_at_times(
    motion_data: Dict,
    motion_times: np.ndarray,
    num_dofs: int = 23,
    num_bodies: int = 27,
) -> Dict[str, np.ndarray]:
    """
    在指定时间点对motion数据进行插值
    
    Args:
        motion_data: 从 load_pkl_motion 加载的数据
        motion_times: 时间点数组 [num_steps]
        num_dofs: 关节数量
        num_bodies: 总body数量
    
    Returns:
        包含插值结果的字典:
            - root_pos: [num_steps, 3]
            - root_rot: [num_steps, 4] (xyzw)
            - root_vel_world: [num_steps, 3]
            - root_ang_vel_world: [num_steps, 3]
            - dof_pos: [num_steps, num_dofs]
            - ref_body_pos: [num_steps, num_bodies_actual, 3]
            - ref_body_rot: [num_steps, num_bodies_actual, 4]
    """
    # 提取motion数据
    dof = motion_data['dof']  # [num_frames, num_dofs]
    root_trans = motion_data['root_trans_offset']  # [num_frames, 3]
    root_rot = motion_data['root_rot']  # [num_frames, 4] (xyzw格式)
    fps = motion_data.get('fps', 30)
    root_vel_global = motion_data['root_vel_global'].squeeze(0)  # [num_frames, 3]
    root_vel_global = root_vel_global.squeeze(1).numpy()
    root_ang_vel_global = motion_data['root_ang_vel_global'].squeeze(0)  # [num_frames, 3]
    root_ang_vel_global = root_ang_vel_global.squeeze(1).numpy()
    # 计算motion长度和帧数
    num_frames = len(dof)
    motion_dt = 1.0 / fps
    motion_length = num_frames * motion_dt
    
    # 获取extended body数据（如果有）
    # 优先检查是否已经有预计算的pose_quat_global（可能在初始化阶段通过FK计算得到）
    has_extended_bodies = 'pose_quat_global' in motion_data
    if has_extended_bodies:
        pose_quat_global = motion_data['pose_quat_global'].numpy().squeeze(0)  # [num_frames, num_bodies, 4]
        # pose_trans_global = motion_data.get('pose_trans_global', 
        #     np.zeros((num_frames, pose_quat_global.shape[1], 3)))
        pose_trans_global = motion_data['pose_trans_global'].numpy().squeeze(0)
        num_bodies_actual = pose_quat_global.shape[1]
        
        # 如果文件中的body数量少于期望的num_bodies，需要扩展
        if num_bodies_actual < num_bodies:
            pose_trans_global_extend = np.zeros((num_frames, num_bodies, 3))
            pose_quat_global_extend = np.zeros((num_frames, num_bodies, 4))
            pose_trans_global_extend[:, :num_bodies_actual] = pose_trans_global
            pose_quat_global_extend[:, :num_bodies_actual] = pose_quat_global
            if num_bodies_actual > 0:
                pose_trans_global_extend[:, num_bodies_actual:] = pose_trans_global[:, -1:, :]
                pose_quat_global_extend[:, num_bodies_actual:] = pose_quat_global[:, -1:, :]
            pose_trans_global = pose_trans_global_extend
            pose_quat_global = pose_quat_global_extend
            num_bodies_actual = num_bodies
        elif num_bodies_actual > num_bodies:
            pose_trans_global = pose_trans_global[:, :num_bodies, :]
            pose_quat_global = pose_quat_global[:, :num_bodies, :]
            num_bodies_actual = num_bodies
    elif 'pose_aa' in motion_data:
        # 如果motion_data中已经有计算好的pose_trans_global和pose_quat_global，直接使用
        if 'pose_trans_global' in motion_data and 'pose_quat_global' in motion_data:
            pose_trans_global = motion_data['pose_trans_global']  # [num_frames, num_bodies, 3]
            pose_quat_global = motion_data['pose_quat_global']  # [num_frames, num_bodies, 4]
            num_bodies_actual = pose_quat_global.shape[1]
            
            # 如果body数量与期望不符，进行调整
            if num_bodies_actual < num_bodies:
                pose_trans_global_extend = np.zeros((num_frames, num_bodies, 3))
                pose_quat_global_extend = np.zeros((num_frames, num_bodies, 4))
                pose_trans_global_extend[:, :num_bodies_actual] = pose_trans_global
                pose_quat_global_extend[:, :num_bodies_actual] = pose_quat_global
                if num_bodies_actual > 0:
                    pose_trans_global_extend[:, num_bodies_actual:] = pose_trans_global[:, -1:, :]
                    pose_quat_global_extend[:, num_bodies_actual:] = pose_quat_global[:, -1:, :]
                pose_trans_global = pose_trans_global_extend
                pose_quat_global = pose_quat_global_extend
                num_bodies_actual = num_bodies
            elif num_bodies_actual > num_bodies:
                pose_trans_global = pose_trans_global[:, :num_bodies, :]
                pose_quat_global = pose_quat_global[:, :num_bodies, :]
                num_bodies_actual = num_bodies
    
    # 对每个时间步进行插值
    root_pos_list = []
    root_rot_list = []
    root_vel_list = []
    root_ang_vel_list = []
    dof_pos_list = []
    ref_body_pos_list = []
    ref_body_rot_list = []
    
    for t in motion_times:
        # 限制时间在motion范围内
        t = np.clip(t, 0, motion_length)
        
        # 计算帧索引和插值系数
        phase = t / motion_length
        phase = np.clip(phase, 0.0, 1.0)
        frame_idx0 = int(phase * (num_frames - 1))
        frame_idx1 = min(frame_idx0 + 1, num_frames - 1)
        
        if frame_idx0 == frame_idx1:
            blend = 0.0
        else:
            blend = np.clip((t - frame_idx0 * motion_dt) / motion_dt, 0.0, 1.0)
        
        # 插值root位置
        root_pos0 = root_trans[frame_idx0]
        root_pos1 = root_trans[frame_idx1]
        root_pos = (1.0 - blend) * root_pos0 + blend * root_pos1
        root_pos_list.append(root_pos)
        
        # 插值root旋转（四元数slerp）
        root_rot0 = root_rot[frame_idx0]
        root_rot1 = root_rot[frame_idx1]
        root_rot_interp = quat_slerp(root_rot0, root_rot1, blend)
        root_rot_list.append(root_rot_interp)
        
        root_vel0 = root_vel_global[frame_idx0]
        root_vel1 = root_vel_global[frame_idx1]
        root_vel = (1.0 - blend) * root_vel0 + blend * root_vel1
        root_vel_list.append(root_vel)
        
        root_ang_vel0 = root_ang_vel_global[frame_idx0]
        root_ang_vel1 = root_ang_vel_global[frame_idx1]
        root_ang_vel = (1.0 - blend) * root_ang_vel0 + blend * root_ang_vel1
        root_ang_vel_list.append(root_ang_vel)
        
        # 插值dof位置
        dof_pos0 = dof[frame_idx0]
        dof_pos1 = dof[frame_idx1]
        dof_pos = (1.0 - blend) * dof_pos0 + blend * dof_pos1
        dof_pos_list.append(dof_pos)
        
        # 插值body位置和旋转
        ref_body_pos0 = pose_trans_global[frame_idx0]  # [num_bodies_actual, 3]
        ref_body_pos1 = pose_trans_global[frame_idx1]  # [num_bodies_actual, 3]
        ref_body_pos = (1.0 - blend) * ref_body_pos0 + blend * ref_body_pos1  # [num_bodies_actual, 3]
        ref_body_pos_list.append(ref_body_pos)
        
        ref_body_rot0 = pose_quat_global[frame_idx0]  # [num_bodies_actual, 4]
        ref_body_rot1 = pose_quat_global[frame_idx1]  # [num_bodies_actual, 4]
        # 对每个body的旋转进行slerp
        ref_body_rot = np.array([quat_slerp(np.array(ref_body_rot0[i]), np.array(ref_body_rot1[i]), blend) 
                            for i in range(len(ref_body_rot0))])  # [num_bodies_actual, 4]
        ref_body_rot_list.append(ref_body_rot)
    
    # 转换为numpy数组
    root_pos = np.array(root_pos_list)  # [num_steps, 3]
    root_rot = np.array(root_rot_list)  # [num_steps, 4]
    root_vel_world = np.array(root_vel_list)  # [num_steps, 3]
    root_ang_vel_world = np.array(root_ang_vel_list)  # [num_steps, 3]
    dof_pos = np.array(dof_pos_list)  # [num_steps, num_dofs]
    ref_body_pos = np.array(ref_body_pos_list)  # [num_steps, num_bodies_actual, 3]
    ref_body_rot = np.array(ref_body_rot_list)  # [num_steps, num_bodies_actual, 4]
    
    return {
        "root_pos": root_pos,
        "root_rot": root_rot,
        "root_vel_world": root_vel_world,
        "root_ang_vel_world": root_ang_vel_world,
        "dof_pos": dof_pos,
        "ref_body_pos": ref_body_pos,
        "ref_body_rot": ref_body_rot,
        "num_bodies_actual": num_bodies_actual
    }

def compute_local_key_body_positions(
    ref_body_pos: np.ndarray,
    ref_body_rot: np.ndarray,
    anchor_index: int,
    key_body_id: List[int],
    num_bodies: int
) -> np.ndarray:
    """
    计算关键body的局部位置（相对于anchor）
    """
    num_steps = ref_body_pos.shape[0]
    num_bodies_actual = ref_body_pos.shape[1]
    
    ref_body_pos_extend = np.zeros((num_steps, num_bodies, 3))
    ref_body_rot_extend = np.zeros((num_steps, num_bodies, 4))
    ref_body_pos_extend[:, :num_bodies_actual] = ref_body_pos
    ref_body_rot_extend[:, :num_bodies_actual] = ref_body_rot

    
    # 转换为torch tensor进行批量计算
    ref_body_pos_torch = torch.from_numpy(ref_body_pos_extend).float()  # [num_steps, num_bodies, 3]
    ref_body_rot_torch = torch.from_numpy(ref_body_rot_extend).float()  # [num_steps, num_bodies, 4]
    
    # 获取anchor的位置和旋转
    anchor_pos = ref_body_pos_torch[:, anchor_index, :]  # [num_steps, 3]
    anchor_rot = ref_body_rot_torch[:, anchor_index, :]  # [num_steps, 4]
    
    # 广播anchor到所有body: [num_steps, 3] -> [num_steps, num_bodies, 3]
    anchor_pos_w_repeat = anchor_pos.unsqueeze(1).expand(-1, num_bodies, -1)  # [num_steps, num_bodies, 3]
    anchor_rot_w_repeat = anchor_rot.unsqueeze(1).expand(-1, num_bodies, -1)  # [num_steps, num_bodies, 4]
    
    # 计算相对位置
    rel_pos = ref_body_pos_torch - anchor_pos_w_repeat  # [num_steps, num_bodies, 3]
    
    # 批量计算局部位置
    from common.rotation_helper import quat_inverse, quat_apply
    anchor_rot_inv = quat_inverse(anchor_rot_w_repeat, w_last=True)  # [num_steps, num_bodies, 4]
    local_pos = quat_apply(anchor_rot_inv, rel_pos, w_last=True)  # [num_steps, num_bodies, 3]
    
    # 选择关键body
    key_body_id_tensor = torch.tensor(key_body_id, dtype=torch.long)
    local_ref_key_body_pos = local_pos[:, key_body_id_tensor, :]  # [num_steps, len(key_body_id), 3]
    local_ref_key_body_pos = local_ref_key_body_pos.reshape(num_steps, -1)  # [num_steps, len(key_body_id)*3]
    
    
    return local_ref_key_body_pos.numpy()

# @staticmethod
def _compute_velocity(p, time_delta, guassian_filter=True):
    velocity = np.gradient(p.numpy(), axis=-3) / time_delta
    if guassian_filter:
        velocity = torch.from_numpy(filters.gaussian_filter1d(velocity, 2, axis=-3, mode="nearest")).to(p)
    else:
        velocity = torch.from_numpy(velocity).to(p)
    
    return velocity

def _compute_angular_velocity(r, time_delta: float, guassian_filter=True):
    # assume the second last dimension is the time axis
    diff_quat_data = quat_identity_like(r).to(r)
    diff_quat_data[..., :-1, :, :] = quat_mul_norm(r[..., 1:, :, :], quat_inverse(r[..., :-1, :, :], w_last=True), w_last=True)
    diff_angle, diff_axis = quat_angle_axis(diff_quat_data, w_last=True)
    angular_velocity = diff_axis * diff_angle.unsqueeze(-1) / time_delta
    if guassian_filter:
        angular_velocity = torch.from_numpy(filters.gaussian_filter1d(angular_velocity.numpy(), 2, axis=-3, mode="nearest"),)
    return angular_velocity  