import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from torch import Tensor
from common.math_helper import normalize, copysign
import torch.nn.functional as F
from typing import List, Tuple


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def transform_imu_data(waist_yaw, waist_yaw_omega, imu_quat, imu_omega):
    RzWaist = R.from_euler("z", waist_yaw).as_matrix()
    R_torso = R.from_quat([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]]).as_matrix()
    R_pelvis = np.dot(R_torso, RzWaist.T)
    w = np.dot(RzWaist, imu_omega[0]) - np.array([0, 0, waist_yaw_omega])
    return R.from_matrix(R_pelvis).as_quat()[[3, 0, 1, 2]], w

@torch.jit.script
def quat_mul(a, b, w_last: bool):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    if w_last:
        x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    else:
        w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    if w_last:
        quat = torch.stack([x, y, z, w], dim=-1).view(shape)
    else:
        quat = torch.stack([w, x, y, z], dim=-1).view(shape)

    return quat

@torch.jit.script
def matrix_from_quat(quaternions: torch.Tensor, w_last: bool = False) -> torch.Tensor:
        if w_last:
            i, j, k, r = torch.unbind(quaternions, -1)
        else:
            r, i, j, k = torch.unbind(quaternions, -1)

        two_s = 2.0 / (quaternions * quaternions).sum(-1)

        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return o.reshape(quaternions.shape[:-1] + (3, 3))

@torch.jit.script
def quat_conjugate(a: Tensor, w_last: bool) -> Tensor:
    shape = a.shape
    a = a.reshape(-1, 4)
    if w_last:
        return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)
    else:
        return torch.cat((a[:, 0:1], -a[:, 1:]), dim=-1).view(shape)
    
@torch.jit.script
def quat_inverse(x, w_last):
    # type: (Tensor, bool) -> Tensor
    """
    The inverse of the rotation
    """
    return quat_conjugate(x, w_last=w_last)

@torch.jit.script
def quat_apply(a: Tensor, b: Tensor, w_last: bool) -> Tensor:
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    if w_last:
        xyz = a[:, :3]
        w = a[:, 3:]
    else:
        xyz = a[:, 1:]
        w = a[:, :1]
    t = xyz.cross(b, dim=-1) * 2
    return (b + w * t + xyz.cross(t, dim=-1)).view(shape)

# XYZW
@torch.jit.script
def my_quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

@torch.jit.script
def calc_heading(q):
    # type: (Tensor) -> Tensor
    # calculate heading direction from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    # this is the x axis heading
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 0] = 1
    rot_dir = my_quat_rotate(q, ref_dir)

    heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])
    return heading

@torch.jit.script
def quat_unit(a):
    return normalize(a)

@torch.jit.script
def quat_from_angle_axis(angle: Tensor, axis: Tensor, w_last: bool) -> Tensor:
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    if w_last:
        return quat_unit(torch.cat([xyz, w], dim=-1))
    else:
        return quat_unit(torch.cat([w, xyz], dim=-1))

@torch.jit.script
def calc_heading_quat(q, w_last):
    # type: (Tensor, bool) -> Tensor
    # calculate heading rotation from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(heading, axis, w_last=w_last)
    return heading_q

@torch.jit.script
def get_euler_xyz_in_tensor(q):
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

@torch.jit.script
def quat_rotate_inverse(q: Tensor, v: Tensor, w_last: bool) -> Tensor:
    shape = q.shape
    if w_last:
        q_w = q[:, -1]
        q_vec = q[:, :3]
    else:
        q_w = q[:, 0]
        q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = (
        q_vec
        * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1)
        * 2.0
    )
    return a - b + c

@torch.jit.script
def quat_apply(a: Tensor, b: Tensor, w_last: bool) -> Tensor:
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    if w_last:
        xyz = a[:, :3]
        w = a[:, 3:]
    else:
        xyz = a[:, 1:]
        w = a[:, :1]
    t = xyz.cross(b, dim=-1) * 2
    return (b + w * t + xyz.cross(t, dim=-1)).view(shape)

@torch.jit.script
def slerp(q0, q1, t):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    # 这个code对xyzw和wxyz都适用, 只要 q0和q1的顺序一致
    cos_half_theta = torch.sum(q0 * q1, dim=-1)

    neg_mask = cos_half_theta < 0
    q1 = q1.clone()
    q1[neg_mask] = -q1[neg_mask]
    cos_half_theta = torch.abs(cos_half_theta)
    cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

    half_theta = torch.acos(cos_half_theta)
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

    ratioA = torch.sin((1 - t) * half_theta) / sin_half_theta
    ratioB = torch.sin(t * half_theta) / sin_half_theta

    new_q = ratioA * q0 + ratioB * q1

    new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
    new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)

    return new_q

def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """
    numpy版本的四元数球面线性插值 (xyzw格式)
    
    Args:
        q0: 起始四元数 [4] (xyzw)
        q1: 结束四元数 [4] (xyzw)
        t: 插值系数 [0, 1]
    
    Returns:
        插值后的四元数 [4] (xyzw)
    """
    # 计算点积
    cos_half_theta = np.dot(q0, q1)
    
    # 如果点积为负，取反以保证走最短路径
    if cos_half_theta < 0:
        q1 = -q1
        cos_half_theta = -cos_half_theta
    
    cos_half_theta = np.abs(cos_half_theta)
    cos_half_theta = np.clip(cos_half_theta, -1.0, 1.0)
    
    # 如果两个四元数几乎相同，直接线性插值
    if np.abs(cos_half_theta) >= 1.0:
        return q0
    
    half_theta = np.arccos(cos_half_theta)
    sin_half_theta = np.sqrt(1.0 - cos_half_theta * cos_half_theta)
    
    # 如果sin太小，使用线性插值
    if np.abs(sin_half_theta) < 0.001:
        return (1.0 - t) * q0 + t * q1
    
    ratioA = np.sin((1.0 - t) * half_theta) / sin_half_theta
    ratioB = np.sin(t * half_theta) / sin_half_theta
    
    new_q = ratioA * q0 + ratioB * q1
    return new_q

def get_euler_xyz(q: np.ndarray) -> np.ndarray:
    """
    从四元数提取欧拉角 (xyzw格式)
    
    Args:
        q: 四元数 [4] (xyzw)
    
    Returns:
        欧拉角 [3] (roll, pitch, yaw)
    """
    x, y, z, w = q[0], q[1], q[2], q[3]
    
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = w * w - x * x - y * y + z * z
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2.0, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = w * w + x * x - y * y - z * z
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])

def quat_rotate_inverse_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    numpy版本的逆旋转 (xyzw格式)
    
    Args:
        q: 四元数 [4] (xyzw)
        v: 向量 [3]
    
    Returns:
        旋转后的向量 [3]
    """
    q_w = q[3]
    q_vec = q[:3]
    
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    
    return a - b + c

def quat_apply_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    numpy版本的四元数应用 (xyzw格式)
    
    Args:
        a: 四元数 [4] (xyzw)
        b: 向量 [3]
    
    Returns:
        旋转后的向量 [3]
    """
    xyz = a[:3]
    w = a[3]
    t = np.cross(xyz, b) * 2.0
    return b + w * t + np.cross(xyz, t)

def quat_inverse_np(q: np.ndarray) -> np.ndarray:
    """
    numpy版本的四元数逆 (xyzw格式)
    
    Args:
        q: 四元数 [4] (xyzw)
    
    Returns:
        四元数的逆 [4] (xyzw)
    """
    return np.array([-q[0], -q[1], -q[2], q[3]])

@torch.jit.script
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    
    WXYZ
    
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

@torch.jit.script
def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.
    # wxyz
    
    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (torch.sin(half_angles[~small_angles]) / angles[~small_angles])
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (0.5 - (angles[small_angles] * angles[small_angles]) / 48)
    quaternions = torch.cat([torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1)
    return quaternions

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    w x y z
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(torch.stack(
        [
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ],
        dim=-1,
    ))

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0]**2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1]**2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2]**2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3]**2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
                          ].reshape(batch_dim + (4,))

def wxyz_to_xyzw(quat):
    return quat[..., [1, 2, 3, 0]]

@torch.jit.script
def quat_pos(x):
    """
    make all the real part of the quaternion positive
    """
    q = x
    z = (q[..., 3:] < 0).float()
    q = (1 - 2 * z) * q
    return q

@torch.jit.script
def quat_normalize(q):
    """
    Construct 3D rotation from quaternion (the quaternion needs not to be normalized).
    """
    q = quat_unit(quat_pos(q))  # normalized to positive and unit quaternion
    return q

@torch.jit.script
def quat_identity(shape: List[int]):
    """
    Construct 3D identity rotation given shape
    """
    w = torch.ones(shape + [1])
    xyz = torch.zeros(shape + [3])
    q = torch.cat([xyz, w], dim=-1)
    return quat_normalize(q)

@torch.jit.script
def quat_identity_like(x):
    """
    Construct identity 3D rotation with the same shape
    """
    return quat_identity(x.shape[:-1])

@torch.jit.script
def quat_mul_norm(x, y, w_last):
    # type: (Tensor, Tensor, bool) -> Tensor
    """
    Combine two set of 3D rotations together using \**\* operator. The shape needs to be
    broadcastable
    """
    return quat_unit(quat_mul(x, y, w_last))

@torch.jit.script
def quat_angle_axis(x: Tensor, w_last: bool) -> Tuple[Tensor, Tensor]:
    """
    The (angle, axis) representation of the rotation. The axis is normalized to unit length.
    The angle is guaranteed to be between [0, pi].
    """
    if w_last:
        w = x[..., -1]
        axis = x[..., :3]
    else:
        w = x[..., 0]
        axis = x[..., 1:]
    s = 2 * (w**2) - 1
    angle = s.clamp(-1, 1).arccos()  # just to be safe
    axis /= axis.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-9)
    return angle, axis