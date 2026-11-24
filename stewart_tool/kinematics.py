# -*- coding: utf-8 -*-
"""六自由度 Stewart 平台运动学核心。

坐标约定：世界系原点在基座中心、Z 轴向上；上平台本地系原点在平台中心。
零位姿态：上平台中心 (0, 0, H0)，旋转矩阵为单位阵。
姿态输出顺序：[X, Y, Z, Roll, Pitch, Yaw]，长度 mm、角度 deg。
姿态角采用 ZYX 内旋：R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。

算法参考（见 research/ 目录）：
- AsioOtus055/StewartPlatformClass_py : 闭环向量 IK + 阻尼伪逆迭代 FK
- ravichand13/Stewart-platform        : 旋转矩阵与长度残差构造
- matthewmhong/stewart-platform       : 6-RSS 闭式约束与严谨分支处理
- martyushev/fkSGP                    : 一般 6-6 SGP 正解完备性理论
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi


# ---------- 机构结构类型 ----------
# S = 球铰(Spherical, 3 自由度), U = 虎克铰(Universal, 2 自由度),
# P = 移动副(Prismatic, 伸缩缸), R = 转动副(Revolute, 1 自由度)
# 长度方程对所有类型相同: L = ||上铰点世界坐标 - 下铰点世界坐标||
# 差异在于铰点约束与工作空间校验（虎克铰有 allows_cone 摆角限制）。

@dataclass(frozen=True)
class StructureSpec:
    key: str
    label: str
    base_joint: str          # 下平台铰链类型
    platform_joint: str      # 上平台铰链类型
    base_is_spherical: bool  # 下铰点是否可等效为球铰（3R）
    platform_is_spherical: bool
    description: str


STRUCTURES = {
    "SPS": StructureSpec("SPS", "SPS — 两端球铰", "S", "S", True, True,
        "球铰-伸缩缸-球铰。铰点可任意方向摆动，无圆锥摆角限制，标准二力杆模型。"),
    "UPS": StructureSpec("UPS", "UPS — 下虎克/上球铰", "U", "S", False, True,
        "下虎克铰-伸缩缸-上球铰。下铰为 2 自由度虎克铰，需校验下铰摆角。"),
    "SPU": StructureSpec("SPU", "SPU — 下球铰/上虎克", "S", "U", True, False,
        "下球铰(含虎克铰+竖直旋转轴，等效 3R=S)-伸缩缸-上虎克铰。"
        "你的平台即此类型：底部虎克铰带竖直旋转轴可等效球铰，需校验上铰摆角。"),
    "UPU": StructureSpec("UPU", "UPU — 两端虎克铰", "U", "U", False, False,
        "虎克铰-伸缩缸-虎克铰。两端均有圆锥摆角限制，工作空间最小。"),
}


# ---------- 基础工具 ----------

def rot_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """ZYX 欧拉角(deg) -> 旋转矩阵。"""
    return Rot.from_euler("ZYX", [yaw_deg, pitch_deg, roll_deg], degrees=True).as_matrix()


def rot_to_euler(r: np.ndarray) -> tuple[float, float, float]:
    """旋转矩阵 -> (roll, pitch, yaw) 角度 deg。"""
    yaw, pitch, roll = Rot.from_matrix(r).as_euler("ZYX", degrees=True)
    return float(roll), float(pitch), float(yaw)


def quat_from_pose(pose) -> np.ndarray:
    """位姿 [x,y,z,roll,pitch,yaw] -> 四元数 [w,x,y,z]（标量在前，ZYX 约定）。"""
    p = np.asarray(pose, dtype=float).reshape(-1)
    if p.size < 6:
        raise ValueError("位姿需 6 个数值")
    xyzw = Rot.from_euler("ZYX", [p[5], p[4], p[3]], degrees=True).as_quat()  # scipy: [x,y,z,w]
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])  # 统一为 [w,x,y,z]


def pose_from_quat(quat, translation=(0.0, 0.0, 0.0)) -> np.ndarray:
    """四元数 [w,x,y,z]（标量在前）+ 平移 -> 位姿 [x,y,z,roll,pitch,yaw]。"""
    q = np.asarray(quat, dtype=float).reshape(-1)
    if q.size != 4:
        raise ValueError("四元数需 4 个数值")
    xyzw = np.array([q[1], q[2], q[3], q[0]])
    rot = Rot.from_quat(xyzw)
    yaw, pitch, roll = rot.as_euler("ZYX", degrees=True)
    t = np.asarray(translation, dtype=float).reshape(-1)
    if t.size != 3:
        raise ValueError("平移需 3 个数值")
    return np.array([t[0], t[1], t[2], roll, pitch, yaw])


def _as_points(value, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (6, 3):
        raise ValueError(f"{name} 必须是 6x3 数组（6 个铰点的 x,y,z），当前形状 {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 包含无效数值")
    return arr


def _as_six(value, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 6:
        raise ValueError(f"{name} 必须包含 6 个数值，当前为 {arr.size} 个")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 包含无效数值（请检查输入）")
    return arr


# ---------- 平台配置 ----------

@dataclass
class PlatformConfig:
    """平台几何与标定参数。所有长度单位 mm，角度单位 deg。"""

    # ---- 参数化建模（对称 6-6 成对布置）----
    base_radius: float = 500.0          # 下平台铰点分布圆半径
    platform_radius: float = 400.0      # 上平台铰点分布圆半径
    base_pair_angle: float = 30.0       # 下平台同组两铰点夹角（不可与上平台相同，否则雅可比奇异）
    platform_pair_angle: float = 60.0   # 上平台同组两铰点夹角（与下平台错开，保证 6 自由度满秩）
    base_phase: float = 0.0             # 下平台铰点组整体相位
    platform_phase: float = 0.0         # 上平台铰点组整体相位（安装扭转角）
    home_height: float = 600.0          # 零位时上平台中心距基座平面高度
    base_joint_z: float = 0.0           # 下铰点相对基座平面高度
    platform_joint_z: float = 0.0       # 上铰点相对上平台中心的高度
    pair_centers: tuple = (0.0, 120.0, 240.0)  # 三组铰点中心方位角

    # ---- 直接坐标建模（任意 6-6 铰点，优先于参数化）----
    use_direct_points: bool = False
    base_points: Optional[np.ndarray] = None       # (6,3) 世界坐标
    platform_points: Optional[np.ndarray] = None   # (6,3) 上平台本地坐标
    home_translation: Optional[np.ndarray] = None  # (3,) 零位平移

    # ---- 标定 ----
    zero_lengths: Optional[np.ndarray] = None      # (6,) 实测零位缸长；None 则按几何计算
    min_length: float = 0.0                        # 作动器最短（0 表示不校验）
    max_length: float = 0.0                        # 作动器最长（0 表示不校验）

    # ---- 电缸规格（用于限位 / 速度 / 推力计算）----
    stroke: float = 0.0                # 有效行程 (mm)，0 表示未知
    max_speed: float = 0.0             # 额定最大线速度 (mm/s)，0 表示不校验
    max_force: float = 0.0             # 单缸额定推力 (N)，0 表示不校验
    actuator_bore: float = 0.0         # 缸径 (mm)，仅记录
    payload_mass: float = 0.0          # 负载质量 (kg)，用于推力估算
    platform_mass: float = 0.0         # 上平台自重 (kg)
    gravity: float = 9.80665           # 重力加速度 (m/s²)
    # ---- 电缸电机学参数（15 项，见 actuator_specs.py）----
    rated_force: float = 0.0           # 额定推力 N（必要）
    peak_force: float = 0.0            # 最大推力 N
    rated_speed: float = 0.0           # 额定线速度 mm/s（必要）
    peak_speed: float = 0.0            # 最大线速度 mm/s
    max_accel: float = 0.0             # 最大直线加速度 m/s²
    screw_lead: float = 0.0            # 丝杆导程 mm/rev（必要）
    rated_voltage: float = 0.0         # 额定电压 V（必要）
    rated_power: float = 0.0           # 额定输出功率 W（派生）
    rated_torque: float = 0.0          # 额定转矩 N·m（派生，必要）
    rated_current: float = 0.0         # 额定电流 A（派生）
    peak_current: float = 0.0          # 瞬时最大电流 A
    rated_rpm: float = 0.0             # 额定转速 r/min（派生）
    max_rpm: float = 0.0               # 最高转速 r/min
    torque_constant: float = 0.0       # 转矩常数 N·m/A（必要）
    back_emf_constant: float = 0.0     # 线反电势常数 V·s/m（派生）
    debug_mode: bool = False           # 调试模式（由 settings 控制，随档案记录）
    # 位姿包络限位（0 表示不校验）
    max_translation: float = 0.0       # 最大平移幅值 (mm)
    max_tilt: float = 0.0              # 最大倾角 (°，roll/pitch 合成)
    # 零位定义：lowest = 零位在行程最低位；middle = 零位在行程中间
    zero_reference: str = "lowest"
    # 每根缸的身份信息（6 项）：名称 / 编码 / 品牌
    actuator_names: Optional[list] = None
    actuator_codes: Optional[list] = None
    actuator_brands: Optional[list] = None
    # 每根缸独立规格（0 表示沿用全局值）
    leg_strokes: Optional[list] = None
    leg_forces: Optional[list] = None
    leg_speeds: Optional[list] = None

    # ---- 机构结构 ----
    structure: str = "SPU"                         # SPS / UPS / SPU / UPU
    # ---- 虎克铰摆角约束（°）----
    # 虎克铰中轴线相对缸轴的最大允许偏角。0 表示不校验。
    base_joint_limit: float = 0.0                  # 下虎克铰许用摆角
    platform_joint_limit: float = 35.0             # 上虎克铰许用摆角（用户平台为 SPU）
    # 虎克铰中轴方向（本地坐标，单位向量），缸轴相对此方向的夹角即摆角
    base_axis: Optional[np.ndarray] = None
    platform_axis: Optional[np.ndarray] = None

    name: str = "我的平台"

    def __post_init__(self):
        if self.use_direct_points:
            self.base_points = _as_points(self.base_points, "下平台铰点")
            self.platform_points = _as_points(self.platform_points, "上平台铰点")
            t = np.asarray(self.home_translation if self.home_translation is not None
                           else [0.0, 0.0, self.home_height], dtype=float).reshape(-1)
            if t.size != 3 or not np.all(np.isfinite(t)):
                raise ValueError("零位平移必须是 3 个有效数值")
            self.home_translation = t
        if self.zero_lengths is not None:
            self.zero_lengths = _as_six(self.zero_lengths, "零位缸长")

    # ---- 铰点与零位 ----

    def anchors(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 (下铰点世界坐标, 上铰点本地坐标)，形状均为 (6,3)。"""
        if self.use_direct_points:
            return self.base_points.copy(), self.platform_points.copy()

        centers = np.asarray(self.pair_centers, dtype=float)
        if centers.size != 3:
            raise ValueError("铰点组中心方位角必须正好 3 个")
        b, p = [], []
        for c in centers:
            for sign in (1.0, -1.0):
                ang_b = (c + sign * self.base_pair_angle / 2.0 + self.base_phase) * DEG2RAD
                ang_p = (c + sign * self.platform_pair_angle / 2.0 + self.platform_phase) * DEG2RAD
                b.append([self.base_radius * np.cos(ang_b),
                          self.base_radius * np.sin(ang_b),
                          self.base_joint_z])
                p.append([self.platform_radius * np.cos(ang_p),
                          self.platform_radius * np.sin(ang_p),
                          self.platform_joint_z])
        return np.asarray(b), np.asarray(p)

    def origin(self) -> np.ndarray:
        """零位时上平台中心世界坐标。"""
        if self.use_direct_points:
            return self.home_translation.copy()
        return np.array([0.0, 0.0, self.home_height])

    def home_pose(self) -> np.ndarray:
        """零位位姿 [x,y,z,roll,pitch,yaw]。"""
        t = self.origin()
        return np.array([t[0], t[1], t[2], 0.0, 0.0, 0.0])

    def reference_lengths(self) -> np.ndarray:
        """零位基准缸长：优先使用实测标定值。"""
        if self.zero_lengths is not None:
            return self.zero_lengths.copy()
        b, p = self.anchors()
        lens = np.linalg.norm(self.origin() + p - b, axis=1)
        if np.any(lens <= 1e-9):
            raise ValueError("零位缸长计算异常，请检查铰点配置（存在重合点）")
        return lens

    def actuator_limits(self) -> np.ndarray:
        """各缸绝对工作极限 (6,2)：[Lmin_abs, Lmax_abs]。

        行程窗口由 [min_length, max_length] 定义，行程长度 S = max − min。
        零位定义决定零位缸长 L0 在窗口中的位置：
          - lowest：零位在行程起点 → [L0, L0 + S]
          - middle：零位在行程中点 → [L0 − S/2, L0 + S/2]
        即窗口绝对位置 = L0 − 零位在窗口内的偏移，再把 [min,max] 平移过去。
        若上下限都为 0，则退回到电缸管理里配置的 stroke / leg_strokes。
        """
        ref = self.reference_lengths()
        out = np.zeros((6, 2))
        # 行程 S 恒等于 上限−下限；电缸管理的 stroke 只是物理上限校验，不参与计算
        S = self.max_length - self.min_length
        for i in range(6):
            if S > 0:
                # 零位在窗口内的偏移：lowest=起点(min)，middle=中点
                zero_offset = (self.min_length + self.max_length) / 2.0 \
                    if self.zero_reference == "middle" else self.min_length
                lo = ref[i] - zero_offset + self.min_length
                hi = ref[i] - zero_offset + self.max_length
            else:
                # 未配置行程：宽松默认，不误判超限（UI 会提示必填）
                lo, hi = ref[i] * 0.5, ref[i] * 1.5
            out[i] = (lo, hi)
        return out

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in (
            "base_radius", "platform_radius", "base_pair_angle", "platform_pair_angle",
            "base_phase", "platform_phase", "home_height", "base_joint_z",
            "platform_joint_z", "min_length", "max_length", "name",
            "structure", "base_joint_limit", "platform_joint_limit",
            "stroke", "max_speed", "max_force", "actuator_bore",
            "payload_mass", "platform_mass", "gravity",
            "max_translation", "max_tilt", "zero_reference",
            "actuator_names", "actuator_codes", "actuator_brands",
            "leg_strokes", "leg_forces", "leg_speeds",
            "rated_force", "peak_force", "rated_speed", "peak_speed", "max_accel",
            "screw_lead", "rated_voltage", "rated_power", "rated_torque",
            "rated_current", "peak_current", "rated_rpm", "max_rpm",
            "torque_constant", "back_emf_constant", "debug_mode")}
        d["base_axis"] = None if self.base_axis is None else np.asarray(self.base_axis).tolist()
        d["platform_axis"] = None if self.platform_axis is None else np.asarray(self.platform_axis).tolist()
        d["pair_centers"] = list(self.pair_centers)
        d["use_direct_points"] = self.use_direct_points
        d["base_points"] = None if self.base_points is None else self.base_points.tolist()
        d["platform_points"] = None if self.platform_points is None else self.platform_points.tolist()
        d["home_translation"] = None if self.home_translation is None else self.home_translation.tolist()
        d["zero_lengths"] = None if self.zero_lengths is None else self.zero_lengths.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PlatformConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def effective_speed_cap(cfg) -> float:
    """超速校核的统一口径：优先电缸管理『最大线速度』peak_speed，
    缺省回退额定线速度 rated_speed，再回退旧别名 max_speed。
    返回 0 表示不校核。额定语义的派生字段（额定转速/功率等）不受本函数影响。"""
    for attr in ("peak_speed", "rated_speed", "max_speed"):
        try:
            v = float(getattr(cfg, attr, 0) or 0)
        except (TypeError, ValueError):
            v = 0.0
        if v > 0:
            return v
    return 0.0


# ---------- 求解结果 ----------

@dataclass
class SolveResult:
    pose: np.ndarray              # [x,y,z,roll,pitch,yaw] 绝对量
    lengths: np.ndarray           # 由位姿反算的 6 缸长度
    residuals: np.ndarray         # 计算长度 - 输入长度
    converged: bool
    rms_error: float
    max_error: float
    home: np.ndarray = None       # 零位位姿，用于计算相对变化量
    iterations: int = 0
    message: str = ""
    rotation: Optional[np.ndarray] = None
    anchors_world: Optional[np.ndarray] = None
    joint_angles: Optional[np.ndarray] = None   # 6 缸的虎克铰实际摆角(°)，两端取较大值
    joint_violations: Optional[np.ndarray] = None  # 超限布尔数组
    joint_warning: str = ""                      # 约束汇总说明
    # ---- 动力学 / 限位扩展 ----
    velocity: Optional[np.ndarray] = None        # 广义速度 [vx,vy,vz(mm/s),wx,wy,wz(°/s)]
    acceleration: Optional[np.ndarray] = None    # 广义加速度 [mm/s², °/s²]
    leg_velocity: Optional[np.ndarray] = None    # 6 缸线速度 (mm/s)
    leg_force: Optional[np.ndarray] = None       # 6 缸推力 (N)
    total_force: float = 0.0                     # 整体合力 (N，6 缸推力的矢量合成近似)
    speed_violation: bool = False
    force_violation: bool = False
    limit_warning: str = ""                      # 限位汇总
    dynamic_warning: str = ""                    # 速度/推力汇总

    @property
    def delta_pose(self) -> np.ndarray:
        """相对零位的变化量 [Δx,Δy,Δz,Δroll,Δpitch,Δyaw]。"""
        return self.pose - (self.home if self.home is not None else np.zeros(6))

    @property
    def quaternion(self) -> np.ndarray:
        """当前姿态的四元数 [w,x,y,z]（标量在前，ZYX 约定）。"""
        return quat_from_pose(self.pose)

    def summary(self) -> str:
        p = self.pose
        return (f"X={p[0]:+.4f} mm  Y={p[1]:+.4f} mm  Z={p[2]:+.4f} mm\n"
                f"Roll={p[3]:+.5f}°  Pitch={p[4]:+.5f}°  Yaw={p[5]:+.5f}°")


# ---------- 运动学求解器 ----------

class StewartKinematics:
    """给定平台配置，执行正解 / 逆解 / 雅可比分析。"""

    def __init__(self, config: PlatformConfig):
        self.cfg = config
        self.base, self.platform = config.anchors()
        self.origin = config.origin()
        self.ref_lengths = config.reference_lengths()
        self._last_pose: Optional[np.ndarray] = None

    def safe_payloads(self, pose=None, accel: float = 1.0) -> np.ndarray:
        """各动作方向的安全载荷 (6,)，单位 kg。

        定义：在指定动作方向单独施加单位加速度（平移 1 m/s²，转动 1 rad/s²），
        负载质量 m 产生的缸力 F = pinv(Jᵀ)·wrench 与 m 成正比。
        安全载荷 = 额定推力 / (单位质量在该方向产生的最大缸力)。
        取 6 个动作中使任意一根缸最先达到额定的质量上限。
        """
        f_rated = self.cfg.max_force
        if f_rated <= 0:
            return np.zeros(6)
        q = self.home_pose() if pose is None else np.asarray(pose, dtype=float)
        j = self.jacobian(q)
        jt_pinv = np.linalg.pinv(j.T)          # 6x6, F = jt_pinv @ wrench
        # 雅可比平移列单位 mm，转动列单位 mm/rad，故 wrench 力用 N、力矩用 N·mm 保持一致
        r_mm = float(np.linalg.norm(self.platform[:, :2], axis=1).mean())
        g = self.cfg.gravity
        out = np.zeros(6)
        for dof in range(6):
            w = np.zeros(6)
            if dof < 3:
                # 平移：负载在该方向承受 1g 等效加速度（N/kg = g）
                w[dof] = g * accel
            else:
                # 转动：负载位于平台边缘 R(mm)，重力倾覆力矩 m·g·R (N·mm/kg)
                w[dof] = g * r_mm * accel
            forces_per_kg = jt_pinv @ w
            denom = float(np.max(np.abs(forces_per_kg)))
            out[dof] = f_rated / denom if denom > 1e-12 else 0.0
        return out

    # ---- 正/逆运动学 ----

    def leg_lengths(self, pose: np.ndarray) -> np.ndarray:
        """逆解：位姿 -> 6 缸长度（闭式）。"""
        q = np.asarray(pose, dtype=float).reshape(-1)
        if q.size != 6 or not np.all(np.isfinite(q)):
            raise ValueError("位姿必须是 6 个有效数值 [x,y,z,roll,pitch,yaw]")
        r = rot_matrix(q[3], q[4], q[5])
        tips = self.platform @ r.T + q[:3]          # 上铰点世界坐标
        vec = tips - self.base
        lengths = np.linalg.norm(vec, axis=1)
        if np.any(lengths <= 1e-9):
            raise ValueError("作动器长度为零，位姿无效")
        return lengths

    def _residuals(self, q6: np.ndarray, target: np.ndarray) -> np.ndarray:
        """FK 残差：对角度使用弧度（由求解变量进来）。"""
        r = rot_matrix(q6[3] * RAD2DEG, q6[4] * RAD2DEG, q6[5] * RAD2DEG)
        tips = self.platform @ r.T + q6[:3]
        return np.linalg.norm(tips - self.base, axis=1) - target

    # ---- 虎克铰摆角约束校验 ----

    def joint_angles(self, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (下铰摆角, 上铰摆角, 综合摆角)，单位 °，形状 (6,)。

        摆角 = 缸轴与虎克铰中轴线的夹角。
        中轴线默认竖直：下铰 +Z，上铰 -Z（缸从上铰点指向下铰点）。
        """
        q = np.asarray(pose, dtype=float).reshape(-1)
        r = rot_matrix(q[3], q[4], q[5])
        tips = self.platform @ r.T + q[:3]
        vec = tips - self.base
        lens = np.linalg.norm(vec, axis=1)
        unit = vec / lens[:, None]                     # 下铰点 -> 上铰点

        axis_b = np.asarray(self.cfg.base_axis if self.cfg.base_axis is not None
                            else [0.0, 0.0, 1.0], dtype=float)
        axis_p_local = np.asarray(self.cfg.platform_axis if self.cfg.platform_axis is not None
                                  else [0.0, 0.0, -1.0], dtype=float)
        axis_p = r @ axis_p_local                       # 上铰中轴转到世界系

        def angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            cosv = np.clip(np.sum(a * b, axis=1), -1.0, 1.0)
            return np.degrees(np.arccos(cosv))

        ang_b = angle_between(unit, np.tile(axis_b, (6, 1)))
        ang_p = angle_between(-unit, np.tile(axis_p, (6, 1)))  # 上铰看向下方
        return ang_b, ang_p, np.maximum(ang_b, ang_p)

    def joint_constraint(self, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
        """校验虎克铰摆角是否超限。返回 (摆角数组, 超限布尔, 说明)。"""
        spec = STRUCTURES.get(self.cfg.structure, STRUCTURES["SPS"])
        ang_b, ang_p, combined = self.joint_angles(pose)
        base_on = (not spec.base_is_spherical) and self.cfg.base_joint_limit > 0
        plat_on = (not spec.platform_is_spherical) and self.cfg.platform_joint_limit > 0
        # 逐端比较：不能用 max(ang_b,ang_p) 对 min(base_limit,platform_limit)，
        # 否则上下限位不同时会误报（例如下 10°/上 35°，下 8°/上 30° 被错判超限）。
        violated = np.zeros(6, dtype=bool)
        if base_on:
            violated |= ang_b > self.cfg.base_joint_limit + 1e-9
        if plat_on:
            violated |= ang_p > self.cfg.platform_joint_limit + 1e-9
        if not base_on and not plat_on:
            msg = f"结构 {self.cfg.structure}：摆角校验未启用（请设置虎克铰许用摆角）"
        elif np.any(violated):
            idx = [str(i + 1) for i in np.where(violated)[0]]
            parts = []
            if base_on and np.any(ang_b[violated] > self.cfg.base_joint_limit + 1e-9):
                parts.append(f"下铰 {ang_b[violated].max():.1f}° > 许用 {self.cfg.base_joint_limit:.1f}°")
            if plat_on and np.any(ang_p[violated] > self.cfg.platform_joint_limit + 1e-9):
                parts.append(f"上铰 {ang_p[violated].max():.1f}° > 许用 {self.cfg.platform_joint_limit:.1f}°")
            msg = (f"⚠ 结构 {self.cfg.structure}：缸 {', '.join(idx)} 虎克铰摆角超限 "
                   f"({'；'.join(parts)})")
        else:
            parts = []
            if base_on:
                parts.append(f"下铰最大 {ang_b.max():.1f}°/许用 {self.cfg.base_joint_limit:.1f}°")
            if plat_on:
                parts.append(f"上铰最大 {ang_p.max():.1f}°/许用 {self.cfg.platform_joint_limit:.1f}°")
            msg = f"✓ 结构 {self.cfg.structure}：虎克铰摆角正常（{'；'.join(parts)}）"
        return combined, violated, msg

    def forward(self, lengths, initial_pose=None, strategy: str = "home",
                tolerance: float = 1e-6, max_iter: int = 300,
                dt: float = 0.0, prev_pose=None, prev_velocity=None,
                solver: str = "trf") -> SolveResult:
        """正解：6 缸长度 -> 平台位姿。

        strategy: home=从零位开始 / last=从上次解开始 / multi=多起点取最优
        solver: trf=阻尼最小二乘 / lm=Levenberg-Marquardt / auto=两者竞争取最优
        dt/prev_pose/prev_velocity: 用于速度/加速度/推力计算（可选）
        """
        target = _as_six(lengths, "缸长输入")
        # 硬限位改为"记录警告"，不直接抛异常，方便用户看到越界状态
        # 用相对零位行程推导的绝对上下限判断（min/max 是 ΔL，不是绝对缸长）
        limit_msgs = []
        abs_limits = self.cfg.actuator_limits()
        bad_lo = np.where(target < abs_limits[:, 0] - 1e-6)[0]
        bad_hi = np.where(target > abs_limits[:, 1] + 1e-6)[0]
        if bad_lo.size:
            limit_msgs.append(f"缸 {[int(i)+1 for i in bad_lo]} 低于行程下限 "
                              f"(L0{self.cfg.min_length:+.1f}={abs_limits[bad_lo,0].min():.1f}mm)")
        if bad_hi.size:
            limit_msgs.append(f"缸 {[int(i)+1 for i in bad_hi]} 超过行程上限 "
                              f"(L0{self.cfg.max_length:+.1f}={abs_limits[bad_hi,1].max():.1f}mm)")

        starts = []
        if strategy == "last" and self._last_pose is not None:
            starts.append(self._pose_to_rad(self._last_pose))
        if strategy == "multi":
            home = self._pose_to_rad(self.home_pose())
            starts.extend([home,
                           home + np.array([0, 0, 20, 0.03, 0.03, 0.03]),
                           home + np.array([0, 0, -20, -0.03, -0.03, -0.03]),
                           home + np.array([15, -15, 0, 0.05, -0.05, 0.05])])
        if not starts:
            starts.append(self._pose_to_rad(initial_pose if initial_pose is not None
                                            else self.home_pose()))

        best = None
        total_iter = 0
        methods = ("trf", "lm") if solver == "auto" else ((solver,) if solver in ("trf", "lm") else ("trf",))
        for q0 in starts:
            for method in methods:
                # lm 不支持 x_scale 与 trf 相同的语义（lm 仅接受 'jac' 等），且无 bounds
                kwargs = dict(args=(target,), method=method,
                              ftol=tolerance, xtol=tolerance, gtol=tolerance,
                              max_nfev=max_iter)
                if method == "trf":
                    kwargs["x_scale"] = np.array([1.0, 1.0, 1.0, 0.05, 0.05, 0.05])
                try:
                    sol = least_squares(self._residuals, q0, **kwargs)
                except Exception:
                    continue
                total_iter += int(sol.nfev)
                err = float(np.max(np.abs(sol.fun)))
                if best is None or err < best[0]:
                    best = (err, sol)
        if best is None:
            raise ValueError("正解求解失败：所有求解器均未返回结果")

        err, sol = best
        q = sol.x.copy()
        pose = np.array([q[0], q[1], q[2], q[3] * RAD2DEG, q[4] * RAD2DEG, q[5] * RAD2DEG])
        resid = self._residuals(q, target)
        calc = target + resid
        ok = bool(err <= max(tolerance * 100.0, 1e-4))

        r = rot_matrix(pose[3], pose[4], pose[5])
        tips = self.platform @ r.T + pose[:3]
        joint_angles, joint_violations, joint_warning = self.joint_constraint(pose)
        env_violations, env_warning = self.envelope_check(pose)
        dyn = self.kinematics_dynamics(pose, prev_pose, prev_velocity, dt)
        # env_warning 自带 "⚠ 限位：/✓ 限位：" 前缀，统一去掉后重新组装
        env_text = env_warning.split("：", 1)[-1] if "：" in env_warning else env_warning
        all_limit = limit_msgs + ([env_text] if np.any(env_violations) else [])
        limit_text = ("⚠ 限位：" + "；".join(all_limit)) if all_limit else env_warning
        self._last_pose = pose.copy()
        return SolveResult(
            pose=pose, lengths=calc, residuals=resid, converged=ok,
            rms_error=float(np.sqrt(np.mean(resid ** 2))), max_error=err,
            home=self.home_pose(), iterations=total_iter,
            message=(("收敛" if ok else "未收敛：请检查缸长是否物理可达或尝试多起点求解")
                     + f"（求解器 {solver}，迭代 {total_iter} 次）"),
            rotation=r, anchors_world=tips,
            joint_angles=joint_angles, joint_violations=joint_violations,
            joint_warning=joint_warning,
            velocity=dyn["velocity"], acceleration=dyn["acceleration"],
            leg_velocity=dyn["leg_velocity"], leg_force=dyn["leg_force"],
            total_force=dyn["total_force"],
            speed_violation=dyn["speed_bad"], force_violation=dyn["force_bad"],
            limit_warning=limit_text,
            dynamic_warning=dyn["warning"],
        )

    def inverse(self, pose, dt: float = 0.0, prev_pose=None,
                prev_velocity=None) -> SolveResult:
        """逆解：位姿 -> 6 缸长度。"""
        q = np.asarray(pose, dtype=float).reshape(-1)
        if q.size != 6 or not np.all(np.isfinite(q)):
            raise ValueError("位姿必须是 6 个有效数值 [x,y,z,roll,pitch,yaw]")
        lengths = self.leg_lengths(q)
        r = rot_matrix(q[3], q[4], q[5])
        joint_angles, joint_violations, joint_warning = self.joint_constraint(q)
        env_violations, env_warning = self.envelope_check(q)
        dyn = self.kinematics_dynamics(q, prev_pose, prev_velocity, dt)
        return SolveResult(
            pose=q.copy(), lengths=lengths, residuals=np.zeros(6),
            converged=True, rms_error=0.0, max_error=0.0,
            home=self.home_pose(), message="闭式解", rotation=r,
            anchors_world=self.platform @ r.T + q[:3],
            joint_angles=joint_angles, joint_violations=joint_violations,
            joint_warning=joint_warning,
            velocity=dyn["velocity"], acceleration=dyn["acceleration"],
            leg_velocity=dyn["leg_velocity"], leg_force=dyn["leg_force"],
            total_force=dyn["total_force"],
            speed_violation=dyn["speed_bad"], force_violation=dyn["force_bad"],
            limit_warning=env_warning if np.any(env_violations) else "",
            dynamic_warning=dyn["warning"],
        )

    def home_pose(self) -> np.ndarray:
        return self.cfg.home_pose()

    def _pose_to_rad(self, pose: np.ndarray) -> np.ndarray:
        p = np.asarray(pose, dtype=float).reshape(-1)
        return np.array([p[0], p[1], p[2],
                         p[3] * DEG2RAD, p[4] * DEG2RAD, p[5] * DEG2RAD])

    # ---- 雅可比与性能指标 ----

    def jacobian(self, pose=None) -> np.ndarray:
        """速度雅可比 J：缸长速度 = J @ 平台速度 [vx,vy,vz,wx,wy,wz]（角速度 rad/s）。"""
        q = self.home_pose() if pose is None else np.asarray(pose, dtype=float)
        r = rot_matrix(q[3], q[4], q[5])
        tips = self.platform @ r.T + q[:3]
        vec = tips - self.base
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        if np.any(norms < 1e-12):
            raise ValueError("存在零长度缸（上下铰点重合），雅可比不可解")
        unit = vec / norms
        arm = self.platform @ r.T
        j = np.zeros((6, 6))
        for i in range(6):
            j[i, :3] = unit[i]
            j[i, 3:] = np.cross(arm[i], unit[i])
        return j

    def workspace_limits(self) -> np.ndarray:
        """各动作在行程+摆角约束下的可达范围 (6,2) = [最小值, 最大值]（相对零位）。

        关键：不假定零位一定可行（当行程下限 ΔL>0 时零位缸长 L0 会越下限）。
        对每个自由度先粗扫描找到行程窗口内任一可行位姿，再向两侧二分扩展到
        可行区间端点。这样修改行程上下限或摆角都能正确反映到结果。
        """
        limits = self.cfg.actuator_limits()
        # 基准位姿取零位：各动作最大值 = 从零位出发在行程/摆角约束下能走多远。
        # 判定调用统一的 pose_limits_ok，与逆解/正解限位完全同源。
        base_pose = self.home_pose()

        def feasible(dof: int, value: float) -> bool:
            q = base_pose.copy(); q[dof] += value
            return self.pose_limits_ok(q, limits)

        out = np.zeros((6, 2))
        for dof in range(6):
            span = 180.0 if dof >= 3 else 2000.0   # 搜索范围：角度 / 长度
            n_scan = 96
            grid = np.linspace(-span, span, n_scan + 1)
            step = grid[1] - grid[0]
            feasible_grid = [v for v in grid if feasible(dof, v)]
            if not feasible_grid:
                # 网格太粗可能漏掉窄区间：用更细扫描再试一次
                fine = np.linspace(-span, span, 481)
                feasible_grid = [v for v in fine if feasible(dof, v)]
                if not feasible_grid:
                    out[dof] = 0.0
                    continue
                step = fine[1] - fine[0]
            seed = feasible_grid[len(feasible_grid) // 2]   # 取可行区间中部为种子

            # 正向扩展：从 seed 逐步走到不可行，再二分细化边界
            hi = seed
            while hi + step <= span and feasible(dof, hi + step):
                hi += step
            a, b = hi, min(hi + step, span)   # a 可行, b 不可行
            for _ in range(45):
                mid = (a + b) / 2.0
                if feasible(dof, mid):
                    a = mid
                else:
                    b = mid
            v_max = a

            # 负向扩展
            lo = seed
            while lo - step >= -span and feasible(dof, lo - step):
                lo -= step
            a, b = max(lo - step, -span), lo   # a 不可行, b 可行
            for _ in range(45):
                mid = (a + b) / 2.0
                if feasible(dof, mid):
                    b = mid
                else:
                    a = mid
            v_min = b
            # 二分收敛到浮点边界，向内缩极小量确保端点严格可行（与逆解判定一致）
            eps = max(span * 1e-9, 1e-6)
            if not feasible(dof, v_max):
                v_max -= eps * 10
            if not feasible(dof, v_min):
                v_min += eps * 10
            if not feasible(dof, v_max):
                v_max = seed
            if not feasible(dof, v_min):
                v_min = seed
            out[dof] = (v_min, v_max)
        return out

    def workspace_samples(self, axes=(0, 1, 2), grid: int = 12,
                          angles_fixed=None, max_points: int = 4000) -> dict:
        """采样工作空间点云（用于可视化）。

        axes: 要扫描的自由度索引（0..5）；其余自由度固定在零位或 angles_fixed。
        grid: 每个轴采样格数（总点数 = grid^len(axes)，受 max_points 限制）。
        返回 points(N,6) 绝对位姿、offsets(N,6) 相对零位、manip(N,) 可操作性。
        """
        axes = tuple(int(a) for a in axes if 0 <= int(a) < 6)
        if not axes:
            axes = (0, 1, 2)
        limits = self.cfg.actuator_limits()
        base = self.home_pose()
        fixed = base.copy()
        if angles_fixed:
            for d, v in angles_fixed.items():
                if 0 <= int(d) < 6:
                    fixed[int(d)] = float(v)
        # 每个轴的范围取 workspace_limits 的结果（相对零位）
        wsl = self.workspace_limits()
        ranges = []
        for d in axes:
            lo, hi = float(wsl[d, 0]), float(wsl[d, 1])
            if hi <= lo:
                ranges.append(np.array([0.0]))
            else:
                n = max(2, min(grid, 40))
                ranges.append(np.linspace(lo, hi, n))
        total = 1
        for r in ranges:
            total *= len(r)
        if total > max_points:
            # 均匀降采样：减小每轴格数直到满足上限
            scale = (max_points / total) ** (1.0 / len(axes))
            ranges = [np.linspace(r[0], r[-1], max(2, int(len(r) * scale))) for r in ranges]
        meshes = np.meshgrid(*ranges, indexing="ij")
        vals = np.stack([m.ravel() for m in meshes], axis=1)  # (N, len(axes))
        points = np.tile(fixed, (len(vals), 1))
        for j, d in enumerate(axes):
            points[:, d] += vals[:, j]
        ok = np.array([self.pose_limits_ok(q, limits) for q in points], dtype=bool)
        points = points[ok]
        offsets = points - base
        manip = np.array([self.normalized_svd(q)["manipulability"] for q in points]) if len(points) else np.zeros(0)
        return {"points": points, "offsets": offsets, "manip": manip, "axes": axes}

    def actuator_from_platform_velocity(self, pose, platform_velocity) -> np.ndarray:
        return self.jacobian(pose) @ np.asarray(platform_velocity, dtype=float)

    def platform_from_actuator_velocity(self, pose, actuator_velocity) -> np.ndarray:
        return np.linalg.pinv(self.jacobian(pose)) @ np.asarray(actuator_velocity, dtype=float)

    def actuator_from_platform_force(self, pose, platform_force) -> np.ndarray:
        return np.linalg.pinv(self.jacobian(pose).T) @ np.asarray(platform_force, dtype=float)

    # ---- 限位保护 ----

    def pose_limits_ok(self, pose: np.ndarray, limits: np.ndarray = None) -> bool:
        """统一限位判定：缸长行程 + 虎克铰摆角。

        workspace_limits 与逆解/正解的限位校验都调用本方法，保证
        『各动作最大值』与『某位姿是否可行』使用完全相同的规则。
        """
        q = np.asarray(pose, dtype=float).reshape(-1)
        if limits is None:
            limits = self.cfg.actuator_limits()
        # 行程校验仅在配置了 S=上限−下限>0 时启用；stroke 不参与计算
        has_limits = (self.cfg.max_length - self.cfg.min_length) > 0
        if has_limits:
            lens = self.leg_lengths(q)
            if np.any(lens < limits[:, 0] - 1e-6) or np.any(lens > limits[:, 1] + 1e-6):
                return False
        ang_b, ang_p, _ = self.joint_angles(q)
        spec = STRUCTURES.get(self.cfg.structure, STRUCTURES["SPS"])
        if not spec.base_is_spherical and self.cfg.base_joint_limit > 0:
            if np.any(ang_b > self.cfg.base_joint_limit + 1e-6):
                return False
        if not spec.platform_is_spherical and self.cfg.platform_joint_limit > 0:
            if np.any(ang_p > self.cfg.platform_joint_limit + 1e-6):
                return False
        return True

    def envelope_check(self, pose: np.ndarray) -> tuple[np.ndarray, str]:
        """位姿包络校验：缸长上下限 + 平移/倾角包络。返回 (超限布尔, 说明)。"""
        q = np.asarray(pose, dtype=float).reshape(-1)
        violations, msgs = [], []
        lens = self.leg_lengths(q)

        # 缸长限位：仅当配置了 S=上限−下限>0 时校验；stroke 为物理上限校验不参与
        has_limits = (self.cfg.max_length - self.cfg.min_length) > 0
        if has_limits:
            limits = self.cfg.actuator_limits()
            bad_lo = lens < limits[:, 0] - 1e-6
            bad_hi = lens > limits[:, 1] + 1e-6
            if np.any(bad_lo):
                violations.append(bad_lo)
                msgs.append(f"缸 {[int(i)+1 for i in np.where(bad_lo)[0]]} 低于行程下限 "
                            f"{limits[bad_lo,0].min():.1f}mm")
            if np.any(bad_hi):
                violations.append(bad_hi)
                msgs.append(f"缸 {[int(i)+1 for i in np.where(bad_hi)[0]]} 超过行程上限 "
                            f"{limits[bad_hi,1].max():.1f}mm")
        delta = q[:3] - self.home_pose()[:3]
        if self.cfg.max_translation > 0:
            tr = float(np.linalg.norm(delta))
            if tr > self.cfg.max_translation + 1e-6:
                violations.append(np.ones(6, bool))
                msgs.append(f"平移 {tr:.1f}mm 超出包络 {self.cfg.max_translation:.1f}mm")
        if self.cfg.max_tilt > 0:
            tilt = float(np.hypot(q[3], q[4]))
            if tilt > self.cfg.max_tilt + 1e-6:
                violations.append(np.ones(6, bool))
                msgs.append(f"倾角 {tilt:.1f}° 超出包络 {self.cfg.max_tilt:.1f}°")
        either = (np.any(violations, axis=0) if violations else np.zeros(6, bool))
        msg = ("⚠ 限位：" + "；".join(msgs)) if msgs else "✓ 限位：全部在允许范围内"
        return either, msg

    # ---- 速度 / 加速度 / 推力 ----

    def kinematics_dynamics(self, pose: np.ndarray,
                            prev_pose: np.ndarray | None = None,
                            prev_velocity: np.ndarray | None = None,
                            dt: float = 0.0) -> dict:
        """由当前/上一帧位姿计算广义速度、加速度、缸速与缸推力。

        pose/prev_pose: [x,y,z,roll,pitch,yaw]，长度 mm、角度 °。
        dt: 采样间隔 (s)。返回 dict（角度速度单位 °/s）。
        """
        q = np.asarray(pose, dtype=float).reshape(-1)
        out = {"velocity": None, "acceleration": None, "leg_velocity": None,
               "leg_force": None, "total_force": 0.0, "warning": "",
               "speed_bad": False, "force_bad": False}
        if dt <= 0 or prev_pose is None:
            return out
        pp = np.asarray(prev_pose, dtype=float).reshape(-1)
        # 线速度 mm/s，角速度 °/s（位姿里角度本就是度）
        d = q - pp
        # 欧拉角跨 ±180° 时（如 +179°→-179° 实为转 2°）差值会变成 -358°，
        # 必须归一化到 [-180,180) 才能避免虚假的巨大角速度。
        d[3:] = (d[3:] + 180.0) % 360.0 - 180.0
        vel = d / dt
        out["velocity"] = vel
        if prev_velocity is not None:
            acc = (vel - np.asarray(prev_velocity, dtype=float)) / dt
        else:
            acc = np.zeros(6)
        out["acceleration"] = acc
        # 缸线速度：J 的角速度列按 rad 计，先把 °/s 换成 rad/s
        j = self.jacobian(q)
        vel_rad = np.concatenate([vel[:3], np.radians(vel[3:])])
        out["leg_velocity"] = j @ vel_rad

        # 推力估算：平台广义力 = [m*a_xy, m*(g+a_z), I*alpha]
        mass = self.cfg.payload_mass + self.cfg.platform_mass
        msgs = []
        if mass > 0:
            acc_rad = np.concatenate([acc[:3] / 1000.0, np.radians(acc[3:])])  # mm/s²→m/s², °→rad
            # 质心近似在上平台中心，转动惯量用薄圆盘 I=(1/2)m r²。
            # 雅可比转动列单位为 mm/rad（arm 为 mm），故 wrench 转动分量须为 N·mm，
            # 与 safe_payloads() 一致；这里用 kg·mm² 避免 N·m 造成的 1000× 低估。
            r_mm = float(np.linalg.norm(self.platform[:, :2], axis=1).mean())
            inertia = 0.5 * mass * r_mm ** 2
            wrench = np.zeros(6)
            wrench[:3] = mass * acc_rad[:3]
            wrench[2] += mass * self.cfg.gravity
            wrench[3:] = inertia * acc_rad[3:]
            # F_leg = pinv(J^T) @ wrench
            forces = np.linalg.pinv(j.T) @ wrench
            out["leg_force"] = forces
            out["total_force"] = float(np.linalg.norm(forces))
            if self.cfg.max_force > 0:
                bad = np.abs(forces) > self.cfg.max_force
                if np.any(bad):
                    out["force_bad"] = True
                    msgs.append(f"缸 {[int(i)+1 for i in np.where(bad)[0]]} 推力超额定 {self.cfg.max_force:.0f}N")
        speed_cap = effective_speed_cap(self.cfg)
        if speed_cap > 0 and out["leg_velocity"] is not None:
            bad = np.abs(out["leg_velocity"]) > speed_cap
            if np.any(bad):
                out["speed_bad"] = True
                msgs.append(f"缸 {[int(i)+1 for i in np.where(bad)[0]]} 速度超过最大线速度 {speed_cap:.1f}mm/s")
        out["warning"] = ("⚠ " + "；".join(msgs)) if msgs else "✓ 速度/推力在额定范围内"
        return out

    def normalized_svd(self, pose=None) -> dict:
        """归一化雅可比的 SVD 性能诊断。

        J 的列量纲混杂（前 3 列无量纲、后 3 列 mm/rad），直接 cond(J) 会随
        长度单位（mm/m）变化。取特征长度 L（上铰平均半径，mm）把转动列归一化，
        得到无量纲的 Jn，再取 SVD。

        返回: singular_values(6), condition, manipulability, min_singular, isotropy, length_scale
        """
        j = self.jacobian(pose)
        L = float(np.linalg.norm(self.platform[:, :2], axis=1).mean())
        if L < 1e-12:
            L = 1.0
        scale = np.array([1.0, 1.0, 1.0, 1.0 / L, 1.0 / L, 1.0 / L])
        jn = j * scale[None, :]
        s = np.linalg.svd(jn, compute_uv=False)
        s_max, s_min = float(s[0]), float(s[-1])
        return {
            "singular_values": s,
            "condition": s_max / s_min if s_min > 1e-12 else float("inf"),
            "manipulability": float(np.prod(s)),
            "min_singular": s_min,
            "isotropy": s_min / s_max if s_max > 0 else 0.0,
            "length_scale": L,
        }

    def force_ellipsoid(self, pose=None, n: int = 24) -> dict:
        """力传递椭球顶点（世界坐标，相对平台中心）。

        由归一化雅可比 Jn 的 SVD 得到：Jn = U·Σ·Vᵀ。
        沿平台 generalized velocity 单位球映射到缸长速度，其逆映射给出
        "单位缸力能产生的平台速度"椭球；这里用 V 的列 × (1/s) 表示
        等缸力下各方向的响应幅度（归一化到最大值为 1 便于显示）。
        """
        pose = self.home_pose() if pose is None else np.asarray(pose, dtype=float).reshape(-1)
        j = self.jacobian(pose)
        L = float(np.linalg.norm(self.platform[:, :2], axis=1).mean()) or 1.0
        scale = np.array([1.0, 1.0, 1.0, 1.0 / L, 1.0 / L, 1.0 / L])
        jn = j * scale[None, :]
        U, s, Vt = np.linalg.svd(jn, full_matrices=False)
        # 等缸力 → 平台速度椭圆：主轴取 U 的前 3 行（平移分量方向），
        # 半轴 ∝ 1/s（越容易驱动的方向半径越大），归一化后便于显示。
        radii = 1.0 / np.maximum(s[:3], 1e-6)
        radii = radii / float(np.max(radii))
        rot = U[:3, :3]                       # 椭球朝向（3x3 正交阵）
        phi = np.linspace(0, np.pi, max(6, n // 2))
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = []
        for ph in phi:
            for th in theta:
                v = np.array([np.sin(ph) * np.cos(th) * radii[0],
                              np.sin(ph) * np.sin(th) * radii[1],
                              np.cos(ph) * radii[2]])
                pts.append(rot @ v)
        pts = np.array(pts) + pose[:3]
        return {"points": pts, "radii": radii, "rotation": rot, "singular": s}

    def condition_number(self, pose=None) -> float:
        """归一化后的条件数（无量纲，与长度单位无关）。"""
        return float(self.normalized_svd(pose)["condition"])

    def rank(self, pose=None) -> int:
        """雅可比秩：小于 6 表示平台配置奇异，存在不可解算的自由度。"""
        return int(np.linalg.matrix_rank(self.jacobian(pose)))

    def geometry_warning(self) -> str:
        """检查平台配置是否存在结构性奇异。"""
        if self.cfg.use_direct_points:
            return ""
        bp, pp = self.cfg.base_pair_angle, self.cfg.platform_pair_angle
        if abs(bp - pp) < 1e-6:
            return ("警告：上下平台成对夹角相同（均为 %.1f°），雅可比秩亏，平台存在奇异自由度。\n"
                    "建议将上平台夹角改为与下平台不同（例如下 30° / 上 60°），或改用直接坐标模式。" % bp)
        return ""


def default_config() -> PlatformConfig:
    """默认配置：SPU 平台（下虎克铰+竖直旋转轴等效球铰，上虎克铰）。"""
    return PlatformConfig(
        name="示例平台（请按实机修改）",
        base_radius=500.0, platform_radius=400.0,
        base_pair_angle=30.0, platform_pair_angle=60.0,
        base_phase=0.0, platform_phase=0.0,
        home_height=600.0,
        structure="SPU", base_joint_limit=0.0, platform_joint_limit=35.0,
    )
