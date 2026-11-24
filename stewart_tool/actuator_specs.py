# -*- coding: utf-8 -*-
"""电缸电机学参数单一真相源：15 字段元数据 + 派生公式 + 边界处理 + 调试前置校验。

设计：
- FIELDS 是字段定义的唯一来源，UI(gfields)/hints/HintDef/持久化白名单都从它派生。
- level=0 根字段（用户填）；level=1/2 派生字段（公式自动算），按 level 升序遍历即拓扑序，无环。
- necessary=True 的 6 项是调试模式前置校验的必要条件。
- FORMULAS 全为纯函数：输入 dict[str,float]，输出 (value|None, warning)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FieldSpec:
    key: str            # snake_case，同时是 PlatformConfig 字段名
    label: str          # 中文标签
    unit: str
    level: int          # 0=根输入，1/2=派生层级
    necessary: bool     # 调试模式前置必要条件
    derived: bool       # 是否自动回填
    group: str          # mechanical / electrical / derived
    formula_text: str   # 问号里显示的公式说明（派生项有，根项为填表说明）


EFFICIENCY = 0.9        # 丝杠传动效率 η

FIELDS: list[FieldSpec] = [
    FieldSpec("rated_force", "额定推力", "N", 0, True, False, "mechanical",
              "电缸持续输出的最大直线推力，取自铭牌/规格书。"),
    FieldSpec("peak_force", "最大推力", "N", 0, False, False, "mechanical",
              "短时峰值推力，用于瞬时过载校核。"),
    FieldSpec("rated_speed", "额定线速度", "mm/s", 0, True, False, "mechanical",
              "电缸持续运行的最大直线速度。"),
    FieldSpec("peak_speed", "最大线速度", "mm/s", 0, False, False, "mechanical",
              "短时峰值线速度，用于超速告警。"),
    FieldSpec("max_accel", "最大直线加速度", "m/s²", 0, False, False, "mechanical",
              "允许的最大直线加速度，用于惯性力校核。"),
    FieldSpec("screw_lead", "丝杆导程", "mm/rev", 0, True, False, "mechanical",
              "丝杆旋转一圈的直线位移，是圈数与转速换算的核心参数。"),
    FieldSpec("rated_voltage", "额定电压", "V", 0, True, False, "electrical",
              "驱动器/电机的额定母线电压。"),
    FieldSpec("rated_power", "额定输出功率", "W", 2, False, True, "derived",
              "P = F·v/1000 = 额定推力(N)×额定线速度(mm/s)/1000（机械功率 W）"),
    FieldSpec("rated_torque", "额定转矩", "N·m", 1, True, True, "derived",
              "T = F·L/(2π·η·1000) = 额定推力(N)×导程(mm/rev)/(2π×η×1000)，η=0.9"),
    FieldSpec("rated_current", "额定电流", "A", 2, False, True, "derived",
              "I = T/Kt = 额定转矩(N·m)/转矩常数(N·m/A)"),
    FieldSpec("peak_current", "瞬时最大电流", "A", 0, False, False, "electrical",
              "短时峰值电流，用于堵转/瞬时过载告警。"),
    FieldSpec("rated_rpm", "额定转速", "r/min", 1, False, True, "derived",
              "n = 60·v/L = 60×额定线速度(mm/s)/导程(mm/rev)"),
    FieldSpec("max_rpm", "最高转速", "r/min", 0, False, False, "electrical",
              "电机允许的最高转速，用于超速告警。"),
    FieldSpec("torque_constant", "转矩常数", "N·m/A", 0, True, False, "electrical",
              "Kt：单位电流产生的转矩，取自电机规格书。"),
    FieldSpec("back_emf_constant", "线反电势常数", "V·s/m", 1, False, True, "derived",
              "Ke_lin = Kt·2π·1000/L（SI 制 Kt≡Ke，按导程换算为线反电势 V·s/m）"),
]

FIELDS_BY_KEY = {f.key: f for f in FIELDS}
NECESSARY_KEYS = [f.key for f in FIELDS if f.necessary]
DERIVED_FIELDS = sorted([f for f in FIELDS if f.derived], key=lambda x: x.level)

# 调试前置校验的影响说明（非必要项缺失时提示会影响哪些数据）
_IMPACT = {
    "peak_force": "峰值推力校核与瞬时过载提示不可用",
    "peak_speed": "峰值速度告警不可用",
    "max_accel": "加速度上限校核不可用",
    "peak_current": "堵转/瞬时电流告警不可用",
    "max_rpm": "超速告警不可用",
    "rated_power": "功率估算显示为派生值",
    "rated_current": "电流校核不可用",
    "rated_rpm": "额定转速显示不可用",
    "back_emf_constant": "反电势估算不可用",
}


def safe_div(a, b, default=None):
    """除零/空值/非数/负值统一处理，返回 (value|None, warning)。"""
    if a is None or b is None or a == "" or b == "":
        return default, "缺少输入"
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return default, "输入非数值"
    if abs(b) < 1e-12:
        return default, "除数为 0"
    if a < 0 or b < 0:
        return default, "出现负值"
    return a / b, ""


def _num(ctx, key):
    v = ctx.get(key, 0)
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


# 派生公式：ctx -> (value|None, warning)
def _rated_rpm(ctx):
    return safe_div(60.0 * (_num(ctx, "rated_speed") or 0), _num(ctx, "screw_lead"))


def _rated_torque(ctx):
    f, l = _num(ctx, "rated_force"), _num(ctx, "screw_lead")
    if f is None or l is None:
        return None, "缺少推力或导程"
    v, w = safe_div(f * l, 2 * math.pi * EFFICIENCY * 1000.0)
    return v, w


def _rated_current(ctx):
    t, kt = _num(ctx, "rated_torque"), _num(ctx, "torque_constant")
    if t is None or kt is None:
        return None, "缺少转矩或转矩常数"
    return safe_div(t, kt)


def _rated_power(ctx):
    f, v = _num(ctx, "rated_force"), _num(ctx, "rated_speed")
    if f is None or v is None:
        return None, "缺少推力或线速度"
    return f * v / 1000.0, ""


def _back_emf(ctx):
    kt, l = _num(ctx, "torque_constant"), _num(ctx, "screw_lead")
    if kt is None or l is None:
        return None, "缺少转矩常数或导程"
    return safe_div(kt * 2 * math.pi * 1000.0, l)


FORMULAS = {
    "rated_rpm": _rated_rpm,
    "rated_torque": _rated_torque,
    "rated_current": _rated_current,
    "rated_power": _rated_power,
    "back_emf_constant": _back_emf,
}


def recompute(ctx: dict) -> dict[str, tuple]:
    """按 level 升序计算所有派生字段，返回 {key: (value|None, warning)}。"""
    out = {}
    local = dict(ctx)
    for f in DERIVED_FIELDS:
        val, warn = FORMULAS[f.key](local)
        out[f.key] = (val, warn)
        local[f.key] = val       # 下游派生项可用上游结果（如 current 依赖 torque）
    return out


def validate_debug_prerequisites(cfg) -> tuple[bool, list[str], list[str]]:
    """返回 (ok, missing_necessary, warnings_optional)。"""
    def positive(key):
        try:
            return float(getattr(cfg, key, 0) or 0) > 0
        except (TypeError, ValueError):
            return False
    missing = [FIELDS_BY_KEY[k].label for k in NECESSARY_KEYS if not positive(k)]
    warnings = []
    for f in FIELDS:
        if not f.necessary and not positive(f.key):
            impact = _IMPACT.get(f.key, "相关校核不可用")
            warnings.append(f"{f.label} 未填 → {impact}")
    return (not missing), missing, warnings


# ---- 非电机字段（整机规格）：key, 标签, 单位, 类型, hint_key ----
EXTRA_FIELDS = [
    ("actuator_name",  "电缸名称", "",   "str",   "act_name"),
    ("actuator_code",  "电缸编码 / 型号", "", "str", "act_code"),
    ("actuator_brand", "电缸品牌", "",   "str",   "act_brand"),
    ("stroke",         "有效行程", "mm", "float", "act_stroke"),
    ("actuator_bore",  "缸径",     "mm", "float", "act_bore"),
    ("payload_mass",   "负载质量", "kg", "float", "act_payload"),
    ("platform_mass",  "平台自重", "kg", "float", "act_platform_mass"),
]
EXTRA_BY_KEY = {e[0]: e for e in EXTRA_FIELDS}

# 电缸页单卡混排顺序（22 项 = 15 电机 + 7 整机）：按物理量相近排列
UI_ORDER_KEYS = [
    "actuator_name", "actuator_code", "actuator_brand",
    "stroke", "screw_lead", "actuator_bore",
    "rated_force", "peak_force", "rated_torque", "rated_power",
    "rated_speed", "peak_speed", "max_accel", "rated_rpm",
    "rated_voltage", "rated_current", "peak_current", "max_rpm",
    "torque_constant", "back_emf_constant",
    "payload_mass", "platform_mass",
]


def motor_revolutions(delta_len, lead, leg_vel=None):
    """单缸位移(mm)/缸速(mm/s) → (圈数 rev, 转速 rpm|None)。

    delta_len / leg_vel 可为标量或 numpy 数组；lead<=0 或非数值返回 (None, None)。
    rpm = 60·|leg_vel| / lead；leg_vel=None 时 rpm=None。
    """
    try:
        lead = float(lead)
    except (TypeError, ValueError):
        return None, None
    if not math.isfinite(lead) or lead <= 0:
        return None, None
    rev = delta_len / lead
    rpm = (60.0 * abs(leg_vel) / lead) if leg_vel is not None else None
    return rev, rpm


def motor_torque(force_n, lead, efficiency: float = EFFICIENCY):
    """缸推力(N) → 电机轴扭矩(N·m)：T = F·lead/(2π·η·1000)。

    lead 单位 mm，1000 把 mm 转 m；η 传动效率（默认 EFFICIENCY=0.9）。
    force_n 可为标量或可迭代数组（返回同结构）；lead<=0 返回 None。
    """
    try:
        lead = float(lead)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(lead) or lead <= 0:
        return None
    k = lead / (2.0 * math.pi * efficiency * 1000.0)
    try:
        return [float(v) * k for v in force_n]
    except TypeError:
        return float(force_n) * k
