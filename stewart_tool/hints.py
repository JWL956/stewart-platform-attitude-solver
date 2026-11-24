# -*- coding: utf-8 -*-
"""问号帮助系统 —— 三页签：怎么填(大白话) / 在哪个位置(线框图) / 计算公式。

设计原则：
- 输入项：先用大白话讲清楚"这个数怎么量、填什么"，再给线框图指出部位。
- 输出项：先用大白话讲"这个数什么意思、怎么看"，再给公式。
- 线框图对能画的部位（半径/夹角/铰点/缸/高度/轴）做高亮 + 引线标注；
  无法用图表达的量（容差/策略/限位）明确说明"此项无部位图"，纯文字讲清。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QBrush, QPolygonF
from PySide6.QtWidgets import (
    QToolButton, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget,
    QWidget, QScrollArea, QFrame, QPushButton,
)
from viewport3d import Viewport3D


# ==================== 帮助条目定义 ====================

@dataclass(frozen=True)
class HintDef:
    title: str
    plain: str = ""                 # 大白话：这个数怎么量、填什么 / 这个结果怎么看
    diagram: str | None = None      # 框线图高亮部位标识
    diagram_label: str = ""         # 图上的引线标注短文本
    formula: str = ""               # HTML 公式说明
    note: str = ""                  # 一句话简述
    is_output: bool = False         # 输出项 → 默认打开"怎么填/怎么看"页


_EQ_LEN = """
<div class="eq">L<sub>i</sub> = ‖ t + R·p<sub>i</sub> − b<sub>i</sub> ‖<sub>2</sub>
&nbsp;=&nbsp; √[ (x<sub>i</sub>−x<sub>bi</sub>)² + (y<sub>i</sub>−y<sub>bi</sub>)² + (z<sub>i</sub>−z<sub>bi</sub>)² ]</div>
<p>b<sub>i</sub> 是下铰点世界坐标，p<sub>i</sub> 是上铰点本地坐标，
t=(X,Y,Z)<sup>T</sup> 是上平台中心平移，R=R<sub>z</sub>(yaw)·R<sub>y</sub>(pitch)·R<sub>x</sub>(roll) 是旋转矩阵。</p>
"""

_EQ_ROT = """
<div class="eq">R = R<sub>z</sub>(yaw) · R<sub>y</sub>(pitch) · R<sub>x</sub>(roll)</div>
<p>从旋转矩阵反解欧拉角（ZYX 内旋）：</p>
<div class="eq">pitch = asin(−R<sub>31</sub>)&nbsp;&nbsp;
roll = atan2(R<sub>32</sub>, R<sub>33</sub>)&nbsp;&nbsp;
yaw = atan2(R<sub>21</sub>, R<sub>11</sub>)</div>
"""

_EQ_FK = """
<div class="eq">q* = arg&nbsp;min<sub>q</sub>&nbsp;Σ<sub>i=1</sub><sup>6</sup>
[ ‖ t(q) + R(q)·p<sub>i</sub> − b<sub>i</sub> ‖ − L<sub>i</sub> ]²</div>
<p>给定 6 个缸长反求位姿，没有闭式通解。软件用<opt>阻尼最小二乘</opt>
（SciPy TRF / Levenberg–Marquardt）从零位或上一帧解迭代，
最小化长度残差平方和，从而锁定连续的装配分支。</p>
"""

# 运动学计算栏目公式
_EQ_VEL = """
<p>位姿向量 q = [X, Y, Z, Roll, Pitch, Yaw]，相邻帧中心差分：</p>
<div class="eq">v<sub>k</sub> = ( q<sub>k</sub> − q<sub>k−1</sub> ) / dt&nbsp;&nbsp;[mm/s, °/s]</div>
<p>线速度单位 mm/s，角速度单位 °/s（位姿里的角度本来就是度，直接相减即可）。</p>
"""
_EQ_ACC = """
<p>由两帧速度再差一次得到加速度：</p>
<div class="eq">a<sub>k</sub> = ( v<sub>k</sub> − v<sub>k−1</sub> ) / dt&nbsp;&nbsp;[mm/s², °/s²]</div>
<p>需要三帧位姿 q<sub>k−2</sub>、q<sub>k−1</sub>、q<sub>k</sub>（等间隔 dt）：
先用前两帧算 v<sub>k−1</sub>，再用后两帧算 v<sub>k</sub>，最后相减。</p>
<div class="eq">a<sub>k</sub> = ( q<sub>k</sub> − 2q<sub>k−1</sub> + q<sub>k−2</sub> ) / dt²</div>
"""
_EQ_ANGVEL = """
<p>角速度也可由旋转矩阵反解。ZYX 约定下近似（小角度）：</p>
<div class="eq">ω ≈ [ Roll̇, Pitcḣ, Yaẇ ]&nbsp;&nbsp;[°/s]</div>
<p>精确做法用 Ṙ·Rᵀ 的反对称部分提取角速度矢量；本栏按欧拉角差分给出工程近似值。</p>
"""
_EQ_LEGVEL = """
<p>缸长速度与平台广义速度的雅可比映射（注意角速度要转成 rad/s）：</p>
<div class="eq">L̇ = J · [ v<sub>x</sub>, v<sub>y</sub>, v<sub>z</sub>, ω<sub>x</sub>, ω<sub>y</sub>, ω<sub>z</sub> ]<sup>T</sup></div>
<p>J 的第 i 行为 [ 缸轴单位向量, 上铰点 × 缸轴单位向量 ]，由当前位姿计算。</p>
"""

HINTS: dict[str, HintDef] = {
    # ================= 输入：平台几何参数 =================
    "base_radius": HintDef(
        "下平台铰点圆半径 Rb",
        plain="这里填【不动的底座】上，从底座中心到任意一个下铰点安装孔中心的距离。"
              "拿尺子量：底座圆心 → 任一孔中心，单位毫米。比如底座直径 1 米，就填 500。"
              "6 个孔都在同一个圆上，所以只要填这一个半径。",
        diagram="base_radius", diagram_label="Rb：底座中心 → 下铰点",
        note="6 个下铰点分布圆的半径。"),
    "platform_radius": HintDef(
        "上平台铰点圆半径 Rp",
        plain="这里填【会动的上平台】上，从上平台中心到任意一个上铰点孔中心的距离。"
              "把动平台平放，量圆心到孔中心，单位毫米。通常比底座半径小一些。",
        diagram="platform_radius", diagram_label="Rp：平台中心 → 上铰点",
        note="6 个上铰点分布圆的半径。"),
    "base_pair_angle": HintDef(
        "下平台成对夹角 φb",
        plain="底座的 6 个孔不是均匀隔 60° 排的，而是【两个一组、共三组】。"
              "这里填同一组里两个孔，相对这组中心张开的完整夹角。"
              "例如下 30°：一组的两个孔分别在中线两侧各 15°。",
        diagram="base_pair_angle", diagram_label="φb：同组两下铰点的张角",
        note="同组两个下铰点的夹角（全角，度）。"),
    "platform_pair_angle": HintDef(
        "上平台成对夹角 φp",
        plain="和上面的 φb 同理，但这是动平台上【同组两个孔】的张角。"
              "注意：这个数必须和下平台的 φb 不一样（推荐下 30° / 上 60°），"
              "否则平台会丢失一个自由度、姿态解不出来。软件会自动检测并警告。",
        diagram="platform_pair_angle", diagram_label="φp：同组两上铰点的张角",
        note="同组两个上铰点的夹角，必须与 φb 不同。"),
    "base_phase": HintDef(
        "下平台相位角 βb",
        plain="三组下铰点在底座上整体转了多少度。如果 0° 时第一组正好在 X 轴正方向，"
              "实际安装时把它整体转了 10°，这里就填 10。没有扭转就填 0。",
        diagram="base_phase", diagram_label="βb：三组下铰点整体转角",
        note="下铰点组整体绕 Z 轴的安装角。"),
    "platform_phase": HintDef(
        "上平台相位角 βp",
        plain="动平台相对底座的【安装扭转角】。把平台调到零位后，"
              "如果上铰点相对下铰点错开了一个角度，就填这个角度；完全对齐填 0。",
        diagram="platform_phase", diagram_label="βp：上平台相对下平台的扭转",
        note="上平台相对下平台的安装扭转角。"),
    "home_height": HintDef(
        "零位上平台中心高度 H0",
        plain="平台在【零位】（缸不动、姿态归零的参考位置）时，上平台中心离底座平面的高度。"
              "量底座平面到上平台中心的高度，单位毫米。这个数决定 6 根缸的基准长度。",
        diagram="home_height", diagram_label="H0：底座平面 → 上平台中心",
        note="零位时上平台中心距基座平面的高度。"),
    "base_joint_z": HintDef(
        "下铰点 Z 偏移",
        plain="下铰点安装孔相对【底座平面】的高度差。如果铰点就贴在底座平面上，填 0；"
              "如果有垫块把铰点抬高了 20mm，就填 20（往下凹则填负数）。",
        diagram="base_joint_z", diagram_label="下铰点相对底座平面的高度",
        note="下铰点相对基座平面的高度偏移。"),
    "platform_joint_z": HintDef(
        "上铰点 Z 偏移",
        plain="上铰点相对【上平台中心平面】的高度差。铰点与平台中心同高填 0；"
              "铰点装在平台上方 15mm 就填 15，装在下方填负数。",
        diagram="platform_joint_z", diagram_label="上铰点相对平台中心的高度",
        note="上铰点相对上平台中心的高度偏移。"),
    "pair_centers": HintDef(
        "三组铰点中心方位角",
        plain="三组成对铰点，各自的中线在底座平面上的方向角。"
              "均匀布置就是 0、120、240（度），三组相隔 120°。一般不用改，保持默认即可。",
        diagram="pair_centers", diagram_label="三组中心：0° / 120° / 240°",
        note="三组成对铰点中心的方位角。"),
    "base_joint_limit": HintDef(
        "下节点许用摆角",
        plain="下节点（缸和底座连接的铰）允许缸轴最多歪多少度。"
              "虎克铰常见 ±20°~±40°；如果你的下铰带竖直转轴、等效球铰，就填 0 表示不限制。"
              "填 0 = 这一端不做摆角校验。",
        diagram="base_joint_limit", diagram_label="下节点：缸轴与竖直轴夹角",
        note="下节点最大允许摆角，0 = 不校验。"),
    "platform_joint_limit": HintDef(
        "上节点许用摆角",
        plain="上节点（缸和上平台连接的铰）允许缸轴最多歪多少度。"
              "虎克铰常见 ±20°~±40°，按你的铰点样本参数填；填 0 = 不校验。"
              "运行后如果报超限，把平台幅度减小或换更大摆角的铰。",
        diagram="platform_joint_limit", diagram_label="上节点：缸轴与铰中轴夹角",
        note="上节点最大允许摆角，0 = 不校验。"),
    "stroke": HintDef(
        "行程下限 / 上限（决定电缸行程）",
        plain="这两项定义电缸的整个行程窗口：下限=行程起点，上限=行程终点，"
              "行程长度 S = 上限 − 下限。例：下限0、上限50 → 行程50mm；"
              "下限5、上限45 → 行程40mm。"
              "零位定义只决定零位在窗口中的位置（最低位=零位在起点，中间位=零位在中点 S/2），"
              "不改变行程长度。各项最大值只由 S 与摆角决定。"
              "电缸管理里的『有效行程』是电缸的物理上限，S 不能超过它；"
              "若上下限和电缸行程都为 0，视为未配置（必填）。",
        diagram="stroke", diagram_label="行程窗口：下限 ~ 上限",
        formula='<div class="eq">S = ΔL<sub>max</sub> − ΔL<sub>min</sub></div>'
                '<p>零位=最低位：绝对范围 [L<sub>0</sub>, L<sub>0</sub>+S]；'
                '零位=中间位：[L<sub>0</sub>−S/2, L<sub>0</sub>+S/2]。</p>'
                '<p>约束：S ≤ 电缸有效行程(stroke)。</p>',
        note="行程 S = 上限−下限；受电缸有效行程上限约束。"),
    # ---- 电缸规格专用（此前误用 stroke，已修正）----
    "act_name": HintDef("电缸名称",
        plain="给每根缸起个便于识别的名字（如『左前缸』），纯记录用，不参与计算。",
        diagram="legs", diagram_label="L1~L6 缸编号", note="电缸名称，仅记录。"),
    "act_code": HintDef("电缸编码",
        plain="填这根缸的产品型号或资产编码（如伺服电缸型号），纯记录用，不参与计算。",
        diagram="legs", diagram_label="L1~L6 缸编号", note="电缸编码，仅记录。"),
    "act_brand": HintDef("电缸品牌",
        plain="填这根缸的品牌/厂家名称，纯记录用，不参与计算。",
        diagram="legs", diagram_label="L1~L6 缸编号", note="电缸品牌，仅记录。"),
    "act_stroke": HintDef("有效行程",
        plain="每根电缸能伸缩的总行程（毫米），即最长长度减最短长度。"
              "这是最关键的计算参数：应用参数后软件会据此自动算出每根缸的最大/最小长度。"
              "例：行程 200、零位在最低位，则范围是 L0 ~ L0+200。",
        diagram="legs", diagram_label="缸的伸缩行程 S",
        formula='<div class="eq">最低位基准：L<sub>min</sub>=L<sub>0</sub>, L<sub>max</sub>=L<sub>0</sub>+S</div>'
                '<div class="eq">中间位基准：L<sub>min</sub>=L<sub>0</sub>−S/2, L<sub>max</sub>=L<sub>0</sub>+S/2</div>',
        note="电缸有效行程，用于推导长度上下限。"),
    "act_speed": HintDef("最大线速度",
        plain="电缸短时能跑的最快伸缩速度（毫米/秒），即规格书里的峰值/最大线速度。"
              "解算后若某缸所需速度超过此值会告警。填 0 = 不校验；若没填最大值，"
              "软件会回退用额定线速度校核。",
        diagram="legs", diagram_label="缸速上限",
        formula='<div class="eq">L̇<sub>i</sub> = J<sub>i</sub>·q̇,&nbsp; |L̇<sub>i</sub>| ≤ v<sub>peak</sub></div>',
        note="电缸最大线速度(peak_speed)，0 时回退额定线速度；0 = 不校验。"),
    "act_force": HintDef("额定推力",
        plain="单根电缸额定能输出的最大推力（牛顿 N）。软件根据负载质量和加速度估算所需推力，"
              "超过此值会告警。填 0 = 不校验。",
        diagram="legs", diagram_label="推力上限",
        formula='<div class="eq">F = pinv(J<sup>T</sup>)·w,&nbsp; |F<sub>i</sub>| ≤ F<sub>max</sub></div>',
        note="电缸额定推力，0 = 不校验。"),
    "act_bore": HintDef("缸径",
        plain="电缸缸筒内径（毫米），作为规格记录。若日后要做液压/气压推力换算会用到，当前不直接参与计算。",
        diagram="legs", diagram_label="缸筒直径", note="电缸缸径，仅记录。"),
    "act_payload": HintDef("负载质量",
        plain="上平台要带动的负载总质量（千克 kg），不含平台自重。用于估算各缸所需推力："
              "质量越大、加速度越大，推力越大。填 0 则推力估算为 0。",
        diagram="home_height", diagram_label="平台承载质量",
        formula='<div class="eq">w = m·a + m·g,&nbsp; F = pinv(J<sup>T</sup>)·w</div>',
        note="负载质量，用于推力估算。"),
    "act_platform_mass": HintDef("平台自重",
        plain="上平台自身的结构重量（千克 kg）。与负载质量相加后一起参与推力估算。填 0 表示忽略自重。",
        diagram="platform_radius", diagram_label="上平台自重",
        note="上平台自重，用于推力估算。"),
    "act_zero_ref": HintDef("零位定义",
        plain="决定行程如何在零位两侧分配：\n"
              "『零位=最低位』：零位是缸完全收缩的位置，往上只能伸长一个行程；\n"
              "『零位=中间位』：零位在行程中点，可上可下各半个行程。"
              "选好后点『应用参数』，软件会自动算出每根缸的最大/最小长度。",
        diagram="legs", diagram_label="零位在行程中的位置",
        note="零位是行程最低位还是中间位。"),
    "max_translation": HintDef("最大平移包络",
        plain="允许平台中心相对零位平移的最大距离（毫米）。解算后若平移量超过此值会告警。填 0 = 不校验。",
        diagram="home_height", diagram_label="平移包络半径",
        note="最大平移量，0 = 不校验。"),
    "max_tilt": HintDef("最大倾角包络",
        plain="允许平台倾斜的最大角度（度，roll 与 pitch 合成）。解算后若倾角超过此值会告警。填 0 = 不校验。",
        diagram="platform_pair_angle", diagram_label="倾角包络",
        note="最大倾角，0 = 不校验。"),
    "zero_lengths": HintDef(
        "零位实测缸长 L0",
        plain="平台在零位时，用尺子或编码器实测的 6 根缸孔中心距，按顺序填 6 个数。"
              "这组数比几何计算值更准，能补偿装配误差。没有实测数据就 6 个都填 0，"
              "软件会按几何自动算。",
        diagram="zero_lengths", diagram_label="L1~L6 的零位实测值",
        formula=_EQ_LEN,
        note="实测零位缸长，优先于几何计算值。"),
    "base_points": HintDef(
        "下平台铰点坐标 B1~B6",
        plain="直接坐标模式下用：逐个填 6 个下铰点的 X、Y、Z（世界坐标，毫米）。"
              "以底座中心为原点、Z 向上。第 1~6 行要和 6 根缸一一对应。",
        diagram="base_points", diagram_label="B1~B6：下铰点世界坐标",
        note="下铰点世界坐标 (x,y,z)。"),
    "platform_points": HintDef(
        "上平台铰点坐标 P1~P6",
        plain="直接坐标模式下用：逐个填 6 个上铰点的 X、Y、Z。"
              "注意这是【上平台自己的坐标系】（以上平台中心为原点），不是世界坐标。"
              "第 1~6 行要和对应的下铰点、对应的缸一一对应。",
        diagram="platform_points", diagram_label="P1~P6：上铰点本地坐标",
        note="上铰点在上平台本地系的坐标 (x,y,z)。"),

    # ================= 输入：正解 =================
    "fk_leg_i": HintDef(
        "6 路作动器端部数据",
        plain="这里填你从设备上读回来的 6 个缸的数据，一根缸一行，共 6 行。"
              "选『绝对缸长』就填每根缸的实际孔中心距（比如 620.5mm）；"
              "选『相对零位位移』就填每根缸比零位伸出了多少（比如 +12.3，缩回填 -5.0）。"
              "顺序必须和 B1/P1、B2/P2……对应。",
        diagram="legs", diagram_label="L1~L6：6 根缸的实测长度",
        formula=_EQ_LEN,
        note="每根缸的实测长度或相对零位位移。"),
    "fk_dt": HintDef(
        "采样间隔 dt",
        plain="这一项没有机构图纸部位。它是两帧数据之间的时间间隔（秒）。"
              "填大于 0 的数（如 0.02 表示 50Hz 采样），连续计算两次以上，"
              "软件会用相邻两帧的位姿差算出速度、加速度和缸推力。填 0 则不计算这些动态量。",
        diagram=None,
        formula="<div class=\"eq\">v = (q<sub>k</sub> − q<sub>k−1</sub>) / dt,&nbsp;&nbsp; "
                "a = (v<sub>k</sub> − v<sub>k−1</sub>) / dt</div>"
                "<p>缸速由雅可比映射：L̇ = J·q̇；缸推力 F = pinv(Jᵀ)·wrench。</p>",
        note="相邻两帧的时间间隔，用于动态计算。"),

    # ---- 运动学计算栏目 ----
    "kin_frame": HintDef("位姿帧输入 q",
        plain="这一项没有机构图纸部位。按 X、Y、Z、Roll、Pitch、Yaw 的顺序填一帧位姿。"
              "长度单位毫米、角度单位度。应填绝对位姿（与正解页『绝对位姿』一行一致），"
              "不是相对零位的偏移量。计算速度需要至少两帧，计算加速度需要三帧。"
              "也可在正解页点『填入帧 k / k−1 / k−2』按钮自动带入。",
        diagram=None,
        formula="<div class=\"eq\">q = [ X(mm), Y(mm), Z(mm), Roll(°), Pitch(°), Yaw(°) ]</div>"
                + _EQ_VEL,
        note="一帧位姿，6 个数值。"),
    "kin_dt": HintDef("采样间隔 dt（运动学）",
        plain="相邻两帧之间的时间间隔（秒）。填 0.02 表示采样频率 50Hz。"
              "dt 必须大于 0，否则无法计算速度。三帧需等间隔。",
        diagram=None, formula=_EQ_VEL, note="相邻帧时间间隔 (s)。"),
    "kin_vel": HintDef("线速度 / 角速度输出", is_output=True,
        plain="由相邻两帧位姿差算出：前三个是线速度（mm/s），后三个是角速度（°/s）。"
              "正数表示沿该轴正方向运动。",
        diagram=None, formula=_EQ_VEL + _EQ_ANGVEL,
        note="一阶差分得到的广义速度。"),
    "kin_acc": HintDef("加速度 / 角加速度输出", is_output=True,
        plain="由三帧位姿算出：前三个是线加速度（mm/s²），后三个是角加速度（°/s²）。"
              "只有两帧时无法算加速度，会显示『需三帧』。",
        diagram=None, formula=_EQ_ACC,
        note="二阶差分得到的广义加速度。"),
    "kin_legvel": HintDef("各缸线速度输出", is_output=True,
        plain="由雅可比矩阵把平台速度映射到 6 根缸，得到每根缸的伸缩速度（mm/s）。"
              "正值=伸出，负值=缩回。需要当前帧位姿才能计算雅可比。",
        diagram="legs", diagram_label="L1~L6 缸速",
        formula=_EQ_LEGVEL, note="雅可比映射的缸速。"),

    "fk_tol": HintDef(
        "收敛容差",
        plain="这一项没有对应图纸部位。它是正解迭代的停算精度：数值越小算得越准，"
              "但迭代次数越多。默认 0.000001（百万分之一毫米）已经足够，一般不用改。"
              "如果发现明明收敛却精度不够，可以改到 0.0000001。",
        diagram=None,
        formula=_EQ_FK,
        note="正解迭代的终止阈值。"),
    "fk_strategy": HintDef(
        "初值策略",
        plain="这一项没有对应图纸部位。它决定从哪个姿态开始迭代求解："
              "『从零位开始』适合单次解算；『从上次解开始』适合连续轨迹（帧与帧变化小，最快最稳）；"
              "『多起点全局搜索』在怀疑有分支跳变时用，最慢但最不容易解错。",
        diagram=None,
        formula=_EQ_FK,
        note="迭代起始位姿策略。"),

    # ================= 输入：逆解 =================
    "ik_x": HintDef("目标 X 纵荡",
        plain="想让上平台往世界 X 轴正方向平移多少毫米。往负方向就填负数。",
        diagram="axis_x", diagram_label="ΔX：沿 X 轴平移",
        note="期望上平台沿 X 轴的平移 (mm)。"),
    "ik_y": HintDef("目标 Y 横荡",
        plain="想让上平台往世界 Y 轴正方向平移多少毫米。往负方向填负数。",
        diagram="axis_y", diagram_label="ΔY：沿 Y 轴平移",
        note="期望上平台沿 Y 轴的平移 (mm)。"),
    "ik_z": HintDef("目标 Z 升沉",
        plain="想让上平台往上升或下降多少毫米。往上升填正数（如 +20），下降填负数（如 -20）。"
              "这是相对零位高度的增量，不是绝对高度。",
        diagram="home_height", diagram_label="ΔZ：沿 Z 轴升降",
        note="期望上平台沿 Z 轴的平移 (mm)。"),
    "ik_roll": HintDef("目标 Roll 横滚",
        plain="想让平台绕世界 X 轴转多少度（左右翻滚）。按右手定则，正方向为从 X 轴正向看逆时针。",
        diagram="axis_x", diagram_label="Roll：绕 X 轴转",
        formula=_EQ_ROT, note="绕世界 X 轴的旋转 (°)。"),
    "ik_pitch": HintDef("目标 Pitch 俯仰",
        plain="想让平台绕世界 Y 轴转多少度（前后俯仰）。按右手定则确定正负。",
        diagram="axis_y", diagram_label="Pitch：绕 Y 轴转",
        formula=_EQ_ROT, note="绕世界 Y 轴的旋转 (°)。"),
    "ik_yaw": HintDef("目标 Yaw 偏航",
        plain="想让平台绕世界 Z 轴（竖直轴）转多少度（水平旋转）。正值为逆时针。",
        diagram="axis_z", diagram_label="Yaw：绕 Z 轴转",
        formula=_EQ_ROT, note="绕世界 Z 轴的旋转 (°)。"),

    # ================= 输出：位姿 =================
    "out_x": HintDef("X 纵荡 ΔX", is_output=True,
        plain="解出来的平台中心相对零位，往 X 轴方向移动了多少。正数=往 X 正向走，负数=反向。"
               "单位毫米。旁边的三维图能直接看到这个位移。",
        diagram="axis_x", diagram_label="ΔX：平台中心 X 位移",
        formula="<p>平移向量 t 的第一个分量减去零位分量：</p>"
                '<div class="eq">ΔX = X − X<sub>0</sub> = q<sub>1</sub> − q<sub>0,1</sub></div>' + _EQ_FK,
        note="平台中心沿世界 X 轴相对零位的位移。"),
    "out_y": HintDef("Y 横荡 ΔY", is_output=True,
        plain="解出来的平台中心相对零位，往 Y 轴方向移动了多少。正数=往 Y 正向，负数=反向。",
        diagram="axis_y", diagram_label="ΔY：平台中心 Y 位移",
        formula='<div class="eq">ΔY = Y − Y<sub>0</sub> = q<sub>2</sub> − q<sub>0,2</sub></div>',
        note="平台中心沿世界 Y 轴相对零位的位移。"),
    "out_z": HintDef("Z 升沉 ΔZ", is_output=True,
        plain="解出来的平台中心相对零位，升高（正）或降低（负）了多少毫米。",
        diagram="home_height", diagram_label="ΔZ：平台中心升降",
        formula='<div class="eq">ΔZ = Z − Z<sub>0</sub> = q<sub>3</sub> − q<sub>0,3</sub></div>',
        note="平台中心沿世界 Z 轴相对零位的位移。"),
    "out_roll": HintDef("Roll 横滚", is_output=True,
        plain="解出来的平台绕 X 轴倾斜了多少度。看三维图里平台左右倾斜的样子。",
        diagram="axis_x", diagram_label="Roll：绕 X 轴倾斜",
        formula=_EQ_ROT, note="绕 X 轴的姿态角。"),
    "out_pitch": HintDef("Pitch 俯仰", is_output=True,
        plain="解出来的平台绕 Y 轴倾斜了多少度。看三维图里平台前后俯仰的样子。",
        diagram="axis_y", diagram_label="Pitch：绕 Y 轴倾斜",
        formula=_EQ_ROT, note="绕 Y 轴的姿态角。"),
    "out_yaw": HintDef("Yaw 偏航", is_output=True,
        plain="解出来的平台绕竖直 Z 轴转了多少度（水平方向转了多少）。",
        diagram="axis_z", diagram_label="Yaw：绕 Z 轴旋转",
        formula=_EQ_ROT, note="绕 Z 轴的姿态角。"),

    # ================= 输出：缸长与精度 =================
    "out_length": HintDef("缸长 / 所需缸长", is_output=True,
        plain="解算得到的每根缸应有的孔中心距。看三维图里 6 根缸的长度变化。"
               "逆解时这就是你要发给电机的目标长度。",
        diagram="legs", diagram_label="L1~L6：各缸长度",
        formula="<p>由当前位姿闭式反算每根缸长度：</p>" + _EQ_LEN,
        note="上铰点与下铰点的欧氏距离。"),
    "out_delta": HintDef("相对零位 ΔL", is_output=True,
        plain="当前缸长比零位长了多少（正数=伸出，负数=缩回）。"
               "这个值直接对应作动器要走的距离，是控制上最常用的量。",
        diagram="legs", diagram_label="ΔL1~ΔL6：伸出量",
        formula='<div class="eq">ΔL<sub>i</sub> = L<sub>i</sub> − L<sub>0,i</sub></div>' + _EQ_LEN,
        note="当前缸长与零位缸长之差。"),
    "out_residual": HintDef("长度残差", is_output=True,
        plain="正解算完后，用求出的姿态反算回缸长，和你输入的缸长相差多少。"
               "这个数越接近 0 越好；如果很大（比如超过 0.01mm），"
               "说明你输入的 6 个缸长在现实中凑不出这个姿态，或者解算没收敛。",
        diagram="legs", diagram_label="输入缸长 vs 反算缸长",
        formula="<p>求得位姿反算缸长与输入缸长之差：</p>"
                '<div class="eq">r<sub>i</sub> = L<sub>i</sub><sup>calc</sup> − L<sub>i</sub><sup>input</sup></div>'
                "<p>理想应趋近 0；过大说明输入缸长物理不可达或未收敛。</p>",
        note="衡量正解精度的核心指标。"),
    "out_rms": HintDef("RMS 误差", is_output=True,
        plain="6 根缸残差的总体平均大小。可以当成『这次解算整体有多准』的单一分数，"
               "越接近 0 越好。通常在 1e-8 量级就非常好了。",
        formula='<div class="eq">RMS = √( (1/6) · Σ<sub>i=1</sub><sup>6</sup> r<sub>i</sub>² )</div>',
        note="残差的均方根。"),
    "out_maxerr": HintDef("最大残差", is_output=True,
        plain="6 根缸里残差最大的那个值。软件用它判断收敛与否，比 RMS 更能发现个别缸的异常。",
        formula='<div class="eq">r<sub>max</sub> = max<sub>i</sub> | r<sub>i</sub> |</div>',
        note="残差绝对值的最大值。"),
    "out_joint": HintDef("节点摆角", is_output=True,
        plain="每根缸相对节点铰中轴线歪了多少度。超过你在参数页设的许用摆角，"
               "就会显示黄色警告——说明这个姿态会顶到铰的机械极限，实际做不到。"
               "三维图里能看到缸的倾斜。",
        diagram="platform_joint_limit", diagram_label="缸轴与节点中轴的夹角 θ",
        formula="<p>缸轴与节点铰中轴线的夹角：</p>"
                '<div class="eq">u<sub>i</sub> = ( t + R·p<sub>i</sub> − b<sub>i</sub> ) / L<sub>i</sub></div>'
                '<div class="eq">θ<sub>i</sub> = arccos( u<sub>i</sub> · a ) × 180/π</div>'
                "<p>a 为节点中轴单位向量（默认下节点 +Z、上节点 −Z）。</p>",
        note="节点摆角，用于判断铰是否超限。"),
    "out_rotation": HintDef("旋转矩阵 R", is_output=True,
        plain="这是把平台姿态写成 3×3 矩阵的形式，是程序内部计算用的。"
               "一般看 Roll/Pitch/Yaw 三个角就够了；只有在要做坐标变换、"
               "把平台上的点换算到世界坐标时才需要它。",
        formula=_EQ_ROT, note="由欧拉角生成的 3×3 方向余弦矩阵。"),
    "out_anchor": HintDef("上铰点世界坐标", is_output=True,
        plain="求解后每个上铰点在【世界坐标系】里的实际位置（X,Y,Z）。"
               "看三维图里上平台的 6 个绿色铰点，就是它们。",
        diagram="platform_points", diagram_label="P1~P6 的世界坐标",
        formula='<div class="eq">P<sub>i</sub><sup>world</sup> = t + R · p<sub>i</sub></div>',
        note="上铰点在世界系中的位置。"),

    # ================= 本轮新增功能（无对应机构部位，diagram=None）=================
    "fk_solver": HintDef("正解求解器",
        plain="选哪种数值方法反求平台姿态。TRF 阻尼最小二乘最稳；LM 收敛快但对初值敏感；"
              "AUTO 两种都跑一遍、取残差更小的结果。一般保持推荐项即可。",
        formula=_EQ_FK, note="正解迭代算法：TRF / LM / AUTO。"),
    "perf_svd": HintDef("性能诊断 SVD", is_output=True,
        plain="把雅可比按特征长度归一化后做奇异值分解。σ 越小越接近奇异（该方向动不了）；"
              "条件数越接近 1 越各向同性。这里只做诊断，不改动任何求解结果。",
        formula='<div class="eq">J<sub>n</sub> = J·diag(1,1,1,1/L,1/L,1/L) = U·Σ·V<sup>T</sup></div>',
        note="归一化雅可比奇异值/条件数/可操作性。"),
    "ws_sampling": HintDef("工作空间采样",
        plain="开关在『三维姿态』页工具栏的『点云』按钮。采样后点云直接叠加在主视口，"
              "按当前行程与摆角约束撒点并逐点判限位；颜色按可操作性的 5~95 百分位归一化，"
              "深蓝=差、青绿=中、亮橙红=好。采样完成会自动聚焦点云与机构；"
              "CSV 仍导出完整 XYZ+Roll/Pitch/Yaw 六列。仅作可视化预览，不改变配置。",
        note="点云叠加在三维姿态主视口，颜色=可操作性（高对比三段色带）。"),
    "ws_action_max": HintDef("各动作最大值", is_output=True,
        plain="应用参数后自动列出 X/Y/Z/Roll/Pitch/Yaw 在行程与摆角约束下的可达范围，"
              "由工作空间限位规则实时计算。改动行程数值即可实时预览，无需点应用。",
        note="位姿各轴可达范围（按行程/摆角约束）。"),
    "perf_ellipsoid": HintDef("力椭球",
        plain="开关在『三维姿态』页工具栏。打开后在视口画出等缸力下平台各方向的响应椭球："
              "半轴越长表示该方向越容易被驱动。数值面板（σ/条件数）仍在平台参数页。",
        formula='<div class="eq">半轴 ∝ 1/σ<sub>i</sub>（归一化到最大值 1）</div>',
        note="等缸力响应椭球，开关在三维姿态页。"),
    "ik_quaternion": HintDef("四元数输入",
        plain="四元数 [w,x,y,z] 用于和 ROS/Unity 交换姿态。输入后点『应用』会转成"
              " Roll/Pitch/Yaw 填进上面的目标位姿框。注意顺序是 w 在前，避开欧拉角万向锁。",
        formula=_EQ_ROT, note="四元数 [w,x,y,z] → Roll/Pitch/Yaw。"),
    "vp_drag_pose": HintDef("拖拽平台",
        plain="打开后在主视口左键拖动上平台平移 X/Y，按住 Shift 拖动改 Z；"
              "软件实时反解缸长并把位姿钳制在限位内。超出范围拖不动是正常保护。",
        note="视口内直接拖拽改变上平台位姿。"),
    "traj_play": HintDef("轨迹播放",
        plain="导入 CSV 每行 6 列（位姿或缸长），可选第 7 列 dwell(ms)。解析后可逐帧"
              "逆解/下发。缸速列按相邻帧位移与 dwell 估算，超最大线速度的帧会高亮但不会被拦截。",
        note="CSV 轨迹逐帧播放与下发，超速帧高亮。"),
    "jog_control": HintDef("点动控制",
        plain="六向 +/- 小步移动，按『步长』相对当前位姿微调，逆解后下发，"
              "用于实机对零微调。需先开启调试模式，且步长过大会被限位钳制。",
        note="六向小步点动，用于实机微调。"),
    "out_motor_torque": HintDef("电机扭矩/转速摘要", is_output=True,
        plain="由各缸位移与缸速结合丝杆导程换算出的电机圈数、转速与扭矩摘要，"
              "填写导程后显示。额定转矩在电缸管理里设置，可对照是否过载。",
        formula='<div class="eq">rev = ΔL/导程,&nbsp; n = 60·|v|/导程&nbsp; (r/min),&nbsp; T = F·导程/(2π·η·1000)</div>',
        note="派生：圈数/转速/扭矩，η=0.9。"),
    "batch_async": HintDef("批量计算",
        plain="多行缸长在后台线程批量正解，进度条显示完成情况；再点一次『批量求解』可取消。"
              "结果含各轴偏移、最大残差、铰点校验与状态。",
        note="多行正解后台计算，可取消。"),
    "ik_dyn_cache": HintDef("逆解动态量/采样间隔",
        plain="填写采样间隔 dt 后可输出逆解所需的缸速与电机转速；dt 只影响动态量，"
              "不影响静态逆解缸长。连续计算两次时才会用上一帧速度求加速度。",
        formula=_EQ_ACC, note="逆解速度/转速需采样间隔 dt。"),
}

# ---- 电机学 15 字段的问号：从 actuator_specs 动态生成，公式单一真相源 ----
def _build_motor_hints():
    from actuator_specs import FIELDS as _FIELDS
    for f in _FIELDS:
        body = f.formula_text
        if f.derived:
            plain = (f"这是自动计算项：{body}。"
                     "修改源数据后本框会自动填入；若手动改动会被锁定，"
                     "点旁边的『↺』可恢复自动计算。")
            formula = f'<div class="eq">{body}</div>'
            note = f"{f.label}（派生）：{body}"
        else:
            plain = body
            formula = ""
            note = f"{f.label}（{f.unit}）" + ("，调试模式必要项。" if f.necessary else "。")
        HINTS[f"act_{f.key}"] = HintDef(
            f.label,
            plain=plain,
            diagram="legs",
            diagram_label=f"{f.label} · {f.unit}",
            formula=formula,
            note=note,
        )


_build_motor_hints()

HINTS["dbg_revolutions"] = HintDef(
    "电机圈数与转速",
    plain="调试模式下显示：每根缸相对零位的位移换算成电机圈数，以及当前缸速换算成电机转速。"
          "需要丝杆导程(screw_lead)才能计算。",
    diagram="legs", diagram_label="各缸位移 → 电机圈数",
    formula='<div class="eq">圈数 rev = ΔL / 导程L</div>'
            '<div class="eq">转速 n = 60 × |缸速v| / 导程L&nbsp; (r/min)</div>',
    note="调试模式派生：圈数=位移/导程，转速=60·缸速/导程。")


def hint_for(key: str) -> HintDef:
    if key in HINTS:
        return HINTS[key]
    # 兼容旧的 legs 等裸标识
    for h in HINTS.values():
        if h.diagram == key:
            return h
    return HintDef("说明", plain="这一项没有额外的说明。", note="暂无该数据项的说明。")


# ==================== 问号按钮 ====================

def make_hint_button(key: str, parent=None, tooltip: str | None = None) -> QToolButton:
    """创建统一的圆形小问号按钮。点击信号由 MainWindow 统一连接。"""
    b = QToolButton(parent)
    b.setObjectName("hint")
    b.setText("?")
    b.setToolTip(tooltip or hint_for(key).title + " — 点击查看怎么填")
    b.setCursor(Qt.PointingHandCursor)
    b._hint_key = key
    return b


# ==================== 线框图 ====================

class PlatformDiagram(QWidget):
    """Stewart 平台等轴测线框图，高亮对应部位并加引线标注。"""

    def __init__(self, config, highlight: str | None, theme, label: str = "", parent=None):
        super().__init__(parent)
        self.config = config
        self.highlight = highlight or ""
        self.theme = theme
        self.label_text = label
        self.setMinimumHeight(360)
        self._hl = self._parse_highlight(self.highlight)

    def set_theme(self, theme):
        self.theme = theme
        self.update()

    def _parse_highlight(self, h: str) -> dict:
        res = {"legs": set(), "base": set(), "plat": set(),
               "ring_b": False, "ring_p": False, "height": False,
               "axis": None, "radius": None, "angle": None}
        if h.startswith("leg") and h[3:].isdigit():
            res["legs"].add(int(h[3:]) - 1)
        elif h.startswith("base_point") and h[10:].isdigit():
            res["base"].add(int(h[10:]) - 1)
        elif h.startswith("platform_point") and h[14:].isdigit():
            res["plat"].add(int(h[14:]) - 1)
        elif h in ("base_radius", "platform_radius"):
            res["radius"] = h
            res["ring_b" if h == "base_radius" else "ring_p"] = True
        elif h in ("base_pair_angle", "platform_pair_angle"):
            res["angle"] = h
            res["ring_b" if "base" in h else "ring_p"] = True
        elif h in ("base_phase", "pair_centers", "base_joint_z", "base_points",
                   "base_joint_limit"):
            res["ring_b"] = True
        elif h in ("platform_phase", "platform_joint_z", "platform_points",
                   "platform_joint_limit"):
            res["ring_p"] = True
        elif h in ("home_height",):
            res["height"] = True
        elif h in ("axis_x", "axis_y", "axis_z"):
            res["axis"] = h[-1].upper()
        elif h in ("legs", "stroke", "zero_lengths"):
            res["legs"] = set(range(6))
        elif h in ("platform_joint_limit",):
            res["legs"] = set(range(6))
        return res

    def _line(self, p, a, b, color, width, glow=False, dash=None):
        if glow:
            glow_color = QColor(color); glow_color.setAlphaF(0.22)
            pen = QPen(glow_color)
            pen.setWidthF(width + 10); pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen); p.drawLine(a, b)
        pen = QPen(color); pen.setWidthF(width); pen.setCapStyle(Qt.RoundCap)
        if dash:
            pen.setStyle(dash)
        p.setPen(pen); p.drawLine(a, b)

    def _callout(self, p, anchor: QPointF, text: str, color: QColor, w: int, h: int):
        """画引线 + 标注框，指向高亮部位。"""
        if not text:
            return
        # 引线从锚点斜向引出
        tx = min(max(anchor.x() + 26, 12), w - 190)
        ty = min(max(anchor.y() - 34, 14), h - 56)
        tip = QPointF(tx, ty)
        pen = QPen(color); pen.setWidthF(1.2)
        p.setPen(pen); p.drawLine(anchor, tip)
        box = QRectF(tx, ty - 20, min(178, w - tx - 10), 22)
        p.setBrush(QBrush(QColor(self.theme.surface)))
        p.setPen(QPen(QColor(self.theme.border)))
        p.drawRoundedRect(box, 5, 5)
        p.setPen(QColor(color))
        f = QFont(); f.setPointSize(9); f.setBold(True); p.setFont(f)
        p.drawText(box, Qt.AlignCenter, text)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        t = self.theme
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(t.surface_alt))

        try:
            base, plat_local = self.config.anchors()
            top = plat_local + self.config.origin()
        except Exception:
            p.setPen(QColor(t.danger))
            p.drawText(self.rect(), Qt.AlignCenter, "铰点参数无效，无法绘制示意图")
            return

        pts = np.vstack([base, top])
        span_xy = max(float(np.max(np.abs(pts[:, 0]))), float(np.max(np.abs(pts[:, 1]))), 1.0)
        span_z = max(float(np.max(pts[:, 2])), 1.0)
        scale = min((w * 0.31) / span_xy, (h * 0.56) / span_z)
        cx, cy = w * 0.47, h * 0.76

        def proj(pt):
            return QPointF(cx + (float(pt[0]) - float(pt[1])) * 0.866 * scale,
                           cy + (float(pt[0]) + float(pt[1])) * 0.5 * scale
                           - float(pt[2]) * scale)

        c_b = QColor(t.input)
        c_p = QColor(t.output)
        c_line = QColor(t.text_2)
        c_dim = QColor(t.text_3)
        c_hot = QColor(t.accent)

        poly_b = QPolygonF([proj(x) for x in base])
        poly_t = QPolygonF([proj(x) for x in top])

        # 地面网格
        grid_pen = QPen(QColor(t.grid)); grid_pen.setWidthF(0.7); grid_pen.setStyle(Qt.DotLine)
        p.setPen(grid_pen)
        gr = span_xy * 1.2
        for k in np.linspace(-gr, gr, 7):
            p.drawLine(proj([k, -gr, 0]), proj([k, gr, 0]))
            p.drawLine(proj([-gr, k, 0]), proj([gr, k, 0]))

        # 坐标轴
        o = proj([0, 0, 0])
        alen = span_xy * 0.42
        axis_defs = [(np.array([alen, 0, 0]), "X", QColor(t.danger)),
                     (np.array([0, alen, 0]), "Y", QColor(t.success)),
                     (np.array([0, 0, alen]), "Z", QColor(t.accent))]
        for vec, name, col in axis_defs:
            q = proj(vec)
            hot = self._hl["axis"] == name
            self._line(p, o, q, col, 1.8 if hot else 1.1, glow=hot)
            p.setPen(col)
            f = QFont(); f.setBold(True); f.setPointSize(9 if hot else 8); p.setFont(f)
            p.drawText(q + QPointF(4, -3), name)
        if self._hl["axis"]:
            self._callout(p, o, self.label_text or "坐标轴", c_hot, w, h)

        # 下平台
        p.setBrush(QBrush(QColor(c_b.red(), c_b.green(), c_b.blue(), 26)))
        pen = QPen(c_b); pen.setWidthF(3.2 if self._hl["ring_b"] else 1.8)
        p.setPen(pen); p.drawPolygon(poly_b)

        # 上平台
        p.setBrush(QBrush(QColor(c_p.red(), c_p.green(), c_p.blue(), 32)))
        pen = QPen(c_p); pen.setWidthF(3.2 if self._hl["ring_p"] else 1.8)
        p.setPen(pen); p.drawPolygon(poly_t)

        # 半径标注（底座中心 → 一个下铰点 / 平台中心 → 一个上铰点）
        if self._hl["radius"]:
            is_base = self._hl["radius"] == "base_radius"
            ctr = proj([0, 0, 0] if is_base else self.config.origin())
            ring = proj(base[0] if is_base else top[0])
            self._line(p, ctr, ring, c_hot, 2.6, glow=True, dash=Qt.DashLine)
            self._callout(p, QPointF((ctr.x() + ring.x()) / 2, (ctr.y() + ring.y()) / 2),
                          self.label_text or ("Rb" if is_base else "Rp"), c_hot, w, h)

        # 中心高度虚线
        cb = proj([0, 0, 0]); ct = proj(self.config.origin())
        self._line(p, cb, ct, c_dim, 1.2, glow=self._hl["height"], dash=Qt.DashLine)
        if self._hl["height"]:
            self._callout(p, QPointF((cb.x() + ct.x()) / 2, (cb.y() + ct.y()) / 2),
                          self.label_text or "H0", c_hot, w, h)

        # 6 根缸
        f = QFont(); f.setPointSize(8); p.setFont(f)
        for i in range(6):
            a, b = proj(base[i]), proj(top[i])
            hot = i in self._hl["legs"]
            self._line(p, a, b, c_hot if hot else c_line, 3.4 if hot else 1.5, glow=hot)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(c_hot if hot else c_b)); p.drawEllipse(a, 4 if hot else 2.6, 4 if hot else 2.6)
            p.setBrush(QBrush(c_hot if hot else c_p)); p.drawEllipse(b, 4 if hot else 2.6, 4 if hot else 2.6)
            p.setPen(c_hot if hot else c_dim)
            mid = QPointF((a.x() + b.x()) / 2, (a.y() + b.y()) / 2)
            p.drawText(mid + QPointF(3, -3), f"L{i+1}")
        if self._hl["legs"]:
            self._callout(p, proj((base[0] + top[0]) / 2),
                          self.label_text or "L1~L6", c_hot, w, h)

        # 铰点编号
        f = QFont(); f.setPointSize(8); p.setFont(f)
        for i in range(6):
            a, b = proj(base[i]), proj(top[i])
            p.setPen(c_b if i not in self._hl["base"] else c_hot)
            p.drawText(a + QPointF(-18, 3), f"B{i+1}")
            p.setPen(c_p if i not in self._hl["plat"] else c_hot)
            p.drawText(b + QPointF(4, -6), f"P{i+1}")

        # 图例
        p.setPen(c_dim)
        f = QFont(); f.setPointSize(9); p.setFont(f)
        p.drawText(QRectF(10, h - 24, w - 20, 18), Qt.AlignLeft,
                   "蓝=下平台 B · 绿=上平台 P · 编号=缸 L")
        if self.highlight:
            p.drawText(QRectF(10, h - 24, w - 20, 18), Qt.AlignRight,
                       "高亮：" + hint_for(self.highlight).title)


# ==================== 三页签弹窗 ====================

def _rich_widget(html: str, theme) -> QWidget:
    area = QScrollArea(); area.setWidgetResizable(True); area.setFrameShape(QFrame.NoFrame)
    inner = QWidget(); area.setWidget(inner)
    lay = QVBoxLayout(inner); lay.setContentsMargins(18, 16, 18, 16); lay.addStretch(0)
    lab = QLabel(html); lab.setWordWrap(True); lab.setTextFormat(Qt.RichText)
    lab.setOpenExternalLinks(False)
    lay.addWidget(lab); lay.addStretch(1)
    return area


def _plain_html(hint: HintDef, theme) -> str:
    badge = "输出数据" if hint.is_output else "输入数据"
    badge_color = theme.output if hint.is_output else theme.input
    return f"""
    <style>
      body {{ font-family:'Microsoft YaHei UI','Microsoft YaHei'; font-size:14px;
              color:{theme.text}; line-height:1.9; }}
      h2 {{ color:{theme.accent}; font-size:15px; margin:0 0 12px 0; }}
      .badge {{ color:{badge_color}; font-size:11px; font-weight:700; }}
      .big {{ font-size:15px; color:{theme.text}; }}
      .box {{ background:{theme.surface}; border:1px solid {theme.border};
              border-left:3px solid {badge_color}; border-radius:8px;
              padding:14px 16px; margin:10px 0; }}
      .sub {{ color:{theme.text_2}; font-size:12px; }}
      opt {{ color:{theme.warning}; font-weight:700; }}
    </style>
    <h2>{hint.title} <span class="badge">［{badge}］</span></h2>
    <div class="box"><div class="big">{hint.plain or hint.note}</div></div>
    <p class="sub">{hint.note}</p>
    <p class="sub">点上方「在哪个位置」看可旋转三维模型（红色标出部位）；「计算公式」看推导。</p>
    """


def _formula_html(hint: HintDef, theme) -> str:
    return f"""
    <style>
      body {{ font-family:'Microsoft YaHei UI','Microsoft YaHei'; font-size:13px;
              color:{theme.text}; line-height:1.8; }}
      h2 {{ color:{theme.accent}; font-size:15px; margin:0 0 10px 0; }}
      .eq {{ font-family:'Cambria Math','Times New Roman',serif; font-size:17px;
             color:{theme.output}; background:{theme.surface_alt};
             border:1px solid {theme.border}; border-radius:9px;
             padding:12px 14px; margin:9px 0; }}
      opt {{ color:{theme.warning}; font-weight:700; }}
      .sub {{ color:{theme.text_2}; font-size:12px; }}
    </style>
    <h2>{hint.title} — 计算公式</h2>
    {hint.formula or '<p class="sub">这一项是配置/策略参数，没有数学公式。请看「怎么填」页的说明。</p>'}
    """


class HintDialog(QDialog):
    """单个数据项帮助弹窗：怎么填 / 在哪个位置 / 计算公式。"""

    def __init__(self, key: str, config, theme, parent=None):
        super().__init__(parent)
        self.key = key
        self.config = config
        self.hint = hint_for(key)
        self.setWindowTitle(self.hint.title)
        self.resize(760, 580)
        self.setMinimumSize(620, 460)
        self._build(theme)
        self.apply_theme(theme)
        try:
            from app import frameless
            frameless(self, self.hint.title)
        except Exception:
            pass

    def _build(self, theme):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 16)
        root.setSpacing(10)
        self.tabs = QTabWidget()

        # 页 1：怎么填 / 怎么看
        self.tabs.addTab(_rich_widget(_plain_html(self.hint, theme), theme), "怎么填" if not self.hint.is_output else "怎么看")

        # 页 2：在哪个位置（可旋转三维模型 + 红色高亮）
        if self.hint.diagram:
            holder = self._make_3d_view(theme)
            self.tabs.addTab(holder, "在哪个位置")
        else:
            nofig = _rich_widget(f"""
            <style>
              body {{ font-family:'Microsoft YaHei UI','Microsoft YaHei'; font-size:14px;
                      color:{theme.text}; line-height:1.9; }}
              h2 {{ color:{theme.accent}; font-size:15px; }}
              .box {{ background:{theme.surface_alt}; border:1px solid {theme.border};
                      border-radius:8px; padding:14px 16px; }}
            </style>
            <h2>{self.hint.title}</h2>
            <div class="box">
              <b>这一项没有对应的机构部位，因此没有线框图。</b><br><br>
              {self.hint.plain or self.hint.note}
            </div>
            """, theme)
            self.tabs.addTab(nofig, "在哪个位置")

        # 页 3：计算公式
        self.tabs.addTab(_rich_widget(_formula_html(self.hint, theme), theme), "计算公式")

        root.addWidget(self.tabs, stretch=1)
        btn = QPushButton("知道了")
        btn.setObjectName("primary")
        btn.clicked.connect(self.accept)
        row = QHBoxLayout(); row.addStretch(); row.addWidget(btn); row.addStretch()
        root.addLayout(row)

    def apply_theme(self, theme):
        style = f"""
        QDialog {{ background:{theme.bg}; color:{theme.text}; }}
        QTabWidget::pane {{ border:1px solid {theme.border}; border-radius:9px;
                            background:{theme.bg}; margin-top:6px; }}
        QTabBar::tab {{ background:transparent; color:{theme.text_2}; padding:9px 18px;
                        font-size:12px; border-bottom:2px solid transparent; }}
        QTabBar::tab:selected {{ color:{theme.accent}; border-bottom:2px solid {theme.accent};
                                 font-weight:700; }}
        """
        self.setStyleSheet(style)
        if hasattr(self, "diagram"):
            self.diagram.set_theme(theme)
        # 重建三个页面以刷新内联样式
        idx = self.tabs.currentIndex()
        for i in range(self.tabs.count()):
            self.tabs.removeTab(self.tabs.count() - 1)
        self._build_pages(theme)
        if 0 <= idx < self.tabs.count():
            self.tabs.setCurrentIndex(idx)

    def _make_3d_view(self, theme) -> QWidget:
        """可旋转三维视图：红色高亮该数据对应部位 + 操作提示。"""
        holder = QWidget(); hl = QVBoxLayout(holder)
        hl.setContentsMargins(10, 10, 10, 10); hl.setSpacing(8)

        tip = QLabel(f"<span style='color:{theme.highlight};font-weight:700;'>● 红色</span> "
                     f"标出「{self.hint.title}」对应的部位　·　"
                     "拖拽旋转 · 滚轮缩放 · 双击复位")
        tip.setTextFormat(Qt.RichText)
        hl.addWidget(tip)

        self.diagram = Viewport3D(theme=theme)
        self.diagram.set_model(None, self.config)   # 仅几何，不需要 solver
        self.diagram.show_grid = True
        self.diagram.show_home = False              # 帮助视图隐藏零位参考，减少干扰
        self.diagram.set_highlight(self.hint.diagram, self.hint.diagram_label)
        hl.addWidget(self.diagram, stretch=1)
        return holder

    def _build_pages(self, theme):
        # apply_theme 时重建页内容（保留 tabs 与按钮布局）
        self.tabs.addTab(_rich_widget(_plain_html(self.hint, theme), theme),
                         "怎么填" if not self.hint.is_output else "怎么看")
        if self.hint.diagram:
            self.tabs.addTab(self._make_3d_view(theme), "在哪个位置")
        else:
            self.tabs.addTab(_rich_widget(f"""
            <div style="font-family:'Microsoft YaHei UI','Microsoft YaHei'; font-size:14px;
                        color:{theme.text}; line-height:1.9;">
              <h2 style="color:{theme.accent}; font-size:15px;">{self.hint.title}</h2>
              <div style="background:{theme.surface_alt}; border:1px solid {theme.border};
                          border-radius:8px; padding:14px 16px;">
                <b>这一项没有对应的机构部位，因此没有线框图。</b><br><br>
                {self.hint.plain or self.hint.note}
              </div>
            </div>""", theme), "在哪个位置")
        self.tabs.addTab(_rich_widget(_formula_html(self.hint, theme), theme), "计算公式")
