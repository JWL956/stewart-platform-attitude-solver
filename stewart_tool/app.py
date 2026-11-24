# -*- coding: utf-8 -*-
"""六自由度平台姿态解算软件 —— 简约高级风格 PySide6 界面。

特性：
- 侧边导航 + 卡片式布局，浅色/深色双主题实时切换
- 输入(蓝)/输出(绿)数据色区分，每项数据挂接问号帮助
- 问号：输入项定位到平台框线图部位，输出项展示计算公式
- 内置可交互三维姿态视口（拖拽旋转/滚轮缩放/零位参考）
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import math
from PySide6.QtCore import Qt, QObject, Signal, Slot, QThread, QPointF, QRectF, QRect, QEvent, QPropertyAnimation, QEasingCurve, QSignalBlocker, QTimer, QPoint, QUrl
from PySide6.QtGui import QColor, QBrush, QFont, QPainter, QPainterPath, QPolygonF, QRegion, QPen, QIcon, QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QGridLayout, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView,
    QFileDialog, QMessageBox, QPlainTextEdit, QCheckBox, QRadioButton, QFrame, QScrollArea,
    QButtonGroup, QStackedWidget, QDialog, QToolButton, QInputDialog, QProgressBar,
)

from kinematics import PlatformConfig, StewartKinematics, default_config, STRUCTURES, quat_from_pose, pose_from_quat, effective_speed_cap
from actuator_specs import (
    FIELDS, FIELDS_BY_KEY, DERIVED_FIELDS, recompute, validate_debug_prerequisites,
    EFFICIENCY, EXTRA_FIELDS, EXTRA_BY_KEY, UI_ORDER_KEYS, motor_revolutions, motor_torque,
)
from shiboken6 import isValid as shiboken_valid
from device_link import DeviceLinkDetector, LinkState
from profile_manager import ProfileManager
from theme import ThemeManager, build_style, apply_palette
from hints import HintDialog, make_hint_button
from viewport3d import Viewport3D

APP_TITLE = "六自由度姿态解算"
APP_VERSION = "v3.0"
REPO_OWNER = "JWL956"                      # GitHub 用户名
REPO_NAME = "stewart-platform-attitude-solver"
REPO_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}"


# ==================== 通用组件 ====================

class BrandMark(QWidget):
    """自绘六边形品牌标识（避免字体缺字显示成方块）。"""
    def __init__(self, color="#5C86B8", size=34, parent=None):
        super().__init__(parent)
        self.color = QColor(color)
        self.sz = size
        self.setFixedSize(size, size)

    def set_color(self, color):
        self.color = QColor(color); self.update()

    def paintEvent(self, ev):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(self.color, 2.0))
        p.setBrush(Qt.NoBrush)
        cx = cy = self.sz / 2.0
        r = self.sz / 2.0 - 3.0
        hex_outer = QPolygonF([QPointF(cx + r * __import__("math").cos(i * __import__("math").pi / 3),
                                       cy + r * __import__("math").sin(i * __import__("math").pi / 3))
                               for i in range(6)])
        p.drawPolygon(hex_outer)
        # 中心小六边形
        r2 = r * 0.42
        hex_inner = QPolygonF([QPointF(cx + r2 * __import__("math").cos(i * __import__("math").pi / 3),
                                       cy + r2 * __import__("math").sin(i * __import__("math").pi / 3))
                               for i in range(6)])
        p.drawPolygon(hex_inner)
        p.end()


def app_icon() -> QIcon:
    p = Path(__file__).resolve().parent / "app.ico"
    return QIcon(str(p)) if p.exists() else QIcon()


def draw_nav_icon(p: QPainter, kind: str, cx: float, cy: float, color: QColor, size: float = 20):
    """在 (cx,cy) 绘制简洁线性矢量图标（无外部资源）。"""
    p.save()
    p.setRenderHint(QPainter.Antialiasing, True)
    pen = QPen(color); pen.setWidthF(1.8); pen.setCapStyle(Qt.RoundCap); pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen); p.setBrush(Qt.NoBrush)
    s = size / 2.0
    def ln(x1, y1, x2, y2):
        # 传入的坐标已含 s 缩放，这里只加中心偏移
        p.drawLine(QPointF(cx + x1, cy + y1), QPointF(cx + x2, cy + y2))
    if kind == "actuator":            # 电缸：缸体 + 活塞杆
        p.drawRect(QRectF(cx - 0.8*s, cy - 0.35*s, 1.1*s, 0.7*s))
        ln(0.3*s, 0, 0.95*s, 0); ln(-0.8*s, -0.6*s, -0.8*s, 0.6*s)
    elif kind == "param":             # 参数：齿轮（简化为圆+四齿）
        p.drawEllipse(QPointF(cx, cy), 0.45*s, 0.45*s)
        for a in (0, 90, 180, 270):
            rad = math.radians(a)
            ln(0.45*s*math.cos(rad), 0.45*s*math.sin(rad),
               0.85*s*math.cos(rad), 0.85*s*math.sin(rad))
    elif kind == "forward":           # 正解：缸长→姿态（箭头+平台）
        ln(-0.9*s, 0.6*s, 0.2*s, -0.3*s); ln(-0.5*s, 0.6*s, 0.6*s, -0.3*s)
        p.drawRect(QRectF(cx - 0.2*s, cy - 0.8*s, 1.0*s, 0.28*s))
    elif kind == "inverse":           # 逆解：姿态→缸长
        ln(0.9*s, 0.6*s, -0.2*s, -0.3*s); ln(0.5*s, 0.6*s, -0.6*s, -0.3*s)
        p.drawRect(QRectF(cx - 0.8*s, cy - 0.8*s, 1.0*s, 0.28*s))
    elif kind == "view":              # 三维：立方体线框
        p.drawRect(QRectF(cx - 0.6*s, cy - 0.3*s, 0.9*s, 0.9*s))
        ln(-0.6*s, -0.3*s, -0.2*s, -0.7*s); ln(0.3*s, -0.3*s, 0.7*s, -0.7*s)
        ln(-0.2*s, -0.7*s, 0.7*s, -0.7*s); ln(0.3*s, 0.6*s, 0.7*s, 0.2*s)
        ln(0.7*s, 0.2*s, 0.7*s, -0.7*s)
    elif kind == "kinematics":        # 运动学：波形
        p.drawPolyline([QPointF(cx + x*s, cy + y*s) for x, y in
                        [(-1,0.3),(-0.5,-0.4),(0,0.4),(0.5,-0.4),(1,0.3)]])
    elif kind == "batch":             # 批量：表格
        p.drawRect(QRectF(cx - 0.8*s, cy - 0.75*s, 1.6*s, 1.5*s))
        ln(-0.8*s, -0.25*s, 0.8*s, -0.25*s); ln(-0.8*s, 0.25*s, 0.8*s, 0.25*s)
        ln(0.0*s, -0.75*s, 0.0*s, 0.75*s)
    elif kind == "help":              # 帮助：问号圆圈
        p.drawEllipse(QPointF(cx, cy), 0.75*s, 0.75*s)
        p.drawText(QRectF(cx-0.5*s, cy-0.6*s, s, s), Qt.AlignCenter, "?")
    elif kind == "archive":           # 档案：文件夹
        ln(-0.85*s, -0.55*s, -0.15*s, -0.55*s); ln(-0.15*s, -0.55*s, 0.0*s, -0.3*s)
        ln(0.0*s, -0.3*s, 0.85*s, -0.3*s); ln(0.85*s, -0.3*s, 0.85*s, 0.65*s)
        ln(0.85*s, 0.65*s, -0.85*s, 0.65*s); ln(-0.85*s, 0.65*s, -0.85*s, -0.55*s)
    elif kind == "back":              # 返回：←
        ln(-0.8*s, 0, 0.8*s, 0); ln(-0.8*s, 0, -0.25*s, -0.5*s); ln(-0.8*s, 0, -0.25*s, 0.5*s)
    elif kind == "pin":               # 锁定图钉
        ln(-0.15*s, 0.7*s, -0.15*s, -0.1*s); ln(-0.6*s, -0.15*s, 0.6*s, -0.15*s)
        ln(-0.45*s, -0.15*s, -0.3*s, -0.7*s); ln(0.45*s, -0.15*s, 0.3*s, -0.7*s)
        ln(-0.3*s, -0.7*s, 0.3*s, -0.7*s)
    elif kind == "unpin":             # 解锁（倾斜图钉）
        ln(-0.1*s, 0.7*s, 0.1*s, 0.05*s); ln(-0.55*s, 0.05*s, 0.55*s, -0.25*s)
        ln(-0.35*s, -0.4*s, -0.1*s, -0.6*s); ln(0.15*s, -0.55*s, 0.4*s, -0.75*s)
    else:                             # 默认：圆点
        p.drawEllipse(QPointF(cx, cy), 0.3*s, 0.3*s)
    p.restore()


class RailButton(QPushButton):
    """侧栏导航按钮：展开显示文字，折叠显示矢量图标。"""

    def __init__(self, text, icon_kind, rail, parent=None):
        super().__init__(parent)
        self._full_text = text
        self.icon_kind = icon_kind
        self.rail = rail
        self.setObjectName("navItem")
        self.setText(text)
        self.setToolTip(text.strip())

    def paintEvent(self, ev):
        if not self.rail.collapsed:
            super().paintEvent(ev)
            return
        # 折叠态：自绘图标（选中态用高亮色）
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing, True)
        if self.isChecked():
            p.fillRect(self.rect(), QColor(self.rail.theme.accent))
            col = QColor(self.rail.theme.on_accent)
        elif self.underMouse():
            p.fillRect(self.rect(), QColor(self.rail.theme.glass_hover))
            col = QColor(self.rail.theme.text)
        else:
            col = QColor(self.rail.theme.sidebar_text)
        draw_nav_icon(p, self.icon_kind, self.width()/2.0, self.height()/2.0, col, 22)
        p.end()


class _PinButton(QPushButton):
    """自绘图钉按钮：锁定/解锁状态显示不同图标。"""

    def __init__(self, rail, locked, parent=None):
        super().__init__(parent)
        self.rail = rail
        self._locked = locked
        self.setToolTip("锁定侧栏保持展开" if not locked else "解锁侧栏（恢复悬停展开）")

    def set_locked(self, locked):
        self._locked = locked
        self.setToolTip("锁定侧栏保持展开" if not locked else "解锁侧栏（恢复悬停展开）")
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing, True)
        if self.underMouse():
            p.fillRect(self.rect(), QColor(self.rail.theme.glass_hover))
        col = QColor(self.rail.theme.accent if self._locked else self.rail.theme.sidebar_text)
        draw_nav_icon(p, "pin" if self._locked else "unpin",
                      self.width()/2.0, self.height()/2.0, col, 18)
        p.end()


class SideRail(QFrame):
    """统一侧边栏组件 —— 档案中心与工作窗口共用同一骨架。

    支持：默认收窄为图标轨 → 悬停滑出 → 图钉锁定保持展开。
    结构：品牌区 → 章节导航 → 弹性区 → 信息块 → 底部动作区。
    """

    COLLAPSED_W = 58
    EXPANDED_W = 212

    def __init__(self, theme, width=212, lock_state=None):
        super().__init__()
        self.setObjectName("sidebar")
        self.theme = theme
        self._expanded_w = width
        self.collapsed = False
        self.locked = bool(lock_state and lock_state.get("locked", False))
        self._lock_state = lock_state
        self.setFixedWidth(self.COLLAPSED_W if not self.locked else self._expanded_w)
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(14, 22, 12, 14)
        self.lay.setSpacing(4)
        self._hidden_on_collapse = []
        self._lock_btn = None
        self._build_brand()
        self.setMouseTracking(True)
        if self.locked:
            self.collapsed = False

    def _build_brand(self):
        brand = QHBoxLayout(); brand.setSpacing(11)
        self.brand_mark = BrandMark(self.theme.accent, 36)
        nb = QVBoxLayout(); nb.setSpacing(2)
        self.brand_name = label("KINEMATICS", "brandName")
        self.brand_sub = label("六自由度姿态工作台", "brandSub")
        nb.addWidget(self.brand_name); nb.addWidget(self.brand_sub)
        brand.addWidget(self.brand_mark); brand.addLayout(nb)
        self.lay.addLayout(brand)
        line = QFrame(); line.setObjectName("divider"); line.setFixedHeight(1)
        line.setStyleSheet(f"background:{self.theme.border_strong};")
        self.lay.addSpacing(18); self.lay.addWidget(line); self.lay.addSpacing(14)
        self._hidden_on_collapse += [self.brand_name, self.brand_sub]

    def section(self, title: str):
        lab = label(title, "navSection")
        self.lay.addWidget(lab); self.lay.addSpacing(4)
        self._hidden_on_collapse.append(lab)

    def add_nav(self, text: str, checked=False, icon_kind=None) -> QPushButton:
        if icon_kind is None:
            icon_kind = self._infer_icon(text)
        b = RailButton(f"  {text}", icon_kind, self)
        if checked is not None:
            b.setCheckable(True); b.setChecked(checked)
        self.lay.addWidget(b)
        return b

    @staticmethod
    def _infer_icon(text):
        t = text.lstrip()
        for key in ("01", "02", "03", "04", "05", "06", "07", "08"):
            if t.startswith(key):
                return {"01": "actuator", "02": "param", "03": "forward", "04": "inverse",
                        "05": "view", "06": "kinematics", "07": "batch", "08": "help"}[key]
        if "档案" in t or "我的" in t:
            return "archive"
        if "←" in t or "返回" in t:
            return "back"
        return "param"

    def add_lock_button(self):
        """图钉锁定按钮：自绘矢量图钉，锁定后侧栏保持展开。"""
        b = _PinButton(self, self.locked)
        b.setObjectName("pinBtn")
        b.setFixedHeight(30)
        b.clicked.connect(self.toggle_lock)
        self.lay.addWidget(b)
        self._lock_btn = b
        return b

    def add_nav_group(self, items: list) -> QButtonGroup:
        group = QButtonGroup(); group.setExclusive(True)
        for i, text in enumerate(items):
            b = self.add_nav(text, checked=(i == 0))
            group.addButton(b, i)
        return group

    def add_spacer(self):
        self.lay.addStretch()

    def info_block(self, title: str, value: str, sub: str = "") -> QLabel:
        info = QFrame(); info.setObjectName("card")
        il = QVBoxLayout(info); il.setContentsMargins(14, 12, 14, 12); il.setSpacing(4)
        il.addWidget(label(title, "sectionTitle"))
        val = label(value, "body", bold=True, wrap=True); il.addWidget(val)
        sublab = label(sub, "caption"); il.addWidget(sublab)
        self.lay.addWidget(info); self.lay.addSpacing(10)
        self._hidden_on_collapse.append(info)
        return sublab

    def add_theme_button(self, toggle_slot) -> QPushButton:
        b = QPushButton("深色模式"); b.setObjectName("themeBtn")
        b.clicked.connect(toggle_slot)
        self.lay.addWidget(b)
        self._hidden_on_collapse.append(b)
        return b

    # ---------- 折叠 / 悬停 / 锁定 ----------
    def enterEvent(self, ev):
        if not self.locked:
            self.set_collapsed(False)
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        if not self.locked:
            self.set_collapsed(True)
        super().leaveEvent(ev)

    def set_collapsed(self, collapsed: bool):
        if self.collapsed == collapsed:
            return
        self.collapsed = collapsed
        # 隐藏/显示文字类控件
        for w in self._hidden_on_collapse:
            w.setVisible(not collapsed)
        # 折叠态：导航按钮固定高度，保证图标有足够空间；展开态恢复默认
        for b in self.findChildren(RailButton):
            b.setMinimumHeight(44 if collapsed else 0)
        # 平滑动画：同时驱动 minimumWidth / maximumWidth，动画结束后再固定，
        # 避免 setFixedWidth 立即覆盖动画造成"瞬移"感。
        target = self.COLLAPSED_W if collapsed else self._expanded_w
        start = self.width()
        self.setMinimumWidth(start); self.setMaximumWidth(start)
        self._anims = []
        for prop in (b"minimumWidth", b"maximumWidth"):
            anim = QPropertyAnimation(self, prop, self)
            anim.setDuration(280)
            anim.setEasingCurve(QEasingCurve.InOutCubic)
            anim.setStartValue(start)
            anim.setEndValue(target)
            self._anims.append(anim)
        for anim in self._anims:
            anim.start()
        # 动画结束把宽度固定到目标，防止布局二次拉伸
        self._anims[0].finished.connect(lambda: (self.setMinimumWidth(target),
                                                 self.setMaximumWidth(target)))
        # 让所有 RailButton 重绘（切换文字/图标）
        for b in self.findChildren(RailButton):
            b.update()

    def toggle_lock(self):
        self.locked = not self.locked
        if self._lock_state is not None:
            self._lock_state["locked"] = self.locked
        if self._lock_btn:
            self._lock_btn.set_locked(self.locked)
        self.set_collapsed(not self.locked)

    def normalize_nav_alignment(self):
        """统一所有导航项文字前导空格为 4，保证每次改动后都对齐。"""
        for b in self.findChildren(QPushButton):
            if b.objectName() == "navItem":
                t = b.text()
                stripped = t.lstrip()
                # 保留返回按钮等特殊项原样（以箭头开头）
                if stripped.startswith("←") or stripped.startswith("<"):
                    continue
                b.setText("    " + stripped)

    def set_theme(self, theme):
        self.theme = theme
        self.brand_mark.set_color(theme.accent)
        for line in self.findChildren(QFrame):
            if line.objectName() == "divider":
                line.setStyleSheet(f"background:{theme.border_strong};")
        for b in self.findChildren(RailButton):
            b.update()


def label(text, oid=None, bold=False, wrap=False):
    lab = QLabel(text)
    if oid:
        lab.setObjectName(oid)
    if bold:
        f = lab.font(); f.setBold(True); lab.setFont(f)
    lab.setWordWrap(wrap)
    return lab


def num_edit(text="0", width=100):
    e = QLineEdit(str(text))
    e.setFixedWidth(width)
    e.setAlignment(Qt.AlignRight)
    return e


def parse_floats(edits, name):
    vals = []
    for i, e in enumerate(edits):
        t = e.text().strip().replace(",", " ")
        try:
            vals.append(float(t.split()[0]))
        except (ValueError, IndexError):
            raise ValueError(f"{name} 第 {i + 1} 项输入无效：{e.text()!r}")
    return np.asarray(vals, dtype=float)


def parse_series(text, count=6):
    for ch in (",", ";", "\t", "\n", "，", "；"):
        text = text.replace(ch, " ")
    parts = text.split()
    if len(parts) < count:
        raise ValueError(f"需要至少 {count} 个数值，当前只识别到 {len(parts)} 个")
    try:
        return [float(p) for p in parts[:count]]
    except ValueError as ex:
        raise ValueError(f"存在无效数值：{ex}")


def card(title=None, tint=None):
    """tint ∈ {param, input, output, view}：给卡片极轻微色差以区分模块。"""
    frame = QFrame()
    frame.setObjectName(f"tint{tint.capitalize()}" if tint else "card")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(16, 14, 16, 14)
    lay.setSpacing(10)
    if title:
        lay.addWidget(label(title, "headline"))
    return frame, lay


def param_cell(text, hint_key, widget):
    """参数网格单元：名称 + 问号 + 输入控件。"""
    w = QWidget(); lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(4)
    lay.addWidget(label(text))
    lay.addWidget(make_hint_button(hint_key))
    lay.addWidget(widget)
    return w


def input_row(text, hint_key, widget, unit=""):
    row = QHBoxLayout(); row.setSpacing(6)
    row.addWidget(label(text))
    row.addWidget(make_hint_button(hint_key))
    row.addWidget(widget)
    if unit:
        row.addWidget(label(unit, "unit"))
    row.addStretch()
    return row


def metric_card(name, hint_key, unit, note=None):
    """输出指标卡片：名称 + 问号 + 绿色数值。note 可覆盖底部说明。"""
    frame = QFrame(); frame.setObjectName("card")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(12, 10, 12, 10)
    lay.setSpacing(3)
    head = QHBoxLayout(); head.setSpacing(5)
    head.addWidget(label(name, "caption"))
    if hint_key:
        head.addWidget(make_hint_button(hint_key))
    head.addStretch()
    lay.addLayout(head)
    v = label("—", "outValue")
    lay.addWidget(v)
    lay.addWidget(label(note or f"相对零位 Δ {unit}", "unit"))
    return frame, v


def scroll_page(builder):
    """构建可滚动页面，返回 (scroll, content_widget, layout)。"""
    scroll = QScrollArea(); scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    w = QWidget(); w.setObjectName("page"); scroll.setWidget(w)
    lay = QVBoxLayout(w)
    lay.setContentsMargins(24, 20, 24, 24)
    lay.setSpacing(16)
    builder(lay, w)
    return scroll, w, lay


# ==================== 开发板连接状态光点 ====================

class LinkIndicator(QWidget):
    """右下角悬浮小圆点：黄=未连接 / 绿=已连接 / 绿呼吸=传输正常 / 红呼吸=传输失败。"""

    SIZE = 14

    def __init__(self, detector: DeviceLinkDetector, parent=None):
        super().__init__(parent)
        self.detector = detector
        self.setFixedSize(self.SIZE + 6, self.SIZE + 6)
        self.setCursor(Qt.PointingHandCursor)
        self._state = LinkState.UNCONNECTED
        self._alpha = 1.0
        self._phase = 0.0
        self._theme = None
        detector.state_changed.connect(self._on_state)
        self.timer = QTimer(self)
        self.timer.setInterval(60)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    def set_theme(self, t):
        self._theme = t
        self.update()

    def _on_state(self, state, msg):
        self._state = state
        self.setToolTip(f"{state.value}：{msg}")
        self.update()

    def _tick(self):
        if self._state in (LinkState.TRANSMITTING, LinkState.FAILED):
            self._phase = (self._phase + 0.12) % (2 * 3.14159)
            self._alpha = 0.35 + 0.65 * (0.5 + 0.5 * __import__("math").sin(self._phase))
            self.update()
        elif self._alpha != 1.0:
            self._alpha = 1.0
            self.update()

    def _color(self):
        if self._theme is None:
            return QColor("#E8B93A")
        if self._state == LinkState.UNCONNECTED:
            return QColor(getattr(self._theme, "warning", "#E8B93A"))
        if self._state in (LinkState.CONNECTED, LinkState.TRANSMITTING):
            return QColor(getattr(self._theme, "output", "#4CAF7D"))
        return QColor(getattr(self._theme, "danger", "#E05A5A"))

    def paintEvent(self, ev):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing, True)
        c = self._color(); c.setAlphaF(max(0.2, self._alpha))
        cx, cy = self.width() / 2, self.height() / 2
        # 外圈光晕
        glow = QColor(c); glow.setAlphaF(c.alphaF() * 0.25)
        p.setBrush(QBrush(glow)); p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(cx, cy), self.SIZE / 2 + 3, self.SIZE / 2 + 3)
        # 实心圆点
        p.setBrush(QBrush(c))
        p.drawEllipse(QPointF(cx, cy), self.SIZE / 2, self.SIZE / 2)
        p.end()

    def mousePressEvent(self, ev):
        """单击不切态：状态只由真实心跳/注入信号驱动，避免『一点就绿』。"""
        if self.detector.is_real:
            self.setToolTip("真实串口模式：状态由开发板心跳决定")
        else:
            self.setToolTip("模拟模式：右键可注入/撤销信号（绿色仅由信号驱动）")

    def contextMenuEvent(self, ev):
        """右键菜单：仅模拟模式提供『注入信号』入口，用于验证四态。"""
        if self.detector.is_real:
            return
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        text = "撤销模拟信号" if self.detector._sim_has_signal else "注入模拟信号（显示绿色）"
        act = menu.addAction(text)
        act.triggered.connect(lambda: self.detector.inject_signal(
            not self.detector._sim_has_signal))
        menu.exec(ev.globalPos())


# ==================== 自绘标题栏（无边框窗口） ====================

class TitleBar(QFrame):
    """自绘标题栏：拖动 / 双击最大化 / 最小化 / 最大化 / 关闭。风格与 UI 统一。"""

    def __init__(self, window: QWidget, title: str):
        super().__init__(window)
        self.window_ref = window
        self.setObjectName("titlebar")
        self.setFixedHeight(38)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 0, 6, 0)
        lay.setSpacing(8)
        # 小标识点（品牌色）
        dot = QFrame(); dot.setObjectName("titleDot")
        dot.setFixedSize(8, 8)
        lay.addWidget(dot)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("titleText")
        lay.addWidget(self.title_label)
        lay.addStretch()
        # 三按钮：最小化 / 最大化 / 关闭
        self.btn_min = QPushButton("−"); self.btn_min.setObjectName("winBtn")
        self.btn_max = QPushButton("□"); self.btn_max.setObjectName("winBtn")
        self.btn_close = QPushButton("×"); self.btn_close.setObjectName("winBtnClose")
        for b in (self.btn_min, self.btn_max, self.btn_close):
            b.setFixedSize(30, 28)
        self.btn_min.clicked.connect(window.showMinimized)
        self.btn_max.clicked.connect(self._toggle_max)
        self.btn_close.clicked.connect(window.close)
        lay.addWidget(self.btn_min); lay.addWidget(self.btn_max); lay.addWidget(self.btn_close)
        self._drag_pos = None

    def set_title(self, title: str):
        self.title_label.setText(title)

    def _toggle_max(self):
        if self.window_ref.isMaximized():
            self.window_ref.showNormal()
            self.btn_max.setText("□")
        else:
            self.window_ref.showMaximized()
            self.btn_max.setText("❐")

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._drag_pos = ev.globalPosition().toPoint() - self.window_ref.pos()
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._drag_pos is not None and (ev.buttons() & Qt.LeftButton):
            self.window_ref.move(ev.globalPosition().toPoint() - self._drag_pos)
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        self._drag_pos = None
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        self._toggle_max()


class FramelessWindowHelper(QObject):
    """无边框窗口增强：八向边缘拖拽缩放 + 圆角遮罩。

    以 QApplication 级事件过滤器工作：判断事件目标是否属于已注册窗口的子控件树，
    再按鼠标相对窗口边缘的位置切换光标 / 执行缩放。不干扰标题栏拖动与按钮点击。
    """

    EDGE = 6           # 边缘感应宽度 px
    RADIUS = 20        # 圆角半径 px（DWM 不可用时 QRegion 降级用）
    MIN_W, MIN_H = 480, 360
    DWMWA_CORNER = 33  # DWMWA_WINDOW_CORNER_PREFERENCE
    DWMWCP_ROUND = 2   # 圆角（系统级抗锯齿）

    _instance = None
    _dwm = None         # ctypes windll.dwmapi，None=未初始化/不可用

    @classmethod
    def _init_dwm(cls):
        if cls._dwm is not None:
            return cls._dwm
        try:
            import ctypes
            from ctypes import wintypes
            dll = ctypes.windll.dwmapi
            dll.DwmSetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD,
                                                   ctypes.POINTER(wintypes.DWORD), wintypes.DWORD]
            dll.DwmSetWindowAttribute.restype = ctypes.c_long
            cls._dwm = dll
        except Exception:
            cls._dwm = False
        return cls._dwm

    @classmethod
    def _native_round(cls, window, on: bool):
        """用 Windows 11 DWM 设置系统级抗锯齿圆角；成功返回 True。"""
        dwm = cls._init_dwm()
        if not dwm:
            return False
        try:
            import ctypes
            from ctypes import wintypes
            pref = wintypes.DWORD(cls.DWMWCP_ROUND if on else 0)  # 0=DWMWCP_DEFAULT(方角)
            hwnd = wintypes.HWND(int(window.winId()))
            hr = dwm.DwmSetWindowAttribute(hwnd, cls.DWMWA_CORNER,
                                           ctypes.byref(pref), ctypes.sizeof(pref))
            return hr == 0
        except Exception:
            return False

    def __init__(self):
        super().__init__()
        self.windows = []           # [(window, edge_state)]
        self.native = set()         # 使用 DWM 原生圆角的窗口 id
        app = QApplication.instance()
        if app:
            app.installEventFilter(self)

    @classmethod
    def install(cls, window: QWidget):
        if cls._instance is None:
            cls._instance = cls()
        cls._instance.windows.append([window, None])
        window.setMouseTracking(True)
        # 圆角初始遮罩（等窗口显示后有尺寸再套用）
        QTimer.singleShot(0, lambda: cls._instance._apply_mask(window))

    def _owner(self, obj):
        """返回 obj 所属的已注册窗口（含其子控件），否则 None。自动剔除已销毁窗口。"""
        self.windows = [it for it in self.windows if shiboken_valid(it[0])]
        if not isinstance(obj, QWidget):
            return None
        for item in self.windows:
            w = item[0]
            if obj is w or w.isAncestorOf(obj):
                return item
        return None

    def _edge_at(self, window, pos):
        """鼠标窗口坐标 → 方向标识（左上右下组合），非边缘返回 None。"""
        if window.isMaximized() or window.isFullScreen():
            return None
        x, y = pos.x(), pos.y()
        w, h = window.width(), window.height()
        e = self.EDGE
        left, right = x <= e, x >= w - e
        top, bottom = y <= e, y >= h - e
        if top and left: return "tl"
        if top and right: return "tr"
        if bottom and left: return "bl"
        if bottom and right: return "br"
        if left: return "l"
        if right: return "r"
        if top: return "t"
        if bottom: return "b"
        return None

    @staticmethod
    def _cursor_of(edge):
        return {
            "l": Qt.SizeHorCursor, "r": Qt.SizeHorCursor,
            "t": Qt.SizeVertCursor, "b": Qt.SizeVertCursor,
            "tl": Qt.SizeFDiagCursor, "br": Qt.SizeFDiagCursor,
            "tr": Qt.SizeBDiagCursor, "bl": Qt.SizeBDiagCursor,
        }.get(edge)

    def _apply_mask(self, window):
        maximized = window.isMaximized() or window.isFullScreen()
        # 优先：Windows 11 DWM 系统级圆角（真抗锯齿，边缘圆滑）
        if self._native_round(window, on=not maximized):
            self.native.add(id(window))
            window.clearMask()          # 让出裁剪给 DWM，避免硬边遮罩盖掉抗锯齿
            if maximized:
                window.setCursor(Qt.ArrowCursor)
            return
        # 降级：QRegion 圆角遮罩（像素级，边缘略硬但可用）
        self.native.discard(id(window))
        if maximized:
            window.clearMask()
            window.setCursor(Qt.ArrowCursor)
            return
        r = self.RADIUS
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, window.width(), window.height()), r, r)
        window.setMask(QRegion(path.toFillPolygon().toPolygon()))

    def eventFilter(self, obj, ev):
        t = ev.type()
        # 窗口尺寸/状态变化 → 更新圆角遮罩（全局事件，无需 owner）
        if isinstance(obj, QWidget) and shiboken_valid(obj):
            for item in list(self.windows):
                w = item[0]
                if not shiboken_valid(w):
                    continue
                if obj is w and t in (QEvent.Resize, QEvent.WindowStateChange):
                    self._apply_mask(w)
                    break

        item = self._owner(obj)
        if item is None:
            return False
        window, edge_state = item

        # 标题栏区域不参与缩放（让标题栏自己处理拖动/双击/按钮）
        tb = getattr(window, "_titlebar", None)
        if tb is not None and isinstance(obj, QWidget) and (obj is tb or tb.isAncestorOf(obj)):
            return False

        try:
            gpos = ev.globalPosition().toPoint()
        except (AttributeError, RuntimeError):
            gpos = QPoint()
        local = window.mapFromGlobal(gpos)

        if t == QEvent.MouseMove:
            if edge_state:  # 正在拖拽缩放
                self._do_resize(window, edge_state, gpos)
                return True
            edge = self._edge_at(window, local)
            cur = self._cursor_of(edge)
            if cur and window.cursor().shape() != cur:
                window.setCursor(cur)
            elif not cur and window.cursor().shape() in (
                    Qt.SizeHorCursor, Qt.SizeVertCursor, Qt.SizeFDiagCursor, Qt.SizeBDiagCursor):
                window.unsetCursor()
            return False

        if t == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton:
            edge = self._edge_at(window, local)
            if edge:
                item[1] = {"edge": edge, "origin": gpos, "geom": window.geometry()}
                return True
            return False

        if t == QEvent.MouseButtonRelease and edge_state:
            item[1] = None
            window.unsetCursor()
            return False

        return False

    def _do_resize(self, window, state, gpos):
        edge = state["edge"]
        dx = gpos.x() - state["origin"].x()
        dy = gpos.y() - state["origin"].y()
        g = QRect(state["geom"])
        if "l" in edge:
            g.setLeft(min(g.left() + dx, g.right() - self.MIN_W))
        if "r" in edge:
            g.setRight(max(g.right() + dx, g.left() + self.MIN_W))
        if "t" in edge:
            g.setTop(min(g.top() + dy, g.bottom() - self.MIN_H))
        if "b" in edge:
            g.setBottom(max(g.bottom() + dy, g.top() + self.MIN_H))
        window.setGeometry(g)


def frameless(window: QWidget, title: str) -> TitleBar:
    """把窗口改为无边框并安装自绘标题栏。返回标题栏供更新文字。"""
    window.setWindowFlags(window.windowFlags() | Qt.FramelessWindowHint)
    bar = TitleBar(window, title)
    if isinstance(window, QMainWindow):
        old = window.takeCentralWidget()
        wrapper = QWidget(); wrapper.setObjectName("winWrapper")
        wl = QVBoxLayout(wrapper); wl.setContentsMargins(0, 0, 0, 0); wl.setSpacing(0)
        wl.addWidget(bar)
        if old:
            wl.addWidget(old)
        window.setCentralWidget(wrapper)
    else:  # QDialog：在现有 layout 顶部插入
        lay = window.layout()
        if lay is not None:
            lay.insertWidget(0, bar)
        else:
            # 无 layout 的对话框：包一层
            old = window.layout()
            wrapper = QWidget(); wl = QVBoxLayout(wrapper)
            wl.setContentsMargins(0, 0, 0, 0); wl.setSpacing(0)
            wl.addWidget(bar)
            window.setLayout(wl)
    window._titlebar = bar
    FramelessWindowHelper.install(window)
    return bar


# ==================== 新建档案对话框 ====================

class NewProfileDialog(QDialog):
    """新建档案：名称 / 结构 / 电缸 specs / 限位 / 备注。"""

    def __init__(self, parent, manager, existing_names):
        super().__init__(parent)
        self.setWindowTitle("新建平台档案")
        self.resize(560, 720)
        self.manager = manager
        self.existing_names = existing_names
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget(); scroll.setWidget(inner)
        lay = QVBoxLayout(inner); lay.setContentsMargins(26, 22, 26, 18); lay.setSpacing(10)
        root.addWidget(scroll)

        lay.addWidget(label("新建平台档案", "largeTitle"))
        lay.addWidget(label("填写基础信息，创建后可在参数页继续细化", "caption"))
        lay.addSpacing(10)

        lay.addWidget(label("档案名称", "sectionTitle"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("例如：六自由度平台A / 模拟器1号")
        lay.addWidget(self.name_edit)

        lay.addSpacing(6)
        lay.addWidget(label("机构结构类型", "sectionTitle"))
        self.structure_combo = QComboBox()
        for key, spec in STRUCTURES.items():
            self.structure_combo.addItem(spec.label, key)
        self.structure_combo.setCurrentText(STRUCTURES["SPU"].label)
        lay.addWidget(self.structure_combo)
        self.desc_label = label(STRUCTURES["SPU"].description, "caption", wrap=True)
        lay.addWidget(self.desc_label)
        self.structure_combo.currentIndexChanged.connect(self._on_structure)

        # ---- 电缸参数（自动分配到档案对应字段）----
        lay.addSpacing(8)
        act_card, ac = card("电缸规格（创建后自动填入参数页）", tint="input")
        self.act_edits = {}
        act_fields = [
            ("有效行程 (mm)", "stroke", "act_stroke"),
            ("最大线速度 (mm/s)", "peak_speed", "act_speed"),
            ("单缸额定推力 (N)", "max_force", "act_force"),
            ("缸径 (mm)", "actuator_bore", "act_bore"),
            ("负载质量 (kg)", "payload_mass", "act_payload"),
            ("上平台自重 (kg)", "platform_mass", "act_platform_mass"),
        ]
        ag = QGridLayout(); ag.setHorizontalSpacing(12); ag.setVerticalSpacing(9)
        for idx, (text, key, hint_key) in enumerate(act_fields):
            row, col = divmod(idx, 2)
            e = num_edit("0", 100)
            ag.addWidget(param_cell(text, hint_key, e), row, col)
            self.act_edits[key] = e
        self.zero_ref_combo = QComboBox()
        self.zero_ref_combo.addItem("零位 = 行程最低位", "lowest")
        self.zero_ref_combo.addItem("零位 = 行程中间位", "middle")
        ag.addWidget(param_cell("零位定义", "act_zero_ref", self.zero_ref_combo), 3, 1)
        ac.addLayout(ag)
        lay.addWidget(act_card)

        # ---- 限位保护 ----
        lay.addSpacing(6)
        lim_card, lc = card("限位保护（0 = 不限制）", tint="output")
        self.lim_edits = {}
        lim_fields = [
            ("行程下限 ΔL (mm)", "min_length", "stroke"),
            ("行程上限 ΔL (mm)", "max_length", "stroke"),
            ("最大平移 (mm)", "max_translation", "max_translation"),
            ("最大倾角 (°)", "max_tilt", "max_tilt"),
        ]
        lg = QGridLayout(); lg.setHorizontalSpacing(12); lg.setVerticalSpacing(9)
        for idx, (text, key, hk) in enumerate(lim_fields):
            row, col = divmod(idx, 2)
            e = num_edit("0", 100)
            lg.addWidget(param_cell(text, hk, e), row, col)
            self.lim_edits[key] = e
        lc.addLayout(lg)
        lc.addWidget(label("限位只在解算后给出警告，不会阻止计算，方便查看越界程度。", "caption", wrap=True))
        lay.addWidget(lim_card)

        lay.addSpacing(6)
        lay.addWidget(label("备注（可选）", "sectionTitle"))
        self.note_edit = QLineEdit()
        self.note_edit.setPlaceholderText("例如：直线缸 6-SPU / 底部虎克铰带竖直轴")
        lay.addWidget(self.note_edit)
        lay.addStretch()

        btns = QHBoxLayout(); btns.addStretch()
        b_cancel = QPushButton("取消"); b_cancel.clicked.connect(self.reject)
        b_ok = QPushButton("创建档案"); b_ok.setObjectName("primary")
        b_ok.clicked.connect(self._on_ok)
        btns.addWidget(b_cancel); btns.addWidget(b_ok)
        lay.addLayout(btns)
        frameless(self, "新建平台档案")

    def _on_structure(self):
        self.desc_label.setText(STRUCTURES[self.structure_combo.currentData()].description)

    def _on_ok(self):
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "请输入名称", "档案名称不能为空。"); return
        if name in self.existing_names or self.manager.exists(name):
            QMessageBox.warning(self, "名称已存在", "已有同名档案，请换一个名称。"); return
        try:
            self.result_specs = {k: float(e.text() or 0) for k, e in self.act_edits.items()}
            self.result_limits = {k: float(e.text() or 0) for k, e in self.lim_edits.items()}
            self.result_zero_ref = self.zero_ref_combo.currentData()
        except ValueError:
            QMessageBox.warning(self, "数值无效", "电缸规格和限位都需要填写数字（可填 0）。"); return
        self.result_name = name
        self.result_structure = self.structure_combo.currentData()
        self.result_note = self.note_edit.text().strip()
        self.accept()


# ==================== 启动器（档案中心） ====================

class Launcher(QMainWindow):
    def __init__(self, app_ref):
        super().__init__()
        self.app_ref = app_ref
        self.manager = app_ref.manager
        self.setWindowTitle(f"{APP_TITLE}  {APP_VERSION}")
        self.setWindowIcon(app_icon())
        self.resize(900, 640)
        self.setMinimumSize(720, 520)

        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        # 统一侧边栏（与工作窗口同一组件）
        self.rail = SideRail(self.app_ref.theme_mgr.theme, lock_state=self.app_ref.rail_lock_state)
        self.rail.section("档案中心")
        hint = label("管理你的平台配置\n与电缸规格", "caption", wrap=True)
        self.rail.lay.addWidget(hint); self.rail._hidden_on_collapse.append(hint)
        self.rail.lay.addSpacing(6)
        self.rail.add_nav("01   我的档案", checked=None, icon_kind="archive").setEnabled(False)
        self.rail.normalize_nav_alignment()
        self.rail.add_spacer()
        self.rail.add_lock_button()
        root.addWidget(self.rail)
        if not self.rail.locked:
            self.rail.set_collapsed(True)

        # 右侧内容
        content = QWidget(); root.addWidget(content)
        cl = QVBoxLayout(content); cl.setContentsMargins(0, 0, 0, 0); cl.setSpacing(0)
        top = QFrame(); top.setObjectName("topbar"); top.setFixedHeight(58)
        tl = QHBoxLayout(top); tl.setContentsMargins(30, 0, 30, 0)
        tl.addWidget(label("档案中心", "pageTitle")); tl.addStretch()
        b_settings = QPushButton("设置"); b_settings.setObjectName("ghost")
        b_settings.setToolTip("皮肤 / 玻璃质感 / 默认导出文件夹")
        b_settings.clicked.connect(self.open_settings)
        tl.addWidget(b_settings)
        tl.addWidget(label(f"版本 {APP_VERSION}", "pageSub"))
        cl.addWidget(top)

        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget(); scroll.setWidget(inner)
        il = QVBoxLayout(inner); il.setContentsMargins(36, 32, 36, 32); il.setSpacing(14)
        il.addWidget(label("我的平台档案", "largeTitle"))
        il.addWidget(label("选择已有档案继续编辑，或新建一个平台档案", "caption"))
        il.addSpacing(8)
        self.list_layout = QVBoxLayout(); self.list_layout.setSpacing(12)
        il.addLayout(self.list_layout)
        il.addSpacing(4)
        new_card, nl = card()
        row = QHBoxLayout(); row.setSpacing(10)
        # 两个入口按钮强制完全一致：同为 primary 样式（QSS 统一 padding/font），固定尺寸
        b_new = QPushButton("新建平台档案"); b_new.setObjectName("primary")
        b_new.setFixedSize(170, 44)
        b_new.clicked.connect(self.new_profile); row.addWidget(b_new)
        b_import = QPushButton("导入档案包"); b_import.setObjectName("primary")
        b_import.setFixedSize(170, 44)
        b_import.clicked.connect(self.import_profile)
        row.addWidget(b_import); row.addStretch()
        nl.addLayout(row)
        il.addWidget(new_card)
        il.addStretch()
        cl.addWidget(scroll)
        self.refresh()
        self.apply_theme(self.app_ref.theme_mgr.theme)
        frameless(self, f"{APP_TITLE}  {APP_VERSION} — 档案中心")

    def apply_theme(self, theme):
        if hasattr(self, "rail"):
            self.rail.set_theme(theme)

    def open_settings(self):
        dlg = SettingsDialog(self.app_ref, self)
        dlg.exec()

    def refresh(self):
        while self.list_layout.count():
            item = self.list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        profiles = self.manager.list()
        if not profiles:
            empty = QFrame(); empty.setObjectName("card")
            el = QVBoxLayout(empty); el.setContentsMargins(24, 34, 24, 34)
            el.setAlignment(Qt.AlignCenter)
            el.addWidget(label("还没有档案", "title"))
            el.addWidget(label("点击下方按钮创建你的第一个平台档案", "caption"))
            self.list_layout.addWidget(empty)
            return
        for p in profiles:
            self.list_layout.addWidget(self._profile_card(p))

    def _profile_card(self, p):
        frame = QFrame(); frame.setObjectName("card")
        lay = QHBoxLayout(frame); lay.setContentsMargins(20, 16, 16, 16)
        left = QVBoxLayout(); left.setSpacing(4)
        left.addWidget(label(p["name"], "headline"))
        spec = STRUCTURES.get(p["structure"], STRUCTURES["SPU"])
        left.addWidget(label(f"{spec.label}   ·   修改于 {p['modified']}", "caption"))
        if p["note"]:
            left.addWidget(label(p["note"], "caption"))
        lay.addLayout(left); lay.addStretch()
        b_open = QPushButton("打开"); b_open.clicked.connect(lambda: self.app_ref.open_profile(p["name"]))
        b_export = QPushButton("导出包"); b_export.clicked.connect(lambda: self.export_profile(p["name"]))
        b_del = QPushButton("删除"); b_del.setObjectName("danger")
        b_del.clicked.connect(lambda: self.delete_profile(p["name"]))
        lay.addWidget(b_open); lay.addWidget(b_export); lay.addWidget(b_del)
        return frame

    def import_profile(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "导入档案包", default_export_dir(), "档案包 (*.json);;所有文件 (*.*)")
        if not path:
            return
        name = self.manager.import_package(path)
        if name is None:
            QMessageBox.warning(self, "导入失败", "文件格式无效，请选择本软件导出的档案包（.json）。")
            return
        self.refresh()
        QMessageBox.information(self, "导入成功", f"档案「{name}」已导入。")

    def export_profile(self, name):
        path, _ = QFileDialog.getSaveFileName(
            self, "导出档案包", export_path(f"{name}_档案包.json"), "档案包 (*.json)")
        if not path:
            return
        ok = self.manager.export_package(name, path)
        QMessageBox.information(self, "导出完成" if ok else "导出失败",
                                f"档案「{name}」已导出到：\n{path}" if ok else "档案不存在。")

    def new_profile(self):
        names = [p["name"] for p in self.manager.list()]
        dlg = NewProfileDialog(self, self.manager, names)
        if dlg.exec():
            cfg = default_config()
            cfg.name = dlg.result_name
            cfg.structure = dlg.result_structure
            # 自动分配电缸规格与限位到档案对应字段
            for k, v in dlg.result_specs.items():
                setattr(cfg, k, v)
            for k, v in dlg.result_limits.items():
                setattr(cfg, k, v)
            cfg.zero_reference = dlg.result_zero_ref
            self.manager.save(dlg.result_name, cfg.to_dict(), dlg.result_note)
            self.refresh()
            self.app_ref.open_profile(dlg.result_name)

    def delete_profile(self, name):
        reply = QMessageBox.question(self, "删除档案",
                                     f"确定删除档案「{name}」？原文件会自动备份为 .bak。",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.manager.backup(name); self.manager.delete(name); self.refresh()


# ==================== 电缸管理窗口 ====================

class ActuatorDialog(QDialog):
    """独立电缸管理窗口：整体配置 + 各动作安全载荷计算。

    整体配置写回 config；有额定推力时按雅可比虚功原理计算六个动作
    方向的安全载荷，排版参考正解结果页的指标卡网格。
    """

    def __init__(self, main_window, parent=None):
        super().__init__(parent or main_window)
        self.mw = main_window
        self.cfg = main_window.config
        self.setWindowTitle("电缸管理与安全载荷")
        self.resize(1000, 720)
        self.setMinimumSize(860, 620)
        icon = Path(__file__).resolve().parent / "app.ico"
        if icon.exists():
            self.setWindowIcon(QIcon(str(icon)))

        root = QVBoxLayout(self)
        root.setContentsMargins(22, 18, 22, 18); root.setSpacing(12)
        head = QHBoxLayout()
        hb = QVBoxLayout(); hb.setSpacing(3)
        hb.addWidget(label("电缸管理与安全载荷", "largeTitle"))
        hb.addWidget(label("设置整机电缸规格；填写额定推力后自动计算六轴各动作的安全载荷", "caption"))
        head.addLayout(hb); head.addStretch()
        root.addLayout(head)

        body = QHBoxLayout(); body.setSpacing(16)

        # ---- 左：整体配置（可滚动，容纳电机学参数）----
        left_scroll = QScrollArea(); left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_host = QWidget(); left = QVBoxLayout(left_host)
        left.setContentsMargins(2, 2, 8, 2); left.setSpacing(12)
        glob, gl = card("整机电缸规格", tint="input")
        self.g_edits = {}
        gfields = [
            ("电缸名称", "actuator_name", "act_name", "str"),
            ("电缸编码 / 型号", "actuator_code", "act_code", "str"),
            ("电缸品牌", "actuator_brand", "act_brand", "str"),
            ("有效行程 (mm)", "stroke", "act_stroke", "float"),
            ("额定推力 (N)", "max_force", "act_force", "float"),
            ("最大线速度 (mm/s)", "peak_speed", "act_speed", "float"),
            ("缸径 (mm)", "actuator_bore", "act_bore", "float"),
            ("负载质量 (kg)", "payload_mass", "act_payload", "float"),
            ("平台自重 (kg)", "platform_mass", "act_platform_mass", "float"),
        ]
        gg = QGridLayout(); gg.setHorizontalSpacing(12); gg.setVerticalSpacing(6)
        for idx, (text, key, hk, kind) in enumerate(gfields):
            row, col = idx, 0   # 单列布局，适配窄滚动栏
            if kind == "str":
                val = (self.cfg.actuator_names[0] if self.cfg.actuator_names else "")
                if key == "actuator_code":
                    val = self.cfg.actuator_codes[0] if self.cfg.actuator_codes else ""
                if key == "actuator_brand":
                    val = self.cfg.actuator_brands[0] if self.cfg.actuator_brands else ""
                e = QLineEdit(str(val)); e.setFixedWidth(120)
            else:
                e = num_edit(getattr(self.cfg, key, 0) if hasattr(self.cfg, key) else 0, 100)
            self.g_edits[key] = e
            gg.addWidget(param_cell(text, hk, e), row, col)
        gl.addLayout(gg)

        # 零位定义
        zr = QHBoxLayout(); zr.addWidget(label("零位定义"))
        zr.addWidget(make_hint_button("act_zero_ref"))
        self.zero_combo = QComboBox()
        self.zero_combo.addItem("零位 = 行程最低位", "lowest")
        self.zero_combo.addItem("零位 = 行程中间位", "middle")
        self.zero_combo.setCurrentIndex(0 if self.cfg.zero_reference == "lowest" else 1)
        zr.addWidget(self.zero_combo); zr.addStretch(); gl.addLayout(zr)

        b_calc = QPushButton("应用并计算安全载荷  →"); b_calc.setObjectName("primary")
        b_calc.clicked.connect(self._save_and_calc); gl.addWidget(b_calc)
        self.g_status = label("修改规格后点上方按钮计算。", "caption", wrap=True); gl.addWidget(self.g_status)
        left.addWidget(glob)

        # ---- 电机学参数（15 项）：机械 / 电气 / 派生 ----
        self._build_motor_params(left)

        left.addStretch()
        left_scroll.setWidget(left_host)
        body.addWidget(left_scroll, stretch=4)

        # ---- 右：安全载荷结果（排版同正解结果页） ----
        right = QVBoxLayout(); right.setSpacing(12)
        out, ol = card("六轴各动作安全载荷", tint="output")
        ol.addWidget(label("额定推力约束下，各动作方向可安全承载的负载质量（kg）", "caption", wrap=True))
        load_grid = QGridLayout(); load_grid.setSpacing(9)
        self.safe_labels = []
        load_defs = [
            ("X 纵荡", "out_x"), ("Y 横荡", "out_y"), ("Z 升沉", "out_z"),
            ("Roll 横滚", "out_roll"), ("Pitch 俯仰", "out_pitch"), ("Yaw 偏航", "out_yaw"),
        ]
        for idx, (nm, hk) in enumerate(load_defs):
            mc, mv = metric_card(nm, hk, "kg", note="安全载荷 kg")
            load_grid.addWidget(mc, idx // 2, idx % 2)
            self.safe_labels.append(mv)
        ol.addLayout(load_grid)
        ol.addWidget(label("说明：按虚功原理 F=pinv(Jᵀ)·w，取最先达到额定推力的缸为上限；"
                           "转动方向按平台等效半径折算为边缘质量。", "caption", wrap=True))
        right.addWidget(out); right.addStretch()

        body.addLayout(right, stretch=6)
        root.addLayout(body)

        # 底部动作
        blo = QHBoxLayout(); blo.addStretch()
        b_cancel = QPushButton("关闭"); b_cancel.clicked.connect(self.reject)
        blo.addWidget(b_cancel)
        root.addLayout(blo)
        # 初次打开先算一次
        self._update_safe_labels()
        # 连接本窗口所有问号按钮（独立对话框不会被主窗口的统一连接覆盖）
        theme = self.mw.app_ref.theme_mgr.theme
        for btn in self.findChildren(QToolButton):
            if btn.objectName() == "hint" and hasattr(btn, "_hint_key"):
                key = btn._hint_key
                btn.clicked.connect(
                    lambda checked=False, k=key: HintDialog(
                        k, self.cfg, theme, self).exec())
        frameless(self, "电缸管理")

    def _build_motor_params(self, parent_layout):
        """电机学参数 15 项：机械/电气/派生三组，派生项自动回填。"""
        self.motor_edits = {}
        self._auto_buttons = {}
        groups = [
            ("mechanical", "机械规格（导程/推力/速度）", "input"),
            ("electrical", "电气参数（电压/电流/转矩常数）", "view"),
            ("derived", "派生计算（公式自动填入）", "output"),
        ]
        for gkey, gtitle, tint in groups:
            gcard, gcl = card(gtitle, tint=tint)
            grid = QGridLayout(); grid.setHorizontalSpacing(10); grid.setVerticalSpacing(7)
            members = [f for f in FIELDS if f.group == gkey]
            for idx, f in enumerate(members):
                row, col = idx, 0   # 单列布局：左侧窄栏也能完整显示标签+问号+输入框
                e = num_edit(getattr(self.cfg, f.key, 0), 92)
                e.setObjectName("calcInput" if f.derived else "")
                e._field_key = f.key
                e._dirty = False
                self.motor_edits[f.key] = e
                cell = param_cell(f"{f.label} ({f.unit})", f"act_{f.key}", e)
                if f.derived:
                    auto = QPushButton("↺"); auto.setObjectName("ghost")
                    auto.setFixedWidth(26); auto.setToolTip("恢复自动计算")
                    auto.clicked.connect(lambda checked=False, k=f.key: self._reset_auto(k))
                    self._auto_buttons[f.key] = auto
                    row_widget = QWidget(); rw = QHBoxLayout(row_widget)
                    rw.setContentsMargins(0, 0, 0, 0); rw.setSpacing(2)
                    rw.addWidget(cell); rw.addWidget(auto)
                    grid.addWidget(row_widget, row, col)
                else:
                    grid.addWidget(cell, row, col)
                e.textChanged.connect(self._on_motor_changed)
            gcl.addLayout(grid)
            parent_layout.addWidget(gcard)

        # 初次按公式填派生值（用现有根字段值）
        self._recompute_derived()

    def _collect_motor_ctx(self):
        ctx = {}
        for f in FIELDS:
            if f.key in self.motor_edits:
                ctx[f.key] = self.motor_edits[f.key].text()
        return ctx

    def _recompute_derived(self):
        ctx = self._collect_motor_ctx()
        results = recompute(ctx)
        for key, (val, warn) in results.items():
            e = self.motor_edits.get(key)
            if e is None or getattr(e, "_dirty", False):
                continue
            with QSignalBlocker(e):
                e.setText("" if val is None else f"{val:.6g}")
            btn = self._auto_buttons.get(key)
            if btn:
                btn.setToolTip("恢复自动计算" + (f"（{warn}）" if warn else ""))

    def _on_motor_changed(self):
        s = self.sender()
        if isinstance(s, QLineEdit) and hasattr(s, "_field_key"):
            s._dirty = True     # 用户手改：根字段标记来源，派生字段锁定
        self._recompute_derived()

    def _reset_auto(self, key):
        e = self.motor_edits.get(key)
        if e:
            e._dirty = False
        self._recompute_derived()

    def _update_safe_labels(self):
        try:
            solver = StewartKinematics(self.cfg)
            safe = solver.safe_payloads()
            for i, lab in enumerate(self.safe_labels):
                lab.setText(f"{safe[i]:.1f}")
        except Exception:
            for lab in self.safe_labels:
                lab.setText("—")

    def apply_theme(self, theme):
        self.setStyleSheet(build_style(theme))

    def _save_and_calc(self):
        try:
            for key, w in self.g_edits.items():
                if key == "actuator_name":
                    self.cfg.actuator_names = [w.text().strip()]
                elif key == "actuator_code":
                    self.cfg.actuator_codes = [w.text().strip()]
                elif key == "actuator_brand":
                    self.cfg.actuator_brands = [w.text().strip()]
                else:
                    setattr(self.cfg, key, float(w.text() or 0))
            self.cfg.zero_reference = self.zero_combo.currentData()
            # 电机学 15 字段写入
            for f in FIELDS:
                e = self.motor_edits.get(f.key)
                if e is None:
                    continue
                try:
                    setattr(self.cfg, f.key, float(e.text() or 0))
                except ValueError:
                    setattr(self.cfg, f.key, 0.0)
            # 派生值兜底重算（若用户手改了根字段但未触发回填）
            ctx = self._collect_motor_ctx()
            for key, (val, _) in recompute(ctx).items():
                if val is not None and not self.motor_edits[key]._dirty:
                    setattr(self.cfg, key, float(val))
            # 别名同步：新字段为权威，旧求解器字段跟随
            self.cfg.max_force = self.cfg.rated_force
            self.cfg.max_speed = self.cfg.peak_speed or self.cfg.rated_speed
            # 整体电缸：清空逐缸覆盖值，避免冲突
            self.cfg.leg_strokes = None; self.cfg.leg_forces = None; self.cfg.leg_speeds = None

            # 同步到主窗口
            self.mw._load_config_into_form()
            self.mw.apply_config(silent=True)
            self.mw.auto_save()
            self._update_safe_labels()
            self.g_status.setText("✓ 已保存并同步；安全载荷已按最新规格更新。")
        except Exception as ex:
            self.g_status.setText(f"✗ 失败：{ex}")
            QMessageBox.warning(self, "保存失败", str(ex))


# ==================== 设置窗口 ====================

def read_settings() -> dict:
    from theme import SETTINGS_FILE
    try:
        if SETTINGS_FILE.exists():
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def write_settings(data: dict):
    from theme import SETTINGS_FILE
    try:
        SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def default_export_dir() -> str:
    """所有导出/导入共用的默认文件夹；未设置时为空（用系统默认）。"""
    return read_settings().get("default_export_dir", "")


def export_path(filename: str) -> str:
    """把文件名放到默认导出目录下；未设置时返回原文件名。"""
    d = default_export_dir()
    return str(Path(d) / filename) if d else filename


class SettingsDialog(QDialog):
    """设置：皮肤(深/浅)、玻璃质感模式、默认导出文件夹、计算结果存档。"""

    def __init__(self, app_ref, parent=None):
        super().__init__(parent or app_ref.main or app_ref.launcher)
        self.app_ref = app_ref
        self.setWindowTitle("设置")
        self.resize(600, 780)
        self.setMinimumSize(520, 700)
        icon = Path(__file__).resolve().parent / "app.ico"
        if icon.exists():
            self.setWindowIcon(QIcon(str(icon)))
        self._draft = read_settings()

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20); root.setSpacing(14)
        root.addWidget(label("设置", "largeTitle"))
        root.addWidget(label("调整界面外观与导出行为，保存后立即生效", "caption"))

        # ---- 皮肤 ----
        skin, sl = card("界面皮肤", tint="param")
        self.skin_group = QButtonGroup(); self.skin_group.setExclusive(True)
        skin_row = QHBoxLayout(); skin_row.setSpacing(10)
        for key, name in (("light", "浅色 · 印纸"), ("dark", "深色 · 暗场")):
            b = QPushButton(name); b.setCheckable(True)
            b.setChecked(self.app_ref.theme_mgr.key == key)
            b.setObjectName("segment"); b.setMinimumHeight(40)
            self.skin_group.addButton(b, 0 if key == "light" else 1)
            b._skin_key = key
            skin_row.addWidget(b)
        sl.addLayout(skin_row)
        sl.addWidget(label("浅色为近白纸感，深色为近黑暗场感。", "caption", wrap=True))
        root.addWidget(skin)

        # ---- 玻璃质感 ----
        glass, gl = card("玻璃质感", tint="view")
        from theme import GLASS_MODES, GLASS_MODE_NAMES
        self.glass_group = QButtonGroup(); self.glass_group.setExclusive(True)
        gg = QVBoxLayout(); gg.setSpacing(6)
        for i, mode in enumerate(GLASS_MODES):
            rb = QRadioButton(GLASS_MODE_NAMES[mode])
            self.glass_group.addButton(rb, i); rb._glass_key = mode
            gg.addWidget(rb)
        for b in self.glass_group.buttons():
            b.setChecked(b._glass_key == self.app_ref.theme_mgr.glass_mode)
        gl.addLayout(gg)
        gl.addWidget(label("推荐=明显玻璃+柔光；强=更通透；极简=近纯色无阴影。"
                           "注：PySide6 无法做网页的实时背景折射，以半透明层近似。", "caption", wrap=True))
        root.addWidget(glass)

        # ---- 默认导出文件夹 ----
        exp, el = card("默认导出文件夹", tint="input")
        el.addWidget(label("所有导出（档案包 / 结果 CSV / 全部数据）默认打开此文件夹：",
                           "caption", wrap=True))
        row = QHBoxLayout(); row.setSpacing(8)
        self.dir_edit = QLineEdit(self._draft.get("default_export_dir", ""))
        self.dir_edit.setReadOnly(True); self.dir_edit.setPlaceholderText("未设置（使用系统默认位置）")
        b_pick = QPushButton("选择文件夹…"); b_pick.clicked.connect(self._pick_dir)
        b_clear = QPushButton("清除"); b_clear.setObjectName("ghost")
        b_clear.clicked.connect(lambda: self.dir_edit.setText(""))
        row.addWidget(self.dir_edit); row.addWidget(b_pick); row.addWidget(b_clear)
        el.addLayout(row)
        root.addWidget(exp)

        # ---- 调试模式（前置校验：6 项电机核心必须填且 >0）----
        dbg, dl = card("调试模式", tint="output")
        self.debug_chk = QCheckBox("开启调试模式（显示电机圈数/转速等派生数据）")
        self.debug_chk.setChecked(bool(self._draft.get("debug_mode", False)))
        self.debug_chk.toggled.connect(self._on_debug_toggled)
        dl.addWidget(self.debug_chk)
        self.debug_warn = QLabel(""); self.debug_warn.setWordWrap(True)
        dl.addWidget(self.debug_warn)
        # 开发板驱动方式：串口 / EtherCAT（真正切换 IO worker）
        drv_row = QHBoxLayout(); drv_row.setSpacing(8)
        drv_row.addWidget(label("驱动方式：", "caption"))
        self.driver_combo = QComboBox()
        self.driver_combo.addItem("串口（COM + 心跳）", "serial")
        self.driver_combo.addItem("EtherCAT（pysoem + CiA 402）", "ethercat")
        idx = self.driver_combo.findData(self._draft.get("device_driver", "serial"))
        self.driver_combo.setCurrentIndex(max(idx, 0))
        self.driver_combo.currentIndexChanged.connect(self._on_driver_changed)
        drv_row.addWidget(self.driver_combo); drv_row.addStretch()
        dl.addLayout(drv_row)
        # 开发板串口配置（留空 = 模拟模式；默认心跳 COM + 超时2秒）
        port_row = QHBoxLayout(); port_row.setSpacing(8)
        port_row.addWidget(label("开发板串口：", "caption"))
        self.port_edit = QLineEdit(self._draft.get("device_port", ""))
        self.port_edit.setPlaceholderText("如 COM3 或 /dev/ttyUSB0（留空=模拟模式）")
        port_row.addWidget(self.port_edit)
        self.baud_edit = QLineEdit(str(self._draft.get("device_baud", 115200)))
        self.baud_edit.setFixedWidth(80)
        port_row.addWidget(label("波特率", "caption")); port_row.addWidget(self.baud_edit)
        dl.addLayout(port_row)
        # EtherCAT 网卡序号（多网卡时按 pysoem 枚举顺序选择）
        nic_row = QHBoxLayout(); nic_row.setSpacing(8)
        nic_row.addWidget(label("EtherCAT 网卡序号：", "caption"))
        self.nic_edit = QLineEdit(str(self._draft.get("device_nic_index", 0)))
        self.nic_edit.setFixedWidth(60)
        nic_row.addWidget(self.nic_edit)
        nic_row.addStretch(); dl.addLayout(nic_row)
        self.driver_hint = label("", "caption", wrap=True); dl.addWidget(self.driver_hint)
        # EtherCAT 伺服对象字典配置（默认 CiA 402；不同伺服型号在此改，无需动代码）
        od = read_settings().get("ec_od", {})
        od_row1 = QHBoxLayout(); od_row1.setSpacing(6)
        od_row1.addWidget(label("控制字 6040：", "caption"))
        self.ec_control = QLineEdit(str(od.get("controlword", "6040"))); self.ec_control.setFixedWidth(64)
        od_row1.addWidget(self.ec_control)
        od_row1.addWidget(label("目标位置 607A：", "caption"))
        self.ec_target = QLineEdit(str(od.get("target_pos", "607A"))); self.ec_target.setFixedWidth(64)
        od_row1.addWidget(self.ec_target)
        od_row1.addWidget(label("状态字：", "caption"))
        self.ec_status = QLineEdit(str(od.get("statusword", "6041"))); self.ec_status.setFixedWidth(64)
        od_row1.addWidget(self.ec_status); od_row1.addStretch()
        dl.addLayout(od_row1)
        od_row2 = QHBoxLayout(); od_row2.setSpacing(6)
        od_row2.addWidget(label("使能值：", "caption"))
        self.ec_enable = QLineEdit(str(od.get("enable_word", "000F"))); self.ec_enable.setFixedWidth(58)
        od_row2.addWidget(self.ec_enable)
        od_row2.addWidget(label("单位系数：", "caption"))
        self.ec_scale = QLineEdit(str(od.get("pos_scale", 1.0))); self.ec_scale.setFixedWidth(58)
        od_row2.addWidget(self.ec_scale)
        od_row2.addWidget(label("字节数：", "caption"))
        self.ec_bytes = QComboBox(); self.ec_bytes.addItems(["1", "2", "4"])
        self.ec_bytes.setCurrentIndex(self.ec_bytes.findText(str(od.get("pos_bytes", 4))))
        od_row2.addWidget(self.ec_bytes)
        od_row2.addWidget(label("字节序：", "caption"))
        self.ec_order = QComboBox(); self.ec_order.addItems(["little", "big"])
        self.ec_order.setCurrentIndex(0 if str(od.get("byte_order", "little")) == "little" else 1)
        od_row2.addWidget(self.ec_order); od_row2.addStretch()
        dl.addLayout(od_row2)
        self.ec_od_hint = label("对象字典地址填十六进制（如 6040）。不同伺服型号按需改；"
                                "单位系数=目标位置单位/缸长 mm（默认 1）。", "caption", wrap=True)
        dl.addWidget(self.ec_od_hint)
        self._on_driver_changed()   # 初始化启用状态与提示
        dl.addWidget(label("心跳：默认 COM 查询帧 / EtherCAT 状态字，2 秒无应答视为未连接；"
                           "右下角光点显示连接状态。", "caption", wrap=True))
        hint_b = QPushButton("去电缸管理补全数据 →"); hint_b.setObjectName("ghost")
        hint_b.clicked.connect(self._open_actuator_for_debug)
        dl.addWidget(hint_b)
        root.addWidget(dbg)
        self._refresh_debug_warn()

        root.addStretch()
        blo = QHBoxLayout(); blo.addStretch()
        b_cancel = QPushButton("取消"); b_cancel.clicked.connect(self.reject)
        b_save = QPushButton("保存"); b_save.setObjectName("primary")
        b_save.clicked.connect(self._save)
        blo.addWidget(b_cancel); blo.addWidget(b_save)
        root.addLayout(blo)
        self.apply_theme(self.app_ref.theme_mgr.theme)
        frameless(self, "设置")

    def _on_driver_changed(self):
        """切换驱动方式时更新串口/EtherCAT 输入的可用状态与提示。"""
        if not hasattr(self, "driver_combo"):
            return
        ec = self.driver_combo.currentData() == "ethercat"
        self.port_edit.setEnabled(not ec)
        self.baud_edit.setEnabled(not ec)
        self.nic_edit.setEnabled(ec)
        for w in (self.ec_control, self.ec_target, self.ec_status,
                  self.ec_enable, self.ec_scale, self.ec_bytes, self.ec_order):
            w.setEnabled(ec)
        self.ec_od_hint.setVisible(ec)
        if ec:
            try:
                import pysoem  # noqa: F401
                self.driver_hint.setText(
                    "EtherCAT：需安装 pysoem 与 Npcap，并把 6 路缸长写到从站 "
                    "6040/607A（CiA 402）。未装库或总线上无从站时自动退回模拟模式。")
            except ImportError:
                self.driver_hint.setText(
                    "⚠ 未安装 pysoem。当前选择 EtherCAT 但无法连接真实硬件，"
                    "将按模拟模式运行；需要时 pip install pysoem 并另装 Npcap。")
        else:
            self.driver_hint.setText("串口：填写串口号与波特率；留空即模拟模式。")

    def _pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择默认导出文件夹", self.dir_edit.text())
        if d:
            self.dir_edit.setText(d)

    def apply_theme(self, theme):
        self.setStyleSheet(build_style(theme))
        self._refresh_debug_warn()

    def _current_cfg(self):
        return getattr(self.app_ref.main, "config", None)

    def _refresh_debug_warn(self):
        cfg = self._current_cfg()
        theme = self.app_ref.theme_mgr.theme
        if cfg is None:
            self.debug_warn.setText("（打开档案后可用）")
            return
        ok, missing, warnings = validate_debug_prerequisites(cfg)
        if ok:
            txt = "✓ 6 项核心数据已齐，可开启调试模式。"
            if warnings:
                txt += " 注意：" + "；".join(warnings[:4])
            self.debug_warn.setStyleSheet(f"color:{theme.output};")
        else:
            txt = "⚠ 必要数据未填或≤0：" + "、".join(missing) + "。请在电缸管理中补全。"
            if warnings:
                txt += " 非必要项：" + "；".join(warnings[:4])
            self.debug_warn.setStyleSheet(f"color:{theme.warning};")
        self.debug_warn.setText(txt)

    def _on_debug_toggled(self, checked):
        if not checked:
            self._draft["debug_mode"] = False
            self._refresh_debug_warn()
            return
        cfg = self._current_cfg()
        ok, missing, _ = validate_debug_prerequisites(cfg) if cfg else (False, ["档案未打开"], [])
        if not ok:
            with QSignalBlocker(self.debug_chk):
                self.debug_chk.setChecked(False)
            self._draft["debug_mode"] = False
            self._refresh_debug_warn()
            QMessageBox.warning(self, "无法开启调试模式",
                                "必要数据未填或≤0：" + "、".join(missing) + "。\n请先在电缸管理中补全。")
        else:
            self._draft["debug_mode"] = True
            self._refresh_debug_warn()

    def _open_actuator_for_debug(self):
        if self.app_ref.main:
            self.accept()   # 先关闭设置窗口，再切到内置电缸页
            self.app_ref.main.open_actuator_dialog()

    def _save(self):
        try:
            skin_btn = self.skin_group.checkedButton()
            glass_btn = self.glass_group.checkedButton()
            if skin_btn:
                self._draft["theme"] = skin_btn._skin_key
                self.app_ref.theme_mgr.set(skin_btn._skin_key)
            if glass_btn:
                self._draft["glass_mode"] = glass_btn._glass_key
                self.app_ref.theme_mgr.set_glass_mode(glass_btn._glass_key)
            self._draft["default_export_dir"] = self.dir_edit.text().strip()
            self._draft["device_driver"] = self.driver_combo.currentData()
            self._draft["device_port"] = self.port_edit.text().strip()
            try:
                self._draft["device_baud"] = int(self.baud_edit.text().strip() or 115200)
            except ValueError:
                self._draft["device_baud"] = 115200
            try:
                self._draft["device_nic_index"] = max(0, int(self.nic_edit.text().strip() or 0))
            except ValueError:
                self._draft["device_nic_index"] = 0
            self._draft["ec_od"] = {
                "controlword": self.ec_control.text().strip() or "6040",
                "target_pos": self.ec_target.text().strip() or "607A",
                "statusword": self.ec_status.text().strip() or "6041",
                "enable_word": self.ec_enable.text().strip() or "000F",
                "pos_scale": self.ec_scale.text().strip() or "1.0",
                "pos_bytes": int(self.ec_bytes.currentText()),
                "byte_order": self.ec_order.currentText(),
            }
            # 保存时二次校验调试模式，防止打开设置后数据被改
            if self.debug_chk.isChecked():
                cfg = self._current_cfg()
                ok, missing, _ = validate_debug_prerequisites(cfg) if cfg else (False, ["档案未打开"], [])
                if not ok:
                    self._draft["debug_mode"] = False
                    QMessageBox.warning(self, "调试模式未启用",
                                        "必要数据不足：" + "、".join(missing) + "。")
                else:
                    self._draft["debug_mode"] = True
            else:
                self._draft["debug_mode"] = False
            if self.app_ref.main:
                self.app_ref.main.config.debug_mode = bool(self._draft["debug_mode"])
            write_settings(self._draft)
            QMessageBox.information(self, "已保存", "设置已保存并生效。")
            self.accept()
        except Exception as ex:
            QMessageBox.warning(self, "保存失败", str(ex))


# ==================== 主工作窗口 ====================

class MainWindow(QMainWindow):
    def __init__(self, app_ref, profile_name: str):
        super().__init__()
        self.app_ref = app_ref
        self.manager = app_ref.manager
        self.profile_name = profile_name
        self.note = ""
        self.solver = None
        self.last_fk = None
        self.last_ik = None
        self._prev_ik_pose = None   # 逆解转速用：上一帧目标位姿
        self._prev_ik_velocity = None  # 逆解加速度/推力用：上一帧广义速度
        self.saved_inputs = []
        self.viewports = []

        data = self.manager.load(profile_name)
        if data is None:
            self.config = default_config(); self.config.name = profile_name
        else:
            self.note = data.get("note", "")
            cfg_dict = dict(data.get("config", {}))
            if "pair_centers" in cfg_dict:
                cfg_dict["pair_centers"] = tuple(cfg_dict["pair_centers"])
            self.config = PlatformConfig.from_dict(cfg_dict)
            self.config.name = profile_name
            self.saved_inputs = data.get("fk_inputs", [])
        # 两种输入模式的数据（兼容旧版 list 格式）
        self._saved_mode = 0
        self.fk_input_cache = [[0.0] * 6 for _ in range(3)]
        if isinstance(self.saved_inputs, dict):
            self._saved_mode = int(self.saved_inputs.get("mode", 0))
            for ri, k in enumerate(("absolute", "relative", "revolutions")):
                v = self.saved_inputs.get(k, [0.0] * 6)   # 旧档案无 revolutions → 默认 0
                self.fk_input_cache[ri] = list(v) if isinstance(v, (list, tuple)) and len(v) == 6 else [0.0] * 6
            if self._saved_mode > 2:
                self._saved_mode = 0

        self.setWindowTitle(f"{APP_TITLE} — {profile_name}")
        self.setWindowIcon(app_icon())
        self.resize(1320, 880)
        self.setMinimumSize(1060, 720)
        self._build_ui()
        self._load_config_into_form()
        self.apply_config(silent=True)
        self._sync_structure_ui()
        self._connect_all_hints()   # 修复：问号按钮点击无响应

        if isinstance(self.saved_inputs, dict):
            # 新格式：恢复模式与该模式的数据（屏蔽信号避免覆盖缓存）
            self._fk_mode = self._saved_mode
            self.fk_input_mode.blockSignals(True)
            self.fk_input_mode.setCurrentIndex(self._fk_mode)
            self.fk_input_mode.blockSignals(False)
            for e, v in zip(self.fk_edits, self.fk_input_cache[self._fk_mode]):
                e.setText(str(v))
        elif len(self.saved_inputs) == 6:
            # 旧格式：当作绝对模式
            self.fk_input_cache[0] = list(self.saved_inputs)
            for e, v in zip(self.fk_edits, self.saved_inputs):
                e.setText(str(v))

        self.apply_theme(self.app_ref.theme_mgr.theme)
        frameless(self, f"{APP_TITLE} — {profile_name}")
        self._setup_link_indicator()
        self._set_debug_visible(read_settings().get("debug_mode", False))
        self._refresh_svd_panel()

    def _setup_link_indicator(self):
        """开发板连接光点：右下角悬浮，调试模式开启时显示。"""
        st = read_settings()
        self.link_detector = DeviceLinkDetector(
            st.get("device_port", ""), int(st.get("device_baud", 115200)), self,
            driver=st.get("device_driver", "serial"),
            nic_index=int(st.get("device_nic_index", 0) or 0),
            od_cfg=st.get("ec_od", {}))
        self.link_indicator = LinkIndicator(self.link_detector, self)
        self.link_indicator.set_theme(self.app_ref.theme_mgr.theme)
        self.link_detector.start()
        self._place_link_indicator()

    def _place_link_indicator(self):
        ind = getattr(self, "link_indicator", None)
        if ind:
            ind.move(self.width() - ind.width() - 18, self.height() - ind.height() - 16)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._place_link_indicator()

    # ---------- 整体布局 ----------

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        # ---------- 统一侧边栏（与档案中心同一组件） ----------
        self.rail = SideRail(self.app_ref.theme_mgr.theme, lock_state=self.app_ref.rail_lock_state)
        self.rail.section("工作台")
        # 电缸管理作为 01 页加入导航组，与其他栏统一由 group 管理高亮
        nav_defs = [
            ("01", "电缸管理", "规格 / 电机参数 / 安全载荷"),
            ("02", "平台参数", "填几何 / 选结构"), ("03", "姿态正解", "缸长 → 姿态"),
            ("04", "逆解验证", "姿态 → 缸长"), ("05", "三维姿态", "可视化模型"),
            ("06", "运动学计算", "速度 / 加速度"),
            ("07", "批量计算", "多行数据"), ("08", "轨迹播放", "CSV 轨迹 / 逐帧下发"),
            ("09", "使用说明", "帮助与约定")]
        pages = [(num, name, sub) for num, name, sub in nav_defs]
        self.nav_group = self.rail.add_nav_group(
            [f"  {num}   {name}" for num, name, _ in pages])
        self.b_actuators = self.nav_group.button(0)   # 01 电缸管理按钮
        self.nav_group.idClicked.connect(self._on_nav)
        self.rail.normalize_nav_alignment()   # 保证所有导航文字对齐

        self.rail.add_spacer()
        self.side_struct = self.rail.info_block(
            "当前档案 ACTIVE", self.profile_name, "结构 —")
        b_back = self.rail.add_nav("←  档案中心", checked=None, icon_kind="back")
        b_back.clicked.connect(self.back_to_launcher)
        self.rail.normalize_nav_alignment()
        self.rail.add_lock_button()
        root.addWidget(self.rail)
        if not self.rail.locked:
            self.rail.set_collapsed(True)

        # ---------- 内容区（自带印象派大气背景）----------
        from atmos import FluidPage
        content = FluidPage(self.app_ref.theme_mgr.theme)
        self.content = content
        root.addWidget(content)
        cl = QVBoxLayout(content); cl.setContentsMargins(0, 0, 0, 0); cl.setSpacing(0)

        top = QFrame(); top.setObjectName("topbar"); top.setFixedHeight(66)
        tl = QHBoxLayout(top); tl.setContentsMargins(28, 0, 24, 0); tl.setSpacing(14)
        title_box = QVBoxLayout(); title_box.setSpacing(2)
        self.page_title = label("平台参数", "pageTitle")
        self.page_sub = label("填几何 / 选结构", "pageSub")
        title_box.addWidget(self.page_title); title_box.addWidget(self.page_sub)
        tl.addLayout(title_box); tl.addStretch()
        self.structure_badge = label("SPU", "badge")
        self.structure_badge.setFixedHeight(40)   # 与 primary 按钮同高，居中对齐
        tl.addWidget(self.structure_badge, alignment=Qt.AlignVCenter)
        self.debug_badge = label("调试模式 ACTIVE", "debugBadge")
        self.debug_badge.setFixedHeight(40)
        self.debug_badge.setToolTip("调试模式已开启：显示电机派生数据")
        tl.addWidget(self.debug_badge, alignment=Qt.AlignVCenter)
        b_export = QPushButton("导出全部数据"); b_export.setObjectName("primary")
        b_export.setToolTip("导出档案参数、电缸规格、限位与全部计算结果")
        b_export.clicked.connect(self.export_all); tl.addWidget(b_export)
        b_note = QPushButton("备注"); b_note.setObjectName("ghost")
        b_note.setToolTip("编辑档案备注"); b_note.clicked.connect(self.edit_note); tl.addWidget(b_note)
        b_repo = QPushButton("GitHub"); b_repo.setObjectName("ghost")
        b_repo.setToolTip(f"在浏览器打开项目仓库：{REPO_URL}")
        b_repo.clicked.connect(self.open_repo); tl.addWidget(b_repo)
        b_settings = QPushButton("设置"); b_settings.setObjectName("ghost")
        b_settings.setToolTip("皮肤 / 玻璃质感 / 默认导出文件夹")
        b_settings.clicked.connect(self.open_settings); tl.addWidget(b_settings)
        cl.addWidget(top)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_actuator_tab())   # index 0 = 01 电缸管理
        self.stack.addWidget(self._build_param_tab())      # index 1 = 02 平台参数
        self.stack.addWidget(self._build_fk_tab())
        self.stack.addWidget(self._build_ik_tab())
        self.stack.addWidget(self._build_view_tab())
        self.stack.addWidget(self._build_kinematics_tab())
        self.stack.addWidget(self._build_batch_tab())
        self.stack.addWidget(self._build_trajectory_tab())  # index 7 = 08 轨迹播放
        self.stack.addWidget(self._build_help_tab())         # index 8 = 09 使用说明
        cl.addWidget(self.stack)
        self._page_names = pages
        self.stack.setCurrentIndex(0)
        self.page_title.setText(self._page_names[0][1])
        self.page_sub.setText(self._page_names[0][2])

    def _on_nav(self, index):
        self.stack.setCurrentIndex(index)
        self.page_title.setText(self._page_names[index][1])
        self.page_sub.setText(self._page_names[index][2])

    def open_repo(self):
        """在默认浏览器中打开项目 GitHub 仓库。"""
        QDesktopServices.openUrl(QUrl(REPO_URL))

    def open_settings(self):
        dlg = SettingsDialog(self.app_ref, self)
        dlg.exec()
        # 驱动方式或串口参数可能变化，切换 worker（串口 ↔ EtherCAT）
        det = getattr(self, "link_detector", None)
        if det:
            st = read_settings()
            try:
                baud = int(st.get("device_baud", 115200))
            except (TypeError, ValueError):
                baud = 115200
            det.set_driver(st.get("device_driver", "serial"),
                           st.get("device_port", ""), baud,
                           int(st.get("device_nic_index", 0) or 0),
                           od_cfg=st.get("ec_od", {}))
        self._set_debug_visible(read_settings().get("debug_mode", False))

    def _set_debug_visible(self, on: bool):
        """调试模式统一显隐：顶栏徽章 / 运动学派生区 / FK·IK 调试列 / IK dt / 圈数输入模式。"""
        on = bool(on)
        for name in ("kin_debug_box", "debug_badge", "ik_debug_dt", "link_indicator",
                     "b_send_device", "b_send_device_ik", "b_restore_ik_pose"):
            w = getattr(self, name, None)
            if w:
                w.setVisible(on)
        det = getattr(self, "link_detector", None)
        if det:
            if on and not det.timer.isActive():
                det.start()
            elif not on and det.timer.isActive():
                det.stop()
        fk, ik = getattr(self, "fk_table", None), getattr(self, "ik_table", None)
        if fk:
            for c in (6, 7):
                fk.setColumnHidden(c, not on)
        if ik:
            for c in (3, 4):
                ik.setColumnHidden(c, not on)
        combo = getattr(self, "fk_input_mode", None)
        if combo and combo.count() > 2:
            item = combo.model().item(2)
            if item:
                item.setEnabled(on)
            if not on and self._fk_mode == 2:
                combo.blockSignals(True); combo.setCurrentIndex(0); combo.blockSignals(False)
                self._on_fk_mode_change(0)

    # ---------- 主题 ----------

    def apply_theme(self, theme):
        if hasattr(self, "rail"):
            self.rail.set_theme(theme)
        if hasattr(self, "content"):
            self.content.set_theme(theme)
        if hasattr(self, "link_indicator"):
            self.link_indicator.set_theme(theme)
        self._update_status_colors(theme)
        for v in self.viewports:
            v.set_theme(theme)
        # 表格输入/输出列配色
        for table, in_cols, out_cols in (
            (getattr(self, "fk_table", None), {1}, {2, 3, 6, 7}),
            (getattr(self, "ik_table", None), set(), {1, 2, 3, 4}),
        ):
            if table is None:
                continue
            for r in range(table.rowCount()):
                for c in range(table.columnCount()):
                    it = table.item(r, c)
                    if it is None:
                        continue
                    col = (theme.input if c in in_cols else
                           theme.output if c in out_cols else theme.text)
                    it.setForeground(QBrush(QColor(col)))

    def _update_status_colors(self, theme):
        pairs = [
            (getattr(self, "param_status", None), getattr(self, "_param_ok", True)),
            (getattr(self, "fk_status", None), getattr(self, "_fk_ok", True)),
            (getattr(self, "joint_status", None), not getattr(self, "_joint_bad", False)),
            (getattr(self, "ik_status", None), getattr(self, "_ik_ok", True)),
            (getattr(self, "ik_joint_status", None), not getattr(self, "_ik_joint_bad", False)),
            (getattr(self, "ik_limit_status", None), not getattr(self, "_ik_limit_bad", False)),
            (getattr(self, "verify_label", None), getattr(self, "_verify_ok", True)),
            (getattr(self, "limit_status", None), not getattr(self, "_limit_bad", False)),
            (getattr(self, "dyn_status", None), getattr(self, "_dyn_ok", True)),
            (getattr(self, "kin_status", None), getattr(self, "_kin_ok", True)),
            (getattr(self, "batch_status", None), True),
        ]
        for lab, ok in pairs:
            if lab is None:
                continue
            color = theme.success if ok else (theme.warning if lab is getattr(
                self, "joint_status", None) or lab is getattr(self, "ik_joint_status", None)
                else theme.danger)
            lab.setStyleSheet(f"color:{color}; font-weight:600;")
        if getattr(self, "param_warning", None) is not None:
            self.param_warning.setStyleSheet(
                f"color:{theme.warning}; background:{theme.surface_alt}; "
                f"border:1px solid {theme.warning}; border-radius:10px; padding:10px;")
        # 帮助页随主题重建
        if getattr(self, "help_label", None) is not None:
            self._refresh_help_html(theme)

    # ---------- 档案与结构 ----------

    def _sync_structure_ui(self):
        key = getattr(self.config, "structure", "SPU")
        spec = STRUCTURES.get(key, STRUCTURES["SPU"])
        for b in self.seg_group.buttons():
            b.setChecked(b._struct_key == key)
        self.structure_badge.setText(key)
        self.side_struct.setText(f"结构  {key}")
        self.structure_desc.setText(spec.description)
        # 节点摆角约束：上下两个节点始终可编辑（用户要求），不因结构类型禁用
        if hasattr(self, "base_limit_edit"):
            self.base_limit_edit.setEnabled(True)
            self.platform_limit_edit.setEnabled(True)

    def on_structure_changed(self, key):
        self.config.structure = key
        self._sync_structure_ui()
        self.apply_config(silent=True)
        self.auto_save()
        spec = STRUCTURES[key]
        QMessageBox.information(self, f"已切换为 {key}",
                                f"{spec.label}\n\n{spec.description}\n\n"
                                "长度-位姿方程不变，软件按新结构启用对应的铰点摆角校验。")

    def edit_note(self):
        text, ok = QInputDialog.getText(self, "编辑档案备注", "备注：", text=self.note)
        if ok:
            self.note = text
            self.auto_save()

    def open_actuator_dialog(self):
        """兼容旧调用：切换到内置的 01 电缸管理页（不再弹独立窗口）。"""
        self.nav_group.button(0).setChecked(True)
        self._on_nav(0)

    def auto_save(self):
        inputs = []
        if hasattr(self, "fk_edits"):
            try:
                # 保存当前模式数据到缓存，然后持久化模式 + 两组数据
                self.fk_input_cache[self._fk_mode] = parse_floats(
                    self.fk_edits, "缸长").tolist()
                inputs = {"mode": self._fk_mode,
                          "absolute": self.fk_input_cache[0],
                          "relative": self.fk_input_cache[1],
                          "revolutions": self.fk_input_cache[2]}
            except Exception:
                inputs = []
        self.manager.save(self.profile_name, self.config.to_dict(), self.note, inputs)

    def back_to_launcher(self):
        self.auto_save(); self.app_ref.show_launcher(); self.close()

    def closeEvent(self, event):
        try:
            self.auto_save()
        except Exception:
            pass
        det = getattr(self, "link_detector", None)
        if det is not None:
            try:
                det.shutdown()
            except Exception:
                pass
        super().closeEvent(event)

    def _connect_all_hints(self):
        """统一连接所有问号按钮的点击信号（此前从未连接，导致点不开）。"""
        count = 0
        for btn in self.findChildren(QToolButton):
            if btn.objectName() == "hint" and hasattr(btn, "_hint_key"):
                key = btn._hint_key
                btn.clicked.connect(lambda checked=False, k=key: self._open_hint(k))
                count += 1
        self._hint_count = count

    def _open_hint(self, key):
        dlg = HintDialog(key, self.config, self.app_ref.theme_mgr.theme, self)
        dlg.exec()

    # ==================== 页面 0：电缸管理（内置页） ====================

    def _build_actuator_tab(self):
        """电缸管理：内置页，整机规格 + 15 电机字段 + 安全载荷，单页一体化。"""
        def build(lay, host):
            head = QHBoxLayout()
            hb = QVBoxLayout(); hb.setSpacing(3)
            hb.addWidget(label("电缸管理与安全载荷", "largeTitle"))
            hb.addWidget(label("填写电缸机械/电气规格；派生项按公式自动填入（问号可看推导）", "caption"))
            head.addLayout(hb); head.addStretch()
            lay.addLayout(head)

            body = QHBoxLayout(); body.setSpacing(16)
            left_scroll = QScrollArea(); left_scroll.setWidgetResizable(True)
            left_scroll.setFrameShape(QFrame.NoFrame)
            left_host = QWidget(); left = QVBoxLayout(left_host)
            left.setContentsMargins(2, 2, 8, 2); left.setSpacing(12)

            # 左栏唯一卡片：22 项按物理量相近混排
            self._build_actuator_fields_inline(left)
            left.addStretch()
            left_scroll.setWidget(left_host)
            body.addWidget(left_scroll, stretch=4)

            # 右：安全载荷
            right = QVBoxLayout(); right.setSpacing(12)
            out, ol = card("六轴各动作安全载荷", tint="output")
            ol.addWidget(label("额定推力约束下，各动作方向可安全承载的负载质量（kg）", "caption", wrap=True))
            load_grid = QGridLayout(); load_grid.setSpacing(9)
            self.safe_labels = []
            for idx, (nm, hk) in enumerate([
                ("X 纵荡", "out_x"), ("Y 横荡", "out_y"), ("Z 升沉", "out_z"),
                ("Roll 横滚", "out_roll"), ("Pitch 俯仰", "out_pitch"), ("Yaw 偏航", "out_yaw")]):
                mc, mv = metric_card(nm, hk, "kg", note="安全载荷 kg")
                load_grid.addWidget(mc, idx // 2, idx % 2)
                self.safe_labels.append(mv)
            ol.addLayout(load_grid)
            ol.addWidget(label("说明：按虚功原理 F=pinv(Jᵀ)·w，取最先达到额定推力的缸为上限。",
                               "caption", wrap=True))
            right.addWidget(out); right.addStretch()
            body.addLayout(right, stretch=6)
            lay.addLayout(body)

        page, _, _ = scroll_page(build)
        self._update_safe_labels()
        return page

    def _build_actuator_fields_inline(self, parent_layout):
        """左栏唯一卡片：22 项按物理量相近混排；派生↺/问号/锁定全保留。"""
        self.g_edits = {}          # 整机 7 项（不含 max_force/max_speed 镜像）
        self.motor_edits = {}      # 电机 15 项（权威）
        self._auto_buttons = {}
        acard, acl = card("电缸规格参数（按物理量相近排列）", tint="input")
        grid = QGridLayout(); grid.setHorizontalSpacing(12); grid.setVerticalSpacing(5)

        for row, key in enumerate(UI_ORDER_KEYS):
            if key in FIELDS_BY_KEY:
                f = FIELDS_BY_KEY[key]
                label_txt = f"{f.label} ({f.unit})"
                kind, hk, derived = "float", f"act_{f.key}", f.derived
                val = getattr(self.config, f.key, 0) or 0
            else:
                _, label0, unit, kind, hk = EXTRA_BY_KEY[key]
                label_txt = f"{label0} ({unit})" if unit else label0
                derived = False
                if kind == "str":
                    arr = {"actuator_name": self.config.actuator_names,
                           "actuator_code": self.config.actuator_codes,
                           "actuator_brand": self.config.actuator_brands}[key]
                    val = arr[0] if arr else ""
                else:
                    val = getattr(self.config, key, 0) or 0

            e = QLineEdit(str(val)) if kind == "str" else num_edit(val, 100)
            if kind == "str":
                e.setFixedWidth(140)
            cell = param_cell(label_txt, hk, e)
            # 删镜像框后原 act_force / act_speed 问号不能丢，补挂在权威字段上
            if key == "rated_force":
                cell.layout().insertWidget(2, make_hint_button("act_force"))
            elif key == "rated_speed":
                cell.layout().insertWidget(2, make_hint_button("act_speed"))

            if derived:
                e.setObjectName("calcInput")
                e._field_key = key; e._dirty = False
                auto = QPushButton("↺"); auto.setObjectName("ghost")
                auto.setFixedWidth(26); auto.setToolTip("恢复自动计算")
                auto.clicked.connect(lambda checked=False, k=key: self._reset_auto(k))
                self._auto_buttons[key] = auto
                w = QWidget(); rw = QHBoxLayout(w)
                rw.setContentsMargins(0, 0, 0, 0); rw.setSpacing(2)
                rw.addWidget(cell); rw.addWidget(auto)
                grid.addWidget(w, row, 0)
            else:
                grid.addWidget(cell, row, 0)

            (self.motor_edits if key in FIELDS_BY_KEY else self.g_edits)[key] = e
            if key in FIELDS_BY_KEY:
                e.textChanged.connect(self._on_motor_changed)

        acl.addLayout(grid)
        # 零位定义 / 应用按钮 / 状态：保留原逻辑
        zr = QHBoxLayout(); zr.addWidget(label("零位定义"))
        zr.addWidget(make_hint_button("act_zero_ref"))
        self.zero_combo = QComboBox()
        self.zero_combo.addItem("零位 = 行程最低位", "lowest")
        self.zero_combo.addItem("零位 = 行程中间位", "middle")
        self.zero_combo.setCurrentIndex(0 if self.config.zero_reference == "lowest" else 1)
        zr.addWidget(self.zero_combo); zr.addStretch(); acl.addLayout(zr)
        b_apply = QPushButton("应用电缸参数  →"); b_apply.setObjectName("primary")
        b_apply.clicked.connect(self._save_actuator_page); acl.addWidget(b_apply)
        self.act_status = label("修改规格后点上方按钮应用。", "caption", wrap=True)
        acl.addWidget(self.act_status)
        parent_layout.addWidget(acard)
        self._recompute_derived()

    def _update_safe_labels(self):
        try:
            solver = StewartKinematics(self.config)
            safe = solver.safe_payloads()
            for i, lab in enumerate(self.safe_labels):
                lab.setText(f"{safe[i]:.1f}")
        except Exception:
            for lab in self.safe_labels:
                lab.setText("—")

    def _motor_lead(self):
        try:
            return float(getattr(self.config, "screw_lead", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    def _collect_motor_ctx(self):
        return {f.key: self.motor_edits[f.key].text() for f in FIELDS if f.key in self.motor_edits}

    def _recompute_derived(self):
        results = recompute(self._collect_motor_ctx())
        for key, (val, warn) in results.items():
            e = self.motor_edits.get(key)
            if e is None or getattr(e, "_dirty", False):
                continue
            with QSignalBlocker(e):
                e.setText("" if val is None else f"{val:.6g}")
            btn = self._auto_buttons.get(key)
            if btn:
                btn.setToolTip("恢复自动计算" + (f"（{warn}）" if warn else ""))

    def _on_motor_changed(self):
        s = self.sender()
        if isinstance(s, QLineEdit) and hasattr(s, "_field_key"):
            s._dirty = True
        self._recompute_derived()

    def _reset_auto(self, key):
        e = self.motor_edits.get(key)
        if e:
            e._dirty = False
        self._recompute_derived()

    def _save_actuator_page(self):
        try:
            for key, w in self.g_edits.items():
                if key == "actuator_name":
                    self.config.actuator_names = [w.text().strip()]
                elif key == "actuator_code":
                    self.config.actuator_codes = [w.text().strip()]
                elif key == "actuator_brand":
                    self.config.actuator_brands = [w.text().strip()]
                else:
                    setattr(self.config, key, float(w.text() or 0))
            self.config.zero_reference = self.zero_combo.currentData()
            for f in FIELDS:
                e = self.motor_edits.get(f.key)
                if e:
                    try:
                        setattr(self.config, f.key, float(e.text() or 0))
                    except ValueError:
                        setattr(self.config, f.key, 0.0)
            ctx = self._collect_motor_ctx()
            for key, (val, _) in recompute(ctx).items():
                if val is not None and not self.motor_edits[key]._dirty:
                    setattr(self.config, key, float(val))
            self.config.max_force = self.config.rated_force
            self.config.max_speed = self.config.peak_speed or self.config.rated_speed
            self.config.leg_strokes = None; self.config.leg_forces = None; self.config.leg_speeds = None
            self._load_config_into_form()
            self.apply_config(silent=True)
            self.auto_save()
            self._update_safe_labels()
            self.act_status.setText("✓ 已应用并同步。")
            self.apply_theme(self.app_ref.theme_mgr.theme)
        except Exception as ex:
            self.act_status.setText(f"✗ 失败：{ex}")
            QMessageBox.warning(self, "应用失败", str(ex))

    # ==================== 页面 1：平台参数 ====================

    def _build_param_tab(self):
        def build(lay, w):
            # 结构切换
            c0, l0 = card("机构结构", tint="param")
            srow = QHBoxLayout(); srow.setSpacing(6)
            seg = QFrame(); seg.setObjectName("segContainer")
            scl = QHBoxLayout(seg); scl.setContentsMargins(2, 2, 2, 2); scl.setSpacing(0)
            self.seg_group = QButtonGroup(); self.seg_group.setExclusive(True)
            for key, spec in STRUCTURES.items():
                b = QPushButton(spec.label.split(" — ")[0])
                b.setObjectName("segment"); b.setCheckable(True)
                b.clicked.connect(lambda checked, k=key: self.on_structure_changed(k))
                scl.addWidget(b); self.seg_group.addButton(b); b._struct_key = key
            srow.addWidget(seg); srow.addStretch()
            l0.addLayout(srow)
            self.structure_desc = label("", "caption", wrap=True)
            l0.addWidget(self.structure_desc)
            lay.addWidget(c0)

            # 建模方式
            c1, l1 = card("建模方式", tint="param")
            mr = QHBoxLayout()
            self.rb_param = QCheckBox("参数化建模（对称 6-6 平台）")
            self.rb_param.setChecked(True)
            self.rb_direct = QCheckBox("直接输入铰点坐标（任意 6-6 平台）")
            self.rb_param.toggled.connect(lambda ck: self.rb_direct.setChecked(not ck) if ck else None)
            self.rb_direct.toggled.connect(lambda ck: self.rb_param.setChecked(not ck) if ck else None)
            self.rb_direct.toggled.connect(self._on_model_switch)
            mr.addWidget(self.rb_param); mr.addWidget(self.rb_direct); mr.addStretch()
            l1.addLayout(mr); lay.addWidget(c1)

            # 几何参数
            self.grp_param, pl = card("几何参数（长度 mm · 角度 °）", tint="param")
            self.param_edits = {}
            # 成对排列：每行左=下平台参数、右=对应上平台参数
            fields = [
                ("下平台铰点圆半径 Rb", "base_radius", "base_radius"),
                ("上平台铰点圆半径 Rp", "platform_radius", "platform_radius"),
                ("下平台成对夹角 φb", "base_pair_angle", "base_pair_angle"),
                ("上平台成对夹角 φp", "platform_pair_angle", "platform_pair_angle"),
                ("下平台相位角 βb", "base_phase", "base_phase"),
                ("上平台相位角 βp", "platform_phase", "platform_phase"),
                ("下铰点 Z 偏移", "base_joint_z", "base_joint_z"),
                ("上铰点 Z 偏移", "platform_joint_z", "platform_joint_z"),
                ("零位上平台高度 H0", "home_height", "home_height"),
            ]
            grid = QGridLayout(); grid.setHorizontalSpacing(14); grid.setVerticalSpacing(12)
            for idx, (text, key, hint_key) in enumerate(fields):
                row, col = divmod(idx, 2)
                e = num_edit(getattr(self.config, key), 96)
                grid.addWidget(param_cell(text, hint_key, e), row, col)
                self.param_edits[key] = e
            center_cell = QWidget(); cl = QHBoxLayout(center_cell)
            cl.setContentsMargins(0, 0, 0, 0); cl.setSpacing(4)
            cl.addWidget(label("三组铰点中心方位角"))
            cl.addWidget(make_hint_button("pair_centers"))
            self.centers_edit = QLineEdit("0, 120, 240"); self.centers_edit.setFixedWidth(140)
            cl.addWidget(self.centers_edit)
            grid.addWidget(center_cell, 4, 1)
            pl.addLayout(grid); lay.addWidget(self.grp_param)

            # 直接坐标
            self.grp_direct, dl = card("铰点坐标（本地坐标系，mm）", tint="param")
            dl.addWidget(label("下铰点为世界坐标；上铰点为平台本地坐标。按缸 1~6 对应顺序填写。",
                               "caption", wrap=True))
            tables = QHBoxLayout()
            self.base_table, base_box = self._make_points_table("下平台铰点 B1~B6")
            self.plat_table, plat_box = self._make_points_table("上平台铰点 P1~P6")
            tables.addWidget(base_box); tables.addWidget(plat_box); tables.addStretch()
            dl.addLayout(tables)
            hz = QHBoxLayout(); hz.addWidget(label("零位平移 (x,y,z)"))
            hz.addWidget(make_hint_button("home_height"))
            self.home_t_edit = QLineEdit("0, 0, 600"); self.home_t_edit.setFixedWidth(180)
            hz.addWidget(self.home_t_edit); hz.addStretch(); dl.addLayout(hz)
            self.grp_direct.setVisible(False); lay.addWidget(self.grp_direct)

            # 节点摆角约束（上下均可编辑）
            c3, l3 = card("节点摆角约束", tint="param")
            cg = QHBoxLayout(); cg.setSpacing(6)
            cg.addWidget(label("下节点许用摆角 (°)："))
            cg.addWidget(make_hint_button("base_joint_limit"))
            self.base_limit_edit = num_edit(self.config.base_joint_limit)
            cg.addWidget(self.base_limit_edit)
            cg.addSpacing(16)
            cg.addWidget(label("上节点许用摆角 (°)："))
            cg.addWidget(make_hint_button("platform_joint_limit"))
            self.platform_limit_edit = num_edit(self.config.platform_joint_limit)
            self.base_limit_edit.setObjectName("calcInput")
            self.platform_limit_edit.setObjectName("calcInput")
            cg.addWidget(self.platform_limit_edit); cg.addStretch()
            l3.addLayout(cg)
            l3.addWidget(label("摆角 = 缸轴与节点铰中轴线的夹角，超过许用值将给出警告。"
                               "上下节点独立设置；0 = 不校验。默认中轴：下节点 +Z（竖直），上节点 −Z。",
                               "caption", wrap=True))
            lay.addWidget(c3)

            # 标定与限位
            c4, l4 = card("作动器标定与限位（可选）", tint="param")
            r1 = QHBoxLayout(); r1.addWidget(label("零位实测缸长 L0_1~L0_6"))
            r1.addWidget(make_hint_button("zero_lengths"))
            self.zero_edits = [num_edit(0, 82) for _ in range(6)]
            for e in self.zero_edits: r1.addWidget(e)
            r1.addStretch(); l4.addLayout(r1)
            r2 = QHBoxLayout(); r2.addWidget(label("行程下限 ΔL："))
            self.min_edit = num_edit(0, 82); self.min_edit.setObjectName("calcInput"); r2.addWidget(self.min_edit)
            r2.addWidget(label("行程上限 ΔL：")); self.max_edit = num_edit(0, 82); self.max_edit.setObjectName("calcInput")
            r2.addWidget(self.max_edit); r2.addWidget(make_hint_button("stroke"))
            r2.addStretch(); l4.addLayout(r2)

            # 零位定义：决定行程如何在零位两侧分配
            r3 = QHBoxLayout()
            r3.addWidget(label("零位定义："))
            r3.addWidget(make_hint_button("act_zero_ref"))
            self.zero_ref_combo = QComboBox()
            self.zero_ref_combo.addItem("零位 = 行程最低位", "lowest")
            self.zero_ref_combo.addItem("零位 = 行程中间位", "middle")
            r3.addWidget(self.zero_ref_combo); r3.addStretch()
            l4.addLayout(r3)

            # 应用参数后自动输出各动作最大值（卡片网格，排版同正解结果页）
            ah = QHBoxLayout()
            ah.addWidget(label("各动作最大值（按行程与摆角约束实时计算）", "caption"))
            ah.addWidget(make_hint_button("ws_action_max")); ah.addStretch()
            l4.addLayout(ah)
            action_grid = QGridLayout(); action_grid.setSpacing(9)
            self.action_labels = []
            act_defs = [
                ("X 纵荡", "out_x", "mm"), ("Y 横荡", "out_y", "mm"),
                ("Z 升沉", "out_z", "mm"), ("Roll 横滚", "out_roll", "°"),
                ("Pitch 俯仰", "out_pitch", "°"), ("Yaw 偏航", "out_yaw", "°"),
            ]
            for idx, (nm, hk, unit) in enumerate(act_defs):
                ac, av = metric_card(nm, hk, unit, note=f"最大行程 {unit}")
                action_grid.addWidget(ac, idx // 3, idx % 3)
                self.action_labels.append(av)
            l4.addLayout(action_grid)
            lay.addWidget(c4)
            # 电缸规格/限位/行程等数据统一在侧边栏『电缸管理』窗口编辑，此处不再重复

            # 性能诊断：归一化 SVD 奇异值 / 条件数 / 可操作性 / 各向同性度
            cdiag, ldiag = card("性能诊断（归一化雅可比 SVD）", tint="output")
            dh = QHBoxLayout(); dh.addStretch(); dh.addWidget(make_hint_button("perf_svd"))
            ldiag.insertLayout(0, dh)
            self.svd_labels = []
            for i in range(6):
                row = QHBoxLayout()
                row.addWidget(label(f"σ{i+1}", "caption"))
                row.addStretch()
                bar = QProgressBar(); bar.setRange(0, 100); bar.setTextVisible(True)
                bar.setFixedWidth(180); row.addWidget(bar)
                val = label("—", "inValue"); val.setFixedWidth(70); row.addWidget(val)
                ldiag.addLayout(row)
                self.svd_labels.append((bar, val))
            self.svd_status = label("应用参数后显示。σ 越小越接近奇异。", "caption", wrap=True)
            ldiag.addWidget(self.svd_status)
            lay.addWidget(cdiag)

            # 工作空间点云已迁至『三维姿态』页，叠加在主视口显示。

            self.param_warning = QLabel(""); self.param_warning.setWordWrap(True)
            self.param_warning.setVisible(False); lay.addWidget(self.param_warning)
            b_apply = QPushButton("应用参数并校验"); b_apply.setObjectName("primary")
            b_apply.clicked.connect(lambda: self.apply_config()); lay.addWidget(b_apply)
            self.param_status = label(""); self.param_status.setObjectName("caption")
            self.param_status.setWordWrap(True); lay.addWidget(self.param_status)
            lay.addStretch()

            # 行程/摆角改动时实时刷新各动作最大值（无需点应用）
            for wdg in (self.min_edit, self.max_edit,
                        self.base_limit_edit, self.platform_limit_edit):
                wdg.textChanged.connect(self._preview_action_limits)

        return scroll_page(build)[0]

    def _refresh_svd_panel(self, solver=None):
        """刷新归一化 SVD 性能诊断面板。"""
        if not hasattr(self, "svd_labels"):
            return
        solver = solver or getattr(self, "solver", None)
        if solver is None:
            return
        try:
            m = solver.normalized_svd()
            s = m["singular_values"]
            smax = float(s[0]) if len(s) else 1.0
            for (bar, val), sv in zip(self.svd_labels, s):
                pct = int(max(0.0, min(100.0, float(sv) / smax * 100.0))) if smax > 0 else 0
                bar.setValue(pct); bar.setFormat(f"{pct}%")
                val.setText(f"{float(sv):.4f}")
            self.svd_status.setText(
                f"条件数 {m['condition']:.2f} · 可操作性 {m['manipulability']:.4g} · "
                f"最小奇异值 {m['min_singular']:.4f} · 各向同性度 {m['isotropy']:.3f} · "
                f"特征长度 {m['length_scale']:.1f}mm（σ 越小越接近奇异）")
        except Exception as ex:
            self.svd_status.setText(f"诊断失败：{ex}")

    def _toggle_force_ellipsoid(self, checked):
        """在全部视口显示/隐藏力椭球点云。"""
        if not checked:
            for v in self.viewports:
                v.set_force_ellipsoid(None)
            return
        solver = getattr(self, "solver", None)
        if solver is None:
            return
        try:
            ell = solver.force_ellipsoid()
            pts = ell["points"]
            for v in self.viewports:
                v.set_force_ellipsoid(pts)
            self.svd_status.setText(
                self.svd_status.text() + f"\n力椭球半轴比：{np.round(ell['radii'], 3).tolist()}（1=最强方向）")
        except Exception as ex:
            self.svd_status.setText(f"力椭球计算失败：{ex}")

    def _preview_action_limits(self):
        """用当前输入框的行程/摆角值实时预览各动作最大值，不改主配置。"""
        if not hasattr(self, "action_labels") or not hasattr(self, "config"):
            return
        try:
            import copy
            cfg = copy.deepcopy(self.config)
            cfg.min_length = float(self.min_edit.text() or 0)
            cfg.max_length = float(self.max_edit.text() or 0)
            cfg.base_joint_limit = float(self.base_limit_edit.text() or 0)
            cfg.platform_joint_limit = float(self.platform_limit_edit.text() or 0)
            acts = StewartKinematics(cfg).workspace_limits()
            names = [("X 纵荡", "mm"), ("Y 横荡", "mm"), ("Z 升沉", "mm"),
                     ("Roll 横滚", "°"), ("Pitch 俯仰", "°"), ("Yaw 偏航", "°")]
            for i, (_, unit) in enumerate(names):
                mx = max(abs(acts[i, 0]), abs(acts[i, 1]))
                self.action_labels[i].setText(f"±{mx:.2f} {unit}")
        except Exception:
            pass

    def _make_points_table(self, title):
        box = QWidget(); lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(label(title, "headline"))
        t = QTableWidget(6, 3)
        t.setHorizontalHeaderLabels(["X", "Y", "Z"])
        t.setVerticalHeaderLabels([f"#{i+1}" for i in range(6)])
        t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        t.setFixedHeight(186); t.setFixedWidth(300)
        defaults = [[500, 0, 0], [250, 433, 0], [-250, 433, 0],
                    [-500, 0, 0], [-250, -433, 0], [250, -433, 0]]
        for i in range(6):
            for j in range(3):
                t.setItem(i, j, QTableWidgetItem(str(defaults[i][j])))
        lay.addWidget(t)
        return t, box

    def _sync_mode_visibility(self):
        direct = self.rb_direct.isChecked()
        self.grp_param.setVisible(not direct)
        self.grp_direct.setVisible(direct)

    def _on_model_switch(self, checked):
        """切换建模方式：显示当前配置在另一种方式下的等效结果。"""
        self._sync_mode_visibility()
        if not hasattr(self, "config") or self.config is None:
            return
        # 直接坐标模式：显示缓存的等效坐标结果
        if self.rb_direct.isChecked():
            alt = getattr(self, "_alt_cfg", None)
            if alt is not None:
                self._show_model_result(alt)
        else:
            # 参数化模式：显示当前 cfg 的参数化结果
            self._show_model_result(self.config)

    def _build_alternate_cfg(self, cfg):
        """由当前模型推导出另一种建模方式的等效配置（缓存供切换查看）。"""
        try:
            base, plat = cfg.anchors()
            origin = cfg.origin()
        except Exception:
            return None
        import copy
        alt = copy.deepcopy(cfg)
        alt.use_direct_points = True
        alt.base_points = base.copy()
        alt.platform_points = plat.copy()
        alt.home_translation = origin.copy()
        # 同时拟合参数化字段，切换回去也能看到对应值
        try:
            rb = float(np.mean(np.linalg.norm(base[:, :2], axis=1)))
            rp = float(np.mean(np.linalg.norm(plat[:, :2], axis=1)))
            ang_b = np.degrees(np.arctan2(base[:, 1], base[:, 0])) % 360
            ang_p = np.degrees(np.arctan2(plat[:, 1], plat[:, 0])) % 360

            def pair_gap(angs):
                gaps = []
                for k in range(0, 6, 2):
                    d = (angs[k] - angs[k + 1]) % 360
                    gaps.append(min(d, 360 - d))
                return float(np.mean(gaps)) if gaps else 0.0

            pb, pp = pair_gap(ang_b), pair_gap(ang_p)
            alt.base_radius = rb; alt.platform_radius = rp
            alt.base_pair_angle = pb; alt.platform_pair_angle = pp
            alt.base_phase = float(ang_b[0] - pb / 2.0) % 360
            alt.platform_phase = float(ang_p[0] - pp / 2.0) % 360
            alt.home_height = float(origin[2])
            alt.base_joint_z = float(base[0, 2])
            alt.platform_joint_z = float(plat[0, 2])
        except Exception:
            pass
        return alt

    def _show_model_result(self, cfg):
        """把给定配置的结果显示到两种模型的表单控件上（只读展示）。"""
        try:
            base, plat = cfg.anchors(); origin = cfg.origin()
            for t, pts in ((self.base_table, base), (self.plat_table, plat)):
                for i in range(6):
                    for j in range(3):
                        t.setItem(i, j, QTableWidgetItem(f"{pts[i, j]:.4f}"))
            self.home_t_edit.setText(", ".join(f"{x:.4f}" for x in origin))
            for key, e in self.param_edits.items():
                if hasattr(cfg, key):
                    e.setText(f"{getattr(cfg, key):.4f}")
            self.centers_edit.setText(", ".join(str(x) for x in cfg.pair_centers))
        except Exception:
            pass

    def _load_config_into_form(self):
        c = self.config
        for key, e in self.param_edits.items():
            e.setText(str(getattr(c, key)))
        self.centers_edit.setText(", ".join(str(x) for x in c.pair_centers))
        self.min_edit.setText(str(c.min_length)); self.max_edit.setText(str(c.max_length))
        self.base_limit_edit.setText(str(c.base_joint_limit))
        self.platform_limit_edit.setText(str(c.platform_joint_limit))
        self.zero_ref_combo.setCurrentIndex(0 if c.zero_reference == "lowest" else 1)
        z = c.zero_lengths
        for e, v in zip(self.zero_edits, z if z is not None else [0] * 6):
            e.setText(str(v))
        # 缓存另一种建模方式的等效配置（屏蔽信号避免触发切换处理）
        try:
            self._alt_cfg = self._build_alternate_cfg(c)
        except Exception:
            self._alt_cfg = None
        self.rb_param.blockSignals(True); self.rb_direct.blockSignals(True)
        self.rb_direct.setChecked(bool(c.use_direct_points))
        self.rb_param.setChecked(not bool(c.use_direct_points))
        self.rb_param.blockSignals(False); self.rb_direct.blockSignals(False)
        if c.use_direct_points:
            if c.base_points is not None:
                for t, pts in ((self.base_table, c.base_points), (self.plat_table, c.platform_points)):
                    for i in range(6):
                        for j in range(3):
                            t.setItem(i, j, QTableWidgetItem(str(pts[i, j])))
            if c.home_translation is not None:
                self.home_t_edit.setText(", ".join(str(x) for x in c.home_translation))
        self._sync_mode_visibility()

    def apply_config(self, silent=False):
        try:
            if self.rb_param.isChecked():
                vals = {k: float(e.text()) for k, e in self.param_edits.items()}
                cfg = PlatformConfig(
                    name=self.profile_name, use_direct_points=False,
                    pair_centers=tuple(parse_series(self.centers_edit.text(), 3)),
                    structure=getattr(self.config, "structure", "SPU"), **vals)
            else:
                def read_table(t):
                    arr = []
                    for i in range(6):
                        row = []
                        for j in range(3):
                            it = t.item(i, j)
                            if it is None or not it.text().strip():
                                raise ValueError(f"铰点表第 {i+1} 行第 {j+1} 列为空")
                            row.append(float(it.text()))
                        arr.append(row)
                    return np.asarray(arr, float)
                cfg = PlatformConfig(
                    name=self.profile_name, use_direct_points=True,
                    base_points=read_table(self.base_table),
                    platform_points=read_table(self.plat_table),
                    home_translation=parse_series(self.home_t_edit.text(), 3),
                    structure=getattr(self.config, "structure", "SPU"))
            zraw = [float(e.text()) for e in self.zero_edits]
            if any(abs(v) > 1e-9 for v in zraw):
                cfg.zero_lengths = np.asarray(zraw, float)
            cfg.min_length = float(self.min_edit.text())
            cfg.max_length = float(self.max_edit.text())
            cfg.base_joint_limit = float(self.base_limit_edit.text())
            cfg.platform_joint_limit = float(self.platform_limit_edit.text())
            # 电缸规格/身份字段统一来自『电缸管理』窗口：从旧配置保留
            for key in ("actuator_names", "actuator_codes", "actuator_brands",
                        "leg_strokes", "leg_forces", "leg_speeds",
                        "stroke", "max_speed", "max_force", "actuator_bore",
                        "payload_mass", "platform_mass", "gravity",
                        "max_translation", "max_tilt",
                        "rated_force", "peak_force", "rated_speed", "peak_speed",
                        "max_accel", "screw_lead", "rated_voltage", "rated_power",
                        "rated_torque", "rated_current", "peak_current", "rated_rpm",
                        "max_rpm", "torque_constant", "back_emf_constant", "debug_mode"):
                setattr(cfg, key, getattr(self.config, key, 0))
            cfg.zero_reference = self.zero_ref_combo.currentData()

            # 行程关联校验：行程 S = 上限−下限，不能超过电缸物理行程 stroke
            S = cfg.max_length - cfg.min_length
            stroke = getattr(cfg, "stroke", 0) or 0
            stroke_warn = ""
            if S <= 0 and stroke <= 0:
                stroke_warn = "⚠ 行程未配置：电缸行程(stroke)为必填项，且行程上限应大于下限。"
            elif S > 0 and stroke > 0 and S > stroke + 1e-9:
                stroke_warn = (f"⚠ 行程超限：上限−下限={S:g}mm 超过电缸物理行程 {stroke:g}mm。")
            self._stroke_warn = stroke_warn

            solver = StewartKinematics(cfg)
            self.config = cfg; self.solver = solver
            # 缓存另一种建模方式的等效配置，供切换查看
            self._alt_cfg = self._build_alternate_cfg(cfg)
            # 即时刷新另一种建模方式的显示结果（不切换当前编辑模式）
            self._show_model_result(cfg)
            self._sync_structure_ui()
            for v in self.viewports:
                v.set_model(solver, cfg)
            # 参数变更后旧点云失效（基于旧行程/限位），清除避免误导
            if getattr(self, "_ws_points6", None) is not None:
                self._ws_points6 = None
                self._ws_colors = None
                if hasattr(self, "main_viewport"):
                    self.main_viewport.show_pointcloud = False
                    self.main_viewport.set_pointcloud(None)
                self._set_cloud_button(False)
                if hasattr(self, "ws_status"):
                    self.ws_status.setText("参数已变更，旧点云已清除；请重新采样。")
            if not silent:
                d = (f"✓ 参数已应用 · 零位缸长 "
                     f"[{', '.join(f'{x:.2f}' for x in solver.ref_lengths)}] mm · "
                     f"雅可比秩 {solver.rank()}/6")
                self.param_status.setText(d + ("\n" + stroke_warn if stroke_warn else ""))
                self._param_ok = not bool(stroke_warn)
                warn = solver.geometry_warning()
                if stroke_warn:
                    warn = (warn + "\n" if warn else "") + stroke_warn
                self.param_warning.setVisible(bool(warn))
                self.param_warning.setText(warn)
                self.auto_save()
                self.apply_theme(self.app_ref.theme_mgr.theme)
                QMessageBox.information(self, "参数校验通过", d)
            self._refresh_ref_hint()
            self._refresh_svd_panel(solver)
            return True
        except Exception as ex:
            self.param_status.setText(f"✗ 参数错误：{ex}")
            self._param_ok = False
            self.apply_theme(self.app_ref.theme_mgr.theme)
            if not silent:
                QMessageBox.warning(self, "参数校验失败", str(ex))
            return False

    def _refresh_ref_hint(self):
        if self.solver:
            ref = self.solver.ref_lengths
            lim = self.config.actuator_limits()
            zero_ref = "最低位" if self.config.zero_reference == "lowest" else "中间位"
            self.ref_len_hint.setText(
                f"零位基准缸长 L0 = {', '.join(f'{x:.2f}' for x in ref)} mm · "
                f"生效行程 S={lim[0,1]-lim[0,0]:.1f}mm · "
                f"缸长范围 [{lim[0,0]:.1f}, {lim[0,1]:.1f}]mm · 零位={zero_ref}")
            # 自动输出各动作最大值（含行程 + 摆角约束，随参数实时变化）
            try:
                acts = self.solver.workspace_limits()
            except Exception:
                acts = np.zeros((6, 2))
            names = [("X 纵荡", "mm"), ("Y 横荡", "mm"), ("Z 升沉", "mm"),
                     ("Roll 横滚", "°"), ("Pitch 俯仰", "°"), ("Yaw 偏航", "°")]
            for i, (name, unit) in enumerate(names):
                mx = max(abs(acts[i, 0]), abs(acts[i, 1]))
                self.action_labels[i].setText(f"±{mx:.2f} {unit}")

    # ==================== 页面 2：姿态正解 ====================

    def _build_fk_tab(self):
        page = QWidget(); page.setObjectName("page"); root = QHBoxLayout(page)
        root.setContentsMargins(22, 18, 22, 20); root.setSpacing(16)

        # 左栏改为滚动容器：新增求解器下拉后内容超出可见高度，避免 6 个输入框被压扁
        left_host = QWidget(); left = QVBoxLayout(left_host); left.setSpacing(12)
        c1, l1 = card("① 输入 6 路作动器数据", tint="input")
        mr = QHBoxLayout(); mr.addWidget(label("输入类型"))
        self.fk_input_mode = QComboBox()
        self.fk_input_mode.addItems(["绝对缸长 L1~L6 (mm)", "相对零位位移 ΔL1~ΔL6 (mm)",
                                     "电机圈数 N1~N6 (rev)【调试】"])
        # 三种输入模式各自独立保存数据；圈数模式仅调试模式可选
        self.fk_input_mode.model().item(2).setEnabled(False)
        if not hasattr(self, "_fk_mode"):
            self._fk_mode = 0
        if not hasattr(self, "fk_input_cache"):
            self.fk_input_cache = [[0.0] * 6 for _ in range(3)]
        else:
            while len(self.fk_input_cache) < 3:
                self.fk_input_cache.append([0.0] * 6)
        self.fk_input_mode.currentIndexChanged.connect(self._on_fk_mode_change)
        mr.addWidget(self.fk_input_mode); mr.addStretch(); l1.addLayout(mr)

        self.fk_edits = []
        for i in range(6):
            e = num_edit("0", 130); e.setMinimumHeight(32); self.fk_edits.append(e)
            l1.addLayout(input_row(f"缸 {i+1}", "fk_leg_i", e, "mm"))
        self.ref_len_hint = label("零位基准缸长：—", "caption", wrap=True)
        l1.addWidget(self.ref_len_hint)
        pr = QHBoxLayout()
        for text, slot in [("从文本填充", self.paste_to_fk), ("从文件导入", self.import_fk_file),
                           ("填零位值", self.fill_home_lengths)]:
            b = QPushButton(text); b.clicked.connect(slot); pr.addWidget(b)
        l1.addLayout(pr)
        self.paste_box = QPlainTextEdit()
        self.paste_box.setPlaceholderText("粘贴 6 个数值（逗号/空格/换行均可），再点『从文本填充』")
        self.paste_box.setFixedHeight(54); l1.addWidget(self.paste_box)
        left.addWidget(c1)

        c2, l2 = card("② 求解设置", tint="input")
        self.fk_strategy = QComboBox()
        self.fk_strategy.addItems(["从零位开始（推荐）", "从上次解开始（连续轨迹）", "多起点全局搜索"])
        l2.addLayout(input_row("初值策略", "fk_strategy", self.fk_strategy))
        self.fk_solver = QComboBox()
        self.fk_solver.addItems(["TRF 阻尼最小二乘（推荐）", "LM Levenberg-Marquardt", "AUTO 两者竞争取最优"])
        l2.addLayout(input_row("求解器", "fk_solver", self.fk_solver))
        self.fk_tol = num_edit("0.000001", 120)
        l2.addLayout(input_row("收敛容差 (mm)", "fk_tol", self.fk_tol))
        self.fk_dt = num_edit("0", 90)
        l2.addLayout(input_row("采样间隔 dt (s)", "fk_dt", self.fk_dt))
        l2.addWidget(label("填 >0 且连续计算两次以上，可输出速度/加速度/推力。", "caption", wrap=True))
        left.addWidget(c2)
        b_solve = QPushButton("计算姿态  →"); b_solve.setObjectName("primary")
        b_solve.clicked.connect(self.run_fk); left.addWidget(b_solve)

        # 把当前正解结果的绝对位姿填入运动学页三帧（始终显示，非调试专属）
        fill = QHBoxLayout(); fill.setSpacing(6)
        fill.addWidget(label("填入运动学帧", "caption"))
        for text, slot in [("填入帧 k",   lambda: self.fill_kin_frame(2)),
                           ("填入帧 k−1", lambda: self.fill_kin_frame(1)),
                           ("填入帧 k−2", lambda: self.fill_kin_frame(0))]:
            b = QPushButton(text); b.setObjectName("ghost")
            b.setToolTip("把当前正解结果的绝对位姿 [X, Y, Z, Roll, Pitch, Yaw] "
                         "写入『运动学计算』页对应帧输入框；dt 需手动填写")
            b.clicked.connect(slot); fill.addWidget(b)
        fill.addStretch(); left.addLayout(fill)

        left.addStretch()
        left_scroll = QScrollArea(); left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame); left_scroll.setWidget(left_host)
        root.addWidget(left_scroll, stretch=4)

        # 右侧结果（可滚动）
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        rw = QWidget(); scroll.setWidget(rw)
        right = QVBoxLayout(rw); right.setContentsMargins(2, 2, 8, 2); right.setSpacing(12)
        out, vl = card("③ 解算结果", tint="output")
        cards = QGridLayout(); cards.setSpacing(9)
        self.pose_value_labels = {}
        for i, (name, key, unit) in enumerate([
            ("X 纵荡", "out_x", "mm"), ("Y 横荡", "out_y", "mm"), ("Z 升沉", "out_z", "mm"),
            ("Roll 横滚", "out_roll", "°"), ("Pitch 俯仰", "out_pitch", "°"),
            ("Yaw 偏航", "out_yaw", "°")]):
            mc, v = metric_card(name, key, unit)
            self.pose_value_labels[i] = v
            cards.addWidget(mc, i // 3, i % 3)
        vl.addLayout(cards)

        self.fk_status = label("尚未计算", "caption", wrap=True); vl.addWidget(self.fk_status)
        self.joint_status = QLabel(""); self.joint_status.setObjectName("caption")
        self.joint_status.setWordWrap(True); self.joint_status.setTextFormat(Qt.RichText)
        self.joint_status.linkActivated.connect(self._on_limit_link)
        vl.addWidget(self.joint_status)
        self.limit_status = QLabel(""); self.limit_status.setObjectName("caption")
        self.limit_status.setWordWrap(True); self.limit_status.setTextFormat(Qt.RichText)
        self.limit_status.linkActivated.connect(self._on_limit_link)
        vl.addWidget(self.limit_status)
        self.dyn_status = label("动态数据：填写采样间隔 dt 并连续计算两次以上即可输出速度/加速度/推力。",
                                "caption", wrap=True); vl.addWidget(self.dyn_status)
        tq = QHBoxLayout(); tq.addStretch(); tq.addWidget(make_hint_button("out_motor_torque"))
        vl.addLayout(tq)

        self.fk_table = QTableWidget(6, 8)
        self.fk_table.setHorizontalHeaderLabels(
            ["作动器", "输入长度 (mm)", "反算长度 (mm)", "残差 (mm)", "缸速 (mm/s)", "推力 (N)",
             "电机圈数 (rev)", "所需转速 (rpm)"])
        self.fk_table.setVerticalHeaderLabels([f"缸 {i+1}" for i in range(6)])
        self.fk_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.fk_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.fk_table.setFixedHeight(185); vl.addWidget(self.fk_table)

        er = QHBoxLayout()
        for text, slot in [("导出结果 CSV", self.export_fk_csv),
                           ("保存计算存档", self.export_calculation_archive),
                           ("复制结果", self.copy_fk_result)]:
            b = QPushButton(text); b.clicked.connect(slot); er.addWidget(b)
        self.b_send_device = QPushButton("下发到开发板"); self.b_send_device.setObjectName("primary")
        self.b_send_device.setToolTip("把当前 6 路缸长下发到开发板（调试模式）")
        self.b_send_device.clicked.connect(self._send_to_device)
        er.addWidget(self.b_send_device)
        er.addStretch(); vl.addLayout(er)
        right.addWidget(out)

        # 1:1 大三维预览 + 详细数据并排（排版同逆解页的大预览）
        lower = QHBoxLayout(); lower.setSpacing(12)
        vp_card, vp_lay = card("三维姿态预览", tint="view")
        head = QHBoxLayout(); head.addWidget(make_hint_button("out_rotation")); head.addStretch()
        b_reset = QPushButton("重置视角"); b_reset.setObjectName("ghost")
        b_reset.clicked.connect(self._reset_all_viewports); head.addWidget(b_reset)
        vp_lay.addLayout(head)
        vp = Viewport3D(); vp.setFixedHeight(420); vp_lay.addWidget(vp)
        self.viewports.append(vp); lower.addWidget(vp_card, stretch=6)

        adv, al = card("详细数据（旋转矩阵 / 铰点坐标 / 摆角）", tint="view")
        self.fk_detail = QPlainTextEdit(); self.fk_detail.setReadOnly(True)
        al.addWidget(self.fk_detail)
        lower.addWidget(adv, stretch=4)
        right.addLayout(lower)
        root.addWidget(scroll, stretch=6)
        return page

    # ==================== 页面 3：逆解验证 ====================

    def _build_ik_tab(self):
        page = QWidget(); page.setObjectName("page"); root = QHBoxLayout(page)
        root.setContentsMargins(22, 18, 22, 20); root.setSpacing(16)
        left = QVBoxLayout()
        c1, l1 = card("输入目标位姿（相对零位的偏移量）", tint="input")
        self.ik_edits = []
        # 3列×2行输入卡片网格，排版与正解结果/动作范围卡片一致
        ik_grid = QGridLayout(); ik_grid.setSpacing(9)
        for idx, (name, key, unit) in enumerate([
                ("X 纵荡", "ik_x", "mm"), ("Y 横荡", "ik_y", "mm"),
                ("Z 升沉", "ik_z", "mm"), ("Roll 横滚", "ik_roll", "°"),
                ("Pitch 俯仰", "ik_pitch", "°"), ("Yaw 偏航", "ik_yaw", "°")]):
            cell = QFrame(); cell.setObjectName("card")
            cl = QVBoxLayout(cell); cl.setContentsMargins(12, 10, 12, 10); cl.setSpacing(5)
            head = QHBoxLayout(); head.setSpacing(5)
            head.addWidget(label(name, "caption")); head.addWidget(make_hint_button(key)); head.addStretch()
            cl.addLayout(head)
            e = num_edit("0", 100); self.ik_edits.append(e)
            er = QHBoxLayout(); er.addWidget(e); er.addWidget(label(unit, "unit")); er.addStretch()
            cl.addLayout(er)
            ik_grid.addWidget(cell, idx // 3, idx % 3)
        l1.addLayout(ik_grid)
        # 调试模式：回填上一次成功逆解的目标位姿偏移（相对零位；仅调试可见）
        self.b_restore_ik_pose = QPushButton("↶ 回填上一帧位姿")
        self.b_restore_ik_pose.setObjectName("ghost")
        self.b_restore_ik_pose.setToolTip(
            "把上一次成功逆解时的 6 个目标位姿偏移量（相对零位）填回上方输入框，"
            "便于在两帧之间来回微调；仅调试模式显示。")
        self.b_restore_ik_pose.clicked.connect(self.restore_prev_ik_pose)
        self.b_restore_ik_pose.setVisible(False)
        row_restore = QHBoxLayout(); row_restore.addStretch()
        row_restore.addWidget(self.b_restore_ik_pose)
        l1.addLayout(row_restore)
        # 调试模式：采样间隔 dt（用于逆解转速；仅调试可见）
        self.ik_debug_dt = QWidget()
        dt_lay = QVBoxLayout(self.ik_debug_dt)
        dt_lay.setContentsMargins(0, 6, 0, 0); dt_lay.setSpacing(0)
        self.ik_dt = num_edit("0", 90)
        dt_lay.addLayout(input_row("采样间隔 dt (s)", "kin_dt", self.ik_dt))
        dt_lay.addWidget(make_hint_button("ik_dyn_cache"))
        self.ik_debug_dt.setToolTip("填 > 0 可输出所需转速（调试模式）")
        self.ik_debug_dt.setVisible(False)
        l1.addWidget(self.ik_debug_dt)

        # 四元数输入/输出（[w,x,y,z]，与 ROS/Unity 交换姿态用；始终显示）
        qrow = QHBoxLayout()
        qrow.addWidget(label("四元数 w,x,y,z", "caption"))
        qrow.addWidget(make_hint_button("ik_quaternion"))
        self.ik_quat_edit = QLineEdit("1, 0, 0, 0")
        self.ik_quat_edit.setFixedWidth(200)
        self.ik_quat_edit.setToolTip("输入 [w,x,y,z] 后点『应用』转成 Roll/Pitch/Yaw 填入上方目标位姿")
        qrow.addWidget(self.ik_quat_edit)
        bq = QPushButton("应用"); bq.setObjectName("ghost"); bq.clicked.connect(self.apply_ik_quaternion)
        qrow.addWidget(bq); qrow.addStretch(); l1.addLayout(qrow)
        self.ik_quat_out = label("当前姿态四元数：—", "caption", wrap=True)
        l1.addWidget(self.ik_quat_out)

        b = QPushButton("计算所需缸长  →"); b.setObjectName("primary"); b.clicked.connect(self.run_ik)
        l1.addWidget(b); left.addWidget(c1)

        # 结果 + 精度验证也放左侧（数据栏全部在左）
        c2, l2 = card("逆解结果（6 缸目标长度）", tint="output")
        self.ik_table = QTableWidget(6, 5)
        self.ik_table.setHorizontalHeaderLabels(
            ["作动器", "绝对长度 (mm)", "相对零位 ΔL (mm)", "电机圈数 (rev)", "所需转速 (rpm)"])
        self.ik_table.setVerticalHeaderLabels([f"缸 {i+1}" for i in range(6)])
        self.ik_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ik_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.ik_table.setFixedHeight(170); l2.addWidget(self.ik_table)
        self.ik_status = label("", "caption", wrap=True); l2.addWidget(self.ik_status)
        self.ik_joint_status = label("", "caption", wrap=True); l2.addWidget(self.ik_joint_status)
        self.ik_limit_status = QLabel(""); self.ik_limit_status.setObjectName("caption")
        self.ik_limit_status.setWordWrap(True); self.ik_limit_status.setTextFormat(Qt.RichText)
        self.ik_limit_status.linkActivated.connect(self._on_limit_link)
        l2.addWidget(self.ik_limit_status)
        iqt = QHBoxLayout(); iqt.addStretch(); iqt.addWidget(make_hint_button("out_motor_torque"))
        l2.addLayout(iqt)
        # 下发到开发板（与正解同款，受调试模式显隐控制）
        self.b_send_device_ik = QPushButton("下发到开发板")
        self.b_send_device_ik.setObjectName("primary")
        self.b_send_device_ik.setToolTip(
            "把当前逆解的 6 路目标缸长下发到开发板（需先开启调试模式并计算逆解）")
        self.b_send_device_ik.clicked.connect(self._send_to_device_ik)
        row_send_ik = QHBoxLayout(); row_send_ik.addStretch()
        row_send_ik.addWidget(self.b_send_device_ik)
        l2.addLayout(row_send_ik)
        left.addWidget(c2)

        c4, l4 = card("往返精度验证", tint="output")
        self.verify_label = label("先计算逆解，再将缸长送回正解验证一致性。", "caption", wrap=True)
        l4.addWidget(self.verify_label)
        bv = QPushButton("送回正解验证"); bv.clicked.connect(self.verify_ik); l4.addWidget(bv)
        left.addWidget(c4); left.addStretch()

        # 左侧可滚动，容纳全部数据栏
        scroll_l = QScrollArea(); scroll_l.setWidgetResizable(True); scroll_l.setFrameShape(QFrame.NoFrame)
        lw = QWidget(); scroll_l.setWidget(lw); lw.setLayout(left)
        root.addWidget(scroll_l, stretch=5)

        # 右侧只有三维预览
        right = QVBoxLayout()
        c3, l3 = card("目标姿态三维预览", tint="view")
        vp = Viewport3D(); vp.setFixedHeight(420); l3.addWidget(vp)
        self.viewports.append(vp); right.addWidget(c3); right.addStretch()
        root.addLayout(right, stretch=4)
        return page

    # ==================== 页面 4：三维姿态 ====================

    def _build_view_tab(self):
        page = QWidget(); page.setObjectName("page"); lay = QVBoxLayout(page)
        lay.setContentsMargins(22, 18, 22, 20); lay.setSpacing(12)
        bar = QFrame(); bar.setObjectName("card")
        bl = QHBoxLayout(bar); bl.setContentsMargins(12, 8, 12, 8); bl.setSpacing(8)
        bl.addWidget(label("视图控制", "headline"))
        b_reset = QPushButton("重置视角"); b_reset.clicked.connect(self._reset_all_viewports)
        self.btn_grid = QPushButton("地面网格"); self.btn_grid.setCheckable(True); self.btn_grid.setChecked(True)
        self.btn_grid.toggled.connect(self._toggle_grid)
        self.btn_home = QPushButton("零位参考"); self.btn_home.setCheckable(True); self.btn_home.setChecked(True)
        self.btn_home.toggled.connect(self._toggle_home)
        self.btn_axis = QPushButton("坐标轴"); self.btn_axis.setCheckable(True); self.btn_axis.setChecked(True)
        self.btn_axis.toggled.connect(self._toggle_axis)
        self.btn_auto = QPushButton("自动旋转"); self.btn_auto.setCheckable(True)
        self.btn_auto.toggled.connect(self._toggle_auto)
        self.btn_proj = QPushButton("正交等比"); self.btn_proj.setCheckable(True)
        self.btn_proj.setChecked(True); self.btn_proj.setToolTip("正交等比保证角度/高度真实比例；关闭为透视")
        self.btn_proj.toggled.connect(self._toggle_projection)
        self.btn_coords = QPushButton("点位坐标"); self.btn_coords.setCheckable(True)
        self.btn_coords.setToolTip("在每个铰点旁显示其世界坐标 (X,Y,Z)")
        self.btn_coords.toggled.connect(self._toggle_coords)
        self.btn_drag_pose = QPushButton("拖拽平台"); self.btn_drag_pose.setCheckable(True)
        self.btn_drag_pose.setObjectName("segment")   # 选中态随主题高亮
        self.btn_drag_pose.setToolTip("开启后左键拖动上平台改变 X/Y 位姿（Shift+拖动改 Z），实时反解缸长并钳制限位")
        self.btn_drag_pose.toggled.connect(self._toggle_drag_pose)
        b_reset_pose = QPushButton("重置位置")
        b_reset_pose.setToolTip("把上平台位姿恢复到零位（Z=H0、其余为 0），并刷新全部视口")
        b_reset_pose.clicked.connect(self._reset_pose_to_home)
        self.btn_ellipsoid = QPushButton("力椭球"); self.btn_ellipsoid.setCheckable(True)
        self.btn_ellipsoid.setObjectName("segment")
        self.btn_ellipsoid.setToolTip("在视口显示等缸力下的平台响应椭球；数值见平台参数页性能诊断")
        self.btn_ellipsoid.toggled.connect(self._toggle_force_ellipsoid)
        self.chk_ellipsoid = self.btn_ellipsoid   # 兼容旧属性名
        for b in (b_reset, b_reset_pose, self.btn_grid, self.btn_home, self.btn_axis,
                  self.btn_auto, self.btn_proj, self.btn_coords, self.btn_drag_pose):
            bl.addWidget(b)
        self.btn_cloud = QPushButton("点云"); self.btn_cloud.setCheckable(True)
        self.btn_cloud.setObjectName("segment")
        self.btn_cloud.setToolTip("在主视口叠加工作空间点云；采样后自动聚焦点云与机构")
        self.btn_cloud.toggled.connect(self._toggle_pointcloud)
        bl.addWidget(make_hint_button("vp_drag_pose"))
        bl.addWidget(make_hint_button("perf_ellipsoid"))
        bl.addWidget(self.btn_ellipsoid)
        bl.addWidget(self.btn_cloud)
        bl.addStretch(); lay.addWidget(bar)

        vp = Viewport3D(); lay.addWidget(vp, stretch=1)
        self.viewports.append(vp)
        self.main_viewport = vp
        vp.pose_dragged.connect(self._on_pose_dragged)

        pose_card, pl = card("当前显示位姿（相对零位）", tint="output")
        self.view_pose_labels = {}
        grid = QHBoxLayout(); grid.setSpacing(10)
        for i, (name, key, unit) in enumerate([
            ("ΔX", "out_x", "mm"), ("ΔY", "out_y", "mm"), ("ΔZ", "out_z", "mm"),
            ("Roll", "out_roll", "°"), ("Pitch", "out_pitch", "°"), ("Yaw", "out_yaw", "°")]):
            cell = QFrame(); cell.setObjectName("card")
            cl = QVBoxLayout(cell); cl.setContentsMargins(10, 8, 10, 8); cl.setSpacing(2)
            head = QHBoxLayout(); head.addWidget(label(name, "caption"))
            head.addWidget(make_hint_button(key)); head.addStretch(); cl.addLayout(head)
            v = label("—", "outValue"); cl.addWidget(v)
            cl.addWidget(label(unit, "unit"))
            self.view_pose_labels[i] = v
            grid.addWidget(cell)
        pl.addLayout(grid)
        # 工作空间点云控制条：复用 pose_card，主视口仅让出约 60px
        ws_row = QHBoxLayout(); ws_row.setSpacing(8)
        b_sample = QPushButton("工作空间采样"); b_sample.setObjectName("primary")
        b_sample.clicked.connect(self.sample_workspace)
        b_png = QPushButton("导出 PNG"); b_png.clicked.connect(self.export_workspace_png)
        b_csv = QPushButton("导出 CSV"); b_csv.clicked.connect(self.export_workspace_csv)
        ws_row.addWidget(b_sample); ws_row.addWidget(b_png); ws_row.addWidget(b_csv)
        # 状态与问号同一行，避免多占一行挤压主视口
        self.ws_status = label("未采样。", "caption", wrap=True)
        ws_row.addWidget(self.ws_status, stretch=1)
        ws_row.addWidget(make_hint_button("ws_sampling"))
        pl.addLayout(ws_row)
        lay.addWidget(pose_card)
        self.view_drag_status = label("", "caption", wrap=True)
        lay.addWidget(self.view_drag_status)
        return page

    def _reset_all_viewports(self):
        for v in self.viewports:
            v.reset_view()

    def _reset_pose_to_home(self):
        """把上平台位姿恢复到零位，并同步缸长、视口与状态。"""
        try:
            home = np.asarray(self.config.home_pose(), dtype=float).reshape(-1)[:6].copy()
            # 清除主视口遗留的拖拽基点，避免下次拖动从旧位置跳变
            vp = getattr(self, "main_viewport", None)
            if vp is not None:
                vp._drag_pose_start = None
                vp._drag_pose_base = None
                vp.set_pose(home)
            res = self.solver.inverse(home)
            self.last_ik = res
            self._sync_viewports(res.pose)
            for i in range(3):
                self.view_pose_labels[i].setText(f"{home[i] - self.solver.home_pose()[i]:+.2f}")
            self.view_drag_status.setText("已重置位置：上平台回到零位（Z=H0，其余为 0）。")
        except Exception as ex:
            self.view_drag_status.setText(f"重置位置失败：{ex}")

    def _toggle_drag_pose(self, checked):
        """拖拽平台模式：只对主视口生效，避免多个视口同时发位姿。"""
        for v in self.viewports:
            v.set_drag_pose_mode(checked and v is self.main_viewport)

    def _clamp_pose_to_limits(self, base, target, iters=14):
        """在合法 base 与越界 target 之间二分，返回最远合法位姿（避免拖不动无反馈）。"""
        base = np.asarray(base, dtype=float).reshape(-1)
        target = np.asarray(target, dtype=float).reshape(-1)
        if bool(self.solver.pose_limits_ok(target)):
            return target
        if not bool(self.solver.pose_limits_ok(base)):
            return base
        lo, hi = 0.0, 1.0
        for _ in range(iters):
            mid = (lo + hi) * 0.5
            if bool(self.solver.pose_limits_ok(base + (target - base) * mid)):
                lo = mid
            else:
                hi = mid
        return base + (target - base) * lo

    def _on_pose_dragged(self, pose):
        """拖拽上平台 → 限位钳制 → 逆解缸长 → 回写视口与状态。"""
        try:
            q = np.asarray(pose, dtype=float).reshape(-1)[:6].copy()
            base = getattr(getattr(self, "last_ik", None), "pose", None)
            if base is None:
                base = self.config.home_pose()
            q = self._clamp_pose_to_limits(base, q)
            self.main_viewport.set_pose(q)
            res = self.solver.inverse(q)
            self.last_ik = res
            self._sync_viewports(res.pose)
            home = self.config.home_pose()
            self.view_pose_labels[0].setText(f"{q[0] - home[0]:+.2f}")
            self.view_pose_labels[1].setText(f"{q[1] - home[1]:+.2f}")
            self.view_pose_labels[2].setText(f"{q[2] - home[2]:+.2f}")
            self.view_drag_status.setText("拖拽中：位姿已按行程/摆角限位钳制，缸长实时反解。")
        except Exception as ex:
            self.view_drag_status.setText(f"拖拽失败：{ex}")

    def _toggle_grid(self, checked):
        for v in self.viewports: v.show_grid = checked; v.update()

    def _toggle_home(self, checked):
        for v in self.viewports: v.show_home = checked; v.update()

    def _toggle_axis(self, checked):
        for v in self.viewports: v.show_axis = checked; v.update()

    def _toggle_auto(self, checked):
        for v in self.viewports: v.set_auto_rotate(checked)

    def _toggle_coords(self, checked):
        for v in self.viewports:
            v.show_coords = checked
            v.update()

    def _toggle_projection(self, checked):
        self.btn_proj.setText("正交等比" if checked else "透视投影")
        for v in self.viewports:
            v.orthographic = checked
            v.update()

    def _sync_viewports(self, pose):
        delta = np.asarray(pose, float) - self.solver.home_pose()
        for i, v in enumerate(self.view_pose_labels):
            self.view_pose_labels[i].setText(f"{delta[i]:+.3f}")
        for vp in self.viewports:
            vp.set_pose(pose)

    # ==================== 页面 5：批量计算 ====================

    def _build_kinematics_tab(self):
        """运动学计算：自定义输入位姿帧，输出速度 / 角速度 / 加速度。"""
        page = QWidget(); page.setObjectName("page")
        root = QVBoxLayout(page)
        root.setContentsMargins(22, 18, 22, 20); root.setSpacing(14)

        head = QHBoxLayout()
        hb = QVBoxLayout(); hb.setSpacing(3)
        hb.addWidget(label("运动学计算", "largeTitle"))
        hb.addWidget(label("输入三帧位姿与采样间隔，计算线速度、角速度、加速度与缸速", "caption"))
        head.addLayout(hb); head.addStretch()
        root.addLayout(head)

        body = QHBoxLayout(); body.setSpacing(16)

        # ---- 左：输入 ----
        left = QVBoxLayout(); left.setSpacing(12)
        inp, il = card("输入位姿帧", tint="input")
        il.addWidget(label("按 X, Y, Z, Roll, Pitch, Yaw 顺序，逗号或空格分隔", "caption"))
        self.kin_frame_edits = []
        frame_defs = [
            ("帧 k−2（最早）", "0, 0, 0, 0, 0, 0"),
            ("帧 k−1（中间）", "0, 0, 0, 0, 0, 0"),
            ("帧 k（当前）  ", "0, 0, 0, 0, 0, 0"),
        ]
        for text, ph in frame_defs:
            e = QLineEdit(); e.setText(ph); e.setPlaceholderText(ph)
            il.addLayout(input_row(text, "kin_frame", e))
            self.kin_frame_edits.append(e)
        self.kin_dt = num_edit("0.02", 100)
        il.addLayout(input_row("采样间隔 dt (s)", "kin_dt", self.kin_dt))
        il.addWidget(label("速度需 2 帧（k−1、k）；加速度需 3 帧。dt 必须 > 0。", "caption", wrap=True))
        b_calc = QPushButton("计算速度与加速度  →"); b_calc.setObjectName("primary")
        b_calc.clicked.connect(self.run_kinematics); il.addWidget(b_calc)
        self.kin_status = label("尚未计算", "caption", wrap=True); il.addWidget(self.kin_status)
        left.addWidget(inp); left.addStretch()

        # ---- 右：输出（可滚动；调试模式内容多时不压缩行高）----
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        rw = QWidget(); scroll.setWidget(rw)
        right = QVBoxLayout(rw)
        right.setContentsMargins(2, 2, 8, 2); right.setSpacing(12)
        out, ol = card("计算结果", tint="output")
        # 速度表（6 行）
        vh = QHBoxLayout(); vh.addWidget(label("线速度 / 角速度", "headline"))
        vh.addWidget(make_hint_button("kin_vel")); vh.addStretch(); ol.addLayout(vh)
        self.kin_vel_labels = []
        for name in ("X 纵荡 (mm/s)", "Y 横荡 (mm/s)", "Z 升沉 (mm/s)",
                     "Roll (°/s)", "Pitch (°/s)", "Yaw (°/s)"):
            row = QHBoxLayout()
            lab = label(name, "caption"); v = label("—", "inValue")
            row.addWidget(lab); row.addStretch(); row.addWidget(v)
            ol.addLayout(row); self.kin_vel_labels.append(v)
        # 加速度表
        ah = QHBoxLayout(); ah.addSpacing(2)
        ah.addWidget(label("线加速度 / 角加速度", "headline"))
        ah.addWidget(make_hint_button("kin_acc")); ah.addStretch(); ol.addLayout(ah)
        self.kin_acc_labels = []
        for name in ("X (mm/s²)", "Y (mm/s²)", "Z (mm/s²)",
                     "Roll (°/s²)", "Pitch (°/s²)", "Yaw (°/s²)"):
            row = QHBoxLayout()
            lab = label(name, "caption"); v = label("—", "inValue")
            row.addWidget(lab); row.addStretch(); row.addWidget(v)
            ol.addLayout(row); self.kin_acc_labels.append(v)
        # 缸速
        lh = QHBoxLayout(); lh.addWidget(label("各缸线速度", "headline"))
        lh.addWidget(make_hint_button("kin_legvel")); lh.addStretch(); ol.addLayout(lh)
        self.kin_leg_labels = []
        for i in range(6):
            row = QHBoxLayout()
            lab = label(f"缸 {i+1} (mm/s)", "caption"); v = label("—", "inValue")
            row.addWidget(lab); row.addStretch(); row.addWidget(v)
            ol.addLayout(row); self.kin_leg_labels.append(v)
        # 调试模式：电机圈数 / 转速（由导程换算）
        self.kin_debug_box = QFrame(); self.kin_debug_box.setObjectName("card")
        kdl = QVBoxLayout(self.kin_debug_box)
        dh = QHBoxLayout(); dh.addWidget(label("电机派生（调试模式）", "headline"))
        dh.addWidget(make_hint_button("dbg_revolutions")); dh.addStretch(); kdl.addLayout(dh)
        self.kin_rev_labels = []
        for i in range(6):
            row = QHBoxLayout()
            lab = label(f"缸 {i+1} 圈数 / 转速", "caption"); v = label("—", "outValue")
            row.addWidget(lab); row.addStretch(); row.addWidget(v)
            kdl.addLayout(row); self.kin_rev_labels.append(v)
        ol.addWidget(self.kin_debug_box)
        right.addWidget(out)

        body.addLayout(left, stretch=4); body.addWidget(scroll, stretch=6)
        root.addLayout(body)
        return page

    def run_kinematics(self):
        """由自定义位姿帧计算速度/角速度/加速度/缸速。"""
        try:
            dt = float(self.kin_dt.text())
            if dt <= 0:
                raise ValueError("采样间隔 dt 必须大于 0")
            frames = []
            for i, e in enumerate(self.kin_frame_edits):
                vals = parse_series(e.text(), 6)
                frames.append(np.asarray(vals, float))
            q0, q1, q2 = frames

            # 速度：用最后两帧
            vel = (q2 - q1) / dt
            # 加速度：三帧二阶差分
            acc = (q2 - 2 * q1 + q0) / (dt * dt)

            for i, lab in enumerate(self.kin_vel_labels):
                lab.setText(f"{vel[i]:+.3f}")
            for i, lab in enumerate(self.kin_acc_labels):
                lab.setText(f"{acc[i]:+.3f}")

            # 各缸线速度：用当前帧 q2 的雅可比，角速度转 rad/s
            j = self.solver.jacobian(q2)
            vel_rad = np.concatenate([vel[:3], np.radians(vel[3:])])
            leg_vel = j @ vel_rad
            # 超速校核：一律对照电缸管理的最大线速度（peak → rated → 旧别名）
            speed_cap = effective_speed_cap(self.config)
            self._kin_speed_over = (
                [i + 1 for i in range(6) if abs(leg_vel[i]) > speed_cap + 1e-6]
                if speed_cap > 0 else [])
            for i, lab in enumerate(self.kin_leg_labels):
                lab.setText(f"{leg_vel[i]:+.3f}" + (" ⚠" if (i + 1) in self._kin_speed_over else ""))

            # 调试模式：电机圈数 = 位移/导程，转速 = 60·|缸速|/导程
            lead = self._motor_lead()
            dL = self.solver.leg_lengths(q2) - self.solver.ref_lengths
            revs, rpms = motor_revolutions(dL, lead, leg_vel)
            if revs is None:
                for lab in self.kin_rev_labels:
                    lab.setText("导程未填")
            else:
                for i, lab in enumerate(self.kin_rev_labels):
                    lab.setText(f"{revs[i]:+.2f} rev / {rpms[i]:.0f} rpm")

            warn = (f"　⚠ 超速：缸 {','.join(map(str, self._kin_speed_over))}"
                    f" 超过最大线速度 {speed_cap:g} mm/s") if self._kin_speed_over else ""
            self.kin_status.setText(
                f"✓ 已计算 · dt={dt:g}s · 线速度 {np.linalg.norm(vel[:3]):.2f}mm/s · "
                f"角速度 {np.linalg.norm(vel[3:]):.2f}°/s{warn}")
            self._kin_ok = True
        except Exception as ex:
            self.kin_status.setText(f"✗ 计算失败：{ex}")
            self._kin_ok = False
            self._kin_speed_over = []
        self.apply_theme(self.app_ref.theme_mgr.theme)
        # 超速缸标红（apply_theme 之后设置，避免被统一配色覆盖）
        for i, lab in enumerate(self.kin_leg_labels):
            if (i + 1) in getattr(self, "_kin_speed_over", []):
                lab.setStyleSheet("color: #e57373; font-weight: bold;")
            else:
                lab.setStyleSheet("")

    def _build_batch_tab(self):
        page = QWidget(); page.setObjectName("page"); root = QVBoxLayout(page)
        root.setContentsMargins(22, 18, 22, 20); root.setSpacing(12)
        c, l = card("批量姿态解算", tint="input")
        head = QHBoxLayout(); head.addWidget(label("每行 6 个缸长 (mm)，可粘贴或导入 CSV/TXT"))
        head.addWidget(make_hint_button("fk_leg_i")); head.addStretch(); l.addLayout(head)
        self.batch_input = QPlainTextEdit()
        self.batch_input.setPlaceholderText("645.5, 634.4, 634.7, 645.7, 604.6, 605.0\n620.1 620.2 620.3 620.0 619.8 619.9")
        self.batch_input.setFixedHeight(88); l.addWidget(self.batch_input)
        br = QHBoxLayout()
        b1 = QPushButton("导入数据文件"); b1.clicked.connect(self.import_batch_file); br.addWidget(b1)
        b2 = QPushButton("批量求解"); b2.setObjectName("primary"); b2.clicked.connect(self.run_batch); br.addWidget(b2)
        b3 = QPushButton("导出结果 CSV"); b3.clicked.connect(self.export_batch_csv); br.addWidget(b3)
        br.addWidget(make_hint_button("batch_async"))
        br.addStretch(); l.addLayout(br); root.addWidget(c)

        self.batch_table = QTableWidget()
        cols = ["序号", "ΔX(mm)", "ΔY(mm)", "ΔZ(mm)", "ΔRoll(°)", "ΔPitch(°)", "ΔYaw(°)",
                "最大残差(mm)", "铰点校验", "状态"]
        self.batch_table.setColumnCount(len(cols)); self.batch_table.setHorizontalHeaderLabels(cols)
        self.batch_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.batch_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        root.addWidget(self.batch_table, stretch=1)
        self.batch_progress = QProgressBar(); self.batch_progress.setRange(0, 100)
        self.batch_progress.setVisible(False); root.addWidget(self.batch_progress)
        self.batch_status = label("", "caption"); root.addWidget(self.batch_status)
        return page

    # ==================== 页面 8：轨迹播放 ====================

    def _build_trajectory_tab(self):
        """CSV 轨迹：导入 pose/length 序列，逐帧 IK/FK 并可下发到开发板。"""
        from collections import deque
        self._traj_frames = deque()
        self._traj_total = 0
        self._traj_index = 0
        self._traj_timer = QTimer(self)
        self._traj_timer.timeout.connect(self._traj_next_frame)

        # 三张卡纵向堆叠会超过可见高度：整页加滚动区，并给表格/点动卡设最小高度防压扁
        page = QWidget(); page.setObjectName("page")
        outer = QVBoxLayout(page); outer.setContentsMargins(0, 0, 0, 0)
        traj_scroll = QScrollArea(); traj_scroll.setWidgetResizable(True)
        traj_scroll.setFrameShape(QFrame.NoFrame)
        host = QWidget(); host.setObjectName("page")
        root = QVBoxLayout(host)
        root.setContentsMargins(22, 18, 22, 20); root.setSpacing(12)

        c, l = card("轨迹输入", tint="input")
        head = QHBoxLayout()
        head.addWidget(label("每行 6 列：pose 模式 [X,Y,Z,Roll,Pitch,Yaw]（相对零位）或缸长模式 L1~L6；"
                             "可选第 7 列 dwell(ms)", wrap=True))
        head.addWidget(make_hint_button("traj_play")); head.addStretch(); l.addLayout(head)
        mode_row = QHBoxLayout()
        mode_row.addWidget(label("输入类型", "caption"))
        self.traj_mode = QComboBox()
        self.traj_mode.addItems(["位姿轨迹（→ 逆解求缸长）", "缸长轨迹（直接下发）"])
        mode_row.addWidget(self.traj_mode); mode_row.addStretch(); l.addLayout(mode_row)
        self.traj_input = QPlainTextEdit()
        self.traj_input.setPlaceholderText("10, 0, 5, 0, 0, 0, 100\n20, 0, 5, 0, 0, 0, 100\n30, 0, 5, 0, 0, 0")
        self.traj_input.setFixedHeight(130); l.addWidget(self.traj_input)
        tr = QHBoxLayout()
        b_load = QPushButton("导入 CSV"); b_load.clicked.connect(self.import_trajectory)
        b_parse = QPushButton("解析轨迹"); b_parse.clicked.connect(self.parse_trajectory)
        b_play = QPushButton("播放"); b_play.setObjectName("primary"); b_play.clicked.connect(self.play_trajectory)
        b_pause = QPushButton("暂停"); b_pause.clicked.connect(self.pause_trajectory)
        b_stop = QPushButton("停止"); b_stop.clicked.connect(self.stop_trajectory)
        for b in (b_load, b_parse, b_play, b_pause, b_stop):
            tr.addWidget(b)
        tr.addStretch(); l.addLayout(tr)
        root.addWidget(c)

        c2, l2 = card("播放控制", tint="output")
        self.traj_progress = QProgressBar(); self.traj_progress.setRange(0, 100)
        l2.addWidget(self.traj_progress)
        self.traj_status = label("尚未导入轨迹。", "caption", wrap=True); l2.addWidget(self.traj_status)
        self.traj_table = QTableWidget(0, 8)
        self.traj_table.setHorizontalHeaderLabels(
            ["帧", "X/Y/Z(mm)", "Roll/Pitch/Yaw(°)", "L1~L6(mm)", "缸速(mm/s)", "限位", "dwell(ms)", "状态"])
        self.traj_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.traj_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.traj_table.setMinimumHeight(260)
        l2.addWidget(self.traj_table)
        root.addWidget(c2)

        # 点动控制：六向 +/- 小步移动 → 逆解 → 下发（实机微调）
        cj, lj = card("点动控制（Jog）", tint="input")
        cj.setMinimumHeight(250)
        jh = QHBoxLayout()
        jh.addWidget(label("步长", "caption"))
        jh.addWidget(make_hint_button("jog_control"))
        self.jog_step = num_edit("5", 80); jh.addWidget(self.jog_step)
        jh.addWidget(label("mm / °", "unit"))
        self.jog_mode = QComboBox(); self.jog_mode.addItems(["相对当前位姿", "相对零位"])
        jh.addWidget(self.jog_mode)
        jh.addStretch(); lj.addLayout(jh)
        self.jog_pose = None  # 当前点动基准位姿（绝对）
        for i, name in enumerate(("X", "Y", "Z", "Roll", "Pitch", "Yaw")):
            row = QHBoxLayout(); row.addWidget(label(name, "caption")); row.addStretch()
            for sign in (-1, 1):
                b = QPushButton(f"{'+' if sign>0 else '−'}{name}")
                b.setObjectName("ghost"); b.setFixedWidth(80)
                b.clicked.connect(lambda checked=False, d=i, s=sign: self.jog_move(d, s))
                row.addWidget(b)
            lj.addLayout(row)
        self.jog_status = label("基准取上次逆解/正解结果；点按钮小步移动并下发（需调试模式）。",
                                "caption", wrap=True)
        lj.addWidget(self.jog_status)
        root.addWidget(cj)
        outer.addWidget(traj_scroll); traj_scroll.setWidget(host)
        return page

    # ==================== 页面 9：使用说明 ====================

    def _build_help_tab(self):
        page = QWidget(); page.setObjectName("page"); lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget(); scroll.setWidget(inner)
        il = QVBoxLayout(inner); il.setContentsMargins(24, 20, 24, 24)
        self.help_label = QLabel(); self.help_label.setWordWrap(True)
        self.help_label.setTextFormat(Qt.RichText); self.help_label.setOpenExternalLinks(True)
        il.addWidget(self.help_label); il.addStretch()
        lay.addWidget(scroll)
        self._refresh_help_html(self.app_ref.theme_mgr.theme)
        return page

    def _refresh_help_html(self, t):
        self.help_label.setText(f"""
        <style>
          body {{ font-family:'PingFang SC','Microsoft YaHei'; font-size:13px;
                  color:{t.text}; line-height:1.85; }}
          h2 {{ color:{t.accent}; font-size:17px; margin:20px 0 8px 0; }}
          h2:first-child {{ margin-top:0; }}
          li {{ margin:4px 0; }}
          .tip {{ background:{t.surface}; border:1px solid {t.border};
                  border-radius:10px; padding:12px 14px; }}
          .repo-card {{ background:{t.surface}; border:1px solid {t.accent};
                  border-radius:10px; padding:12px 14px; margin-bottom:10px; }}
          .warn {{ background:{t.surface_alt}; border:1px solid {t.warning};
                   border-radius:10px; padding:12px 14px; }}
          table {{ border-collapse:collapse; width:100%; }}
          td {{ padding:7px 9px; border-bottom:1px solid {t.border}; }}
          b {{ color:{t.output}; }}
          .in {{ color:{t.input}; font-weight:700; }}
          .out {{ color:{t.output}; font-weight:700; }}
        </style>
        <div class="repo-card">
          <b>项目仓库 · GitHub</b><br>
          <a href="{REPO_URL}" style="color:#5FA8D3;">{REPO_URL}</a><br>
          <span>源码、问题反馈与版本更新均在此，点击链接在浏览器打开。</span>
        </div>
        <h2 style="margin-top:0;">快速上手</h2>
        <ol>
          <li>启动后从<b>档案中心</b>选择已有档案，或点「新建平台档案」命名建档。</li>
          <li>在「平台参数」页切换机构结构并填写实机几何数据，点<b>应用参数并校验</b>。</li>
          <li>在「姿态正解」页输入 6 路缸长，点<b>计算姿态</b>得到 X/Y/Z 与 Roll/Pitch/Yaw。</li>
          <li>「三维姿态」页可拖拽旋转、滚轮缩放查看平台运动。</li>
        </ol>
        <h2>问号帮助与配色约定</h2>
        <div class="tip">
          每个数据旁边的 <b>?</b> 都可点击：<br>
          • <span class="in">输入数据（蓝色）</span>的问号 → 弹出平台框线图并高亮该数据的部位；<br>
          • <span class="out">输出数据（绿色）</span>的问号 → 弹出该指标的计算公式与推导；<br>
          • 同时具有两者的项目用页签切换「模型位置 / 计算公式」。
        </div>
        <h2>机构结构说明</h2>
        <table>
          <tr><td><b>SPS</b></td><td>两端球铰，无摆角限制。</td></tr>
          <tr><td><b>UPS</b></td><td>下虎克铰 + 上球铰，校验下铰摆角。</td></tr>
          <tr><td><b>SPU</b></td><td><b>你的平台</b>：下虎克铰带竖直旋转轴（等效球铰 S）+ 上虎克铰，校验上铰摆角。</td></tr>
          <tr><td><b>UPU</b></td><td>两端虎克铰，工作空间最小。</td></tr>
        </table>
        <p>四种结构的<b>长度方程完全相同</b>：L = ‖上铰点世界坐标 − 下铰点世界坐标‖。
        切换结构不改变位姿解，只启用对应的铰点摆角校验。</p>
        <h2>避免奇异配置</h2>
        <div class="warn">上下平台<b>成对夹角不能相同</b>（如都 30° 会导致雅可比秩亏、解不唯一）。
        推荐下 30° / 上 60°。软件会自动检测并在参数页警告。</div>
        <h2>电缸规格 · 限位 · 动态输出</h2>
        <div class="tip">
          • <b>电缸管理</b>：侧边栏『01 电缸管理』内置页，逐缸填名称/编码/品牌/行程/推力，
          并可设全局规格、负载质量与零位定义；数据自动同步到全软件，参数页不再重复显示。<br>
          • <b>限位</b>：缸长上下限、最大平移、最大倾角、节点摆角，超限会在结果里用橙色/红色警告，但不阻止解算。<br>
          • <b>动作范围</b>：参数页点『应用参数』后，会自动列出 X/Y/Z/Roll/Pitch/Yaw 在行程约束下的可达范围。<br>
          • <b>速度/加速度</b>：侧边栏『运动学计算』页自定义输入三帧位姿与采样间隔 dt，
          即可输出线速度、角速度、加速度、角加速度与各缸速度（每项问号可看公式）。<br>
          • <b>推力</b>：需先填「负载质量 + 平台自重」，用雅可比转置把重力与惯性力映射到 6 根缸，
          给出各缸推力与整体合力；超过额定推力会告警。<br>
          • <b>导出全部数据</b>：顶栏按钮可一次性导出档案参数、电缸规格、限位、输入输出、
          正逆解、动态量与批量结果为 JSON 或 CSV。
        </div>
        <h2>参考项目与算法来源</h2>
        <p>本项目独立实现算法思路，参考下列开源项目（点击名称在浏览器打开）。</p>
        <b>① 同类软件 / 上位机</b>
        <ul>
          <li><a href="https://github.com/nairck/Hexapod-Inverse-Kinematics-and-Workspace-Calculator" style="color:#5FA8D3;">nairck/Hexapod IK &amp; Workspace Calculator</a> — MIT，Win/Linux 免安装包，IK + 可达/定向工作空间绘图 + PNG 导出 + 增量调节表（不含正解）</li>
          <li><a href="https://github.com/stewart-platform-control/stewart_platform_control" style="color:#5FA8D3;">stewart_platform_control</a> — CustomTkinter + Matplotlib GUI，手动控制 / 实时 IK / CSV 轨迹 / 限位保护 / 3D 可视化（硬件接口为 mock）</li>
        </ul>
        <b>② 算法库（可移植/嵌入式）</b>
        <ul>
          <li><a href="https://github.com/Yeok-c/Stewart_Py" style="color:#5FA8D3;">Yeok-c/Stewart_Py</a> — Python，calculate() 直接返回 6 舵机角，带逐步推导 Notebook</li>
          <li><a href="https://github.com/rawify/Stewart.js" style="color:#5FA8D3;">rawify/Stewart.js</a> — JS/MIT，IK 库，支持舵机/缸长、四元数、p5.js 可视化</li>
          <li><a href="https://github.com/conroy-cheers/hexapod_kinematics" style="color:#5FA8D3;">conroy-cheers/hexapod_kinematics</a> — 现代 C++（Eigen3），正解 + 逆解，源自 LinuxCNC genhexkins</li>
          <li><a href="https://github.com/duccuongvu/StewartPlatformForwardKinematics" style="color:#5FA8D3;">duccuongvu/StewartPlatformForwardKinematics</a> — 纯 C99 正解，LM / Trust Region 双求解器，可跑 STM32</li>
          <li><a href="https://github.com/Sherin-SEF-AI/robot-kinematics" style="color:#5FA8D3;">Sherin-SEF-AI/robot-kinematics</a> — MIT，通用机器人运动学库（含 Stewart、DH/URDF、奇异检测、workspace）</li>
        </ul>
        <b>③ 分析 / 性能指标</b>
        <ul>
          <li><a href="https://github.com/AsioOtus055/StewartPlatformClass_py" style="color:#5FA8D3;">AsioOtus055/StewartPlatformClass_py</a> — 工作空间、雅可比、奇异值、条件数、驱动力矩</li>
          <li><a href="https://github.com/Flamisell/StewartPlatformClass_py" style="color:#5FA8D3;">Flamisell/StewartPlatformClass_py</a> — SVD 可操作性、局部条件指标、力椭球、工作空间分析</li>
          <li><a href="https://github.com/ravichand13/Stewart-platform" style="color:#5FA8D3;">ravichand13/Stewart-platform</a> — 逆解几何</li>
          <li><a href="https://github.com/matthewmhong/stewart-platform" style="color:#5FA8D3;">matthewmhong/stewart-platform</a> — 运动学框架</li>
          <li><a href="https://github.com/martyushev/fkSGP" style="color:#5FA8D3;">martyushev/fkSGP</a> — 正解数值方法</li>
          <li><a href="https://github.com/coelacant1/StewartPlatformIK" style="color:#5FA8D3;">coelacant1/StewartPlatformIK</a> — C++17 Windows CLI，6 参数 IK</li>
        </ul>
        <b>④ 仿真 / 实机控制</b>
        <ul>
          <li><a href="https://github.com/wei-hsuan-cheng/stewart_platform_sim" style="color:#5FA8D3;">wei-hsuan-cheng/stewart_platform_sim</a> — ROS 2 + RViz2，IK + 微分运动学（缸速/雅可比）</li>
          <li><a href="https://github.com/s-ritwik/Parallel-Manipulator-Stewart-Platform-" style="color:#5FA8D3;">s-ritwik/Parallel-Manipulator-Stewart-Platform-</a> — ROS 2 Humble + Gazebo，IK 节点 + 正弦测试运动</li>
          <li><a href="https://github.com/garethnisbet/Robot" style="color:#5FA8D3;">garethnisbet/Robot</a> — Web 3D 可视化，Newton-Raphson 正解 + 拖拽平台 + 限位钳制</li>
          <li><a href="https://github.com/JD-Patten/stewart_platform" style="color:#5FA8D3;">JD-Patten/stewart_platform</a> — ROS2 + Arduino 串口控制实物</li>
          <li><a href="https://github.com/NicHub/stewart-platform-esp32" style="color:#5FA8D3;">NicHub/stewart-platform-esp32</a> — ESP32 硬件落地</li>
          <li><a href="https://github.com/daniel-s-ingram/stewart" style="color:#5FA8D3;">daniel-s-ingram/stewart</a> — Gazebo + ROS + PS4 手柄（经典参考，2019 后未更新）</li>
        </ul>
        <p>正解用阻尼最小二乘（SciPy TRF），逆解闭式。往返误差 &lt; 1e-10。</p>
        <h2>项目仓库</h2>
        <p><b>六自由度平台姿态演算</b>（MIT 开源）：
           <a href="{REPO_URL}" style="color:#5FA8D3;">{REPO_URL}</a><br>
           源码、问题反馈与版本更新均在此。</p>
        <h2>开源许可与版权声明</h2>
        <div class="tip">
          本项目（六自由度姿态解算）采用 <b>MIT License</b> 开源，可自由使用、修改与分发，
          但须保留版权声明与许可文本。第三方组件的版权与许可归属如下：<br><br>
          <table>
            <tr><td><b>本项目</b></td><td>MIT License © 2026 项目作者</td></tr>
            <tr><td><b>NumPy</b></td><td>BSD-3-Clause 等（NumPy Developers）</td></tr>
            <tr><td><b>SciPy</b></td><td>BSD-3-Clause（SciPy Developers）</td></tr>
            <tr><td><b>PySide6-Essentials / Shiboken6</b></td><td>LGPL-3.0 或 GPL-2.0/3.0（The Qt Company）</td></tr>
            <tr><td><b>PyInstaller</b></td><td>GPL-2.0 + 例外条款（仅打包工具，不进入运行时）</td></tr>
          </table><br>
          <b>通信驱动（在『设置 → 驱动方式』中切换）：</b><br>
          • <b>串口（COM + 心跳）</b>：默认方式，填串口号与波特率即可；未配置时为模拟模式。<br>
          • <b>EtherCAT（pysoem + CiA 402）</b>：在设置里选择后启用；需安装 pysoem 与 Npcap，
            并按网卡序号枚举从站，把 6 路缸长写入 6040/607A。未装库或总线上无从站时自动退回模拟模式。<br>
          • <b>pysoem</b>（EtherCAT 主站绑定）：MIT License（v1.0.2 起）© Synapticon / pysoem contributors。<br>
          • <b>Npcap</b>（Windows 网卡驱动，需单独安装）：Npcap License（BSD 衍生）© Nmap Project。<br>
          • <b>SOEM</b>（pysoem 底层 C 栈）：GPLv3 + 商业双许可，由 RT-Labs / OpenEtherCATsociety 维护；
          以 Python wheel 形式使用时按 pysoem 的 MIT 条款分发，不分发 SOEM 源码。
          个人调试不触发 GPL 的分发义务；如需商业闭源分发请自行核实许可并购买商业授权。<br>
          • <b>CiA 402 / EtherCAT / CANopen</b>：公开技术标准，由 CiA / ETG 发布，接口规范本身不涉及版权授权费用。<br><br>
          上述参考的 Stewart 平台 / 机器人学开源项目版权归各自作者所有，仅借鉴算法思路，未复制其源码；详见上方链接。
        </div>
        <h2>已知限制</h2>
        <ul>
          <li>面向直线缸 + 球铰/虎克铰的 Stewart 类平台。</li>
          <li>正解存在多解，默认从零位或上次解跟踪连续分支。</li>
          <li>纯运动学几何解，不含弹性变形与铰链间隙补偿。</li>
        </ul>
        """)

    # ==================== 业务逻辑（保留） ====================

    def paste_to_fk(self):
        try:
            vals = parse_series(self.paste_box.toPlainText())
            for e, v in zip(self.fk_edits, vals):
                e.setText(f"{v:.6f}")
        except Exception as ex:
            QMessageBox.warning(self, "解析失败", str(ex))

    def import_fk_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入缸长数据", default_export_dir(),
                                              "数据文件 (*.txt *.csv *.dat);;所有文件 (*.*)")
        if not path: return
        try:
            text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
            for e, v in zip(self.fk_edits, parse_series(text)):
                e.setText(f"{v:.6f}")
        except Exception as ex:
            QMessageBox.warning(self, "导入失败", str(ex))

    def _on_fk_mode_change(self, idx):
        """切换输入类型：保存当前模式数据、载入目标模式数据，互不冲突。"""
        if idx == 2 and not read_settings().get("debug_mode", False):
            return   # 圈数模式仅调试模式可用
        try:
            self.fk_input_cache[self._fk_mode] = [
                float(e.text() or 0) for e in self.fk_edits]
        except ValueError:
            pass
        self._fk_mode = idx
        for e, v in zip(self.fk_edits, self.fk_input_cache[idx]):
            e.setText(f"{v:.6f}")
        hints = {
            0: "绝对模式：直接输入 6 缸的绝对孔中心距 L",
            1: "相对模式：输入相对零位的位移 ΔL（正值=伸出），实际缸长 = ΔL + L0",
            2: "调试模式：输入电机圈数 N，实际缸长 = N × 导程 + 零位缸长（导程须 > 0）",
        }
        self.ref_len_hint.setText(hints.get(idx, hints[0]))

    def fill_home_lengths(self):
        """填零位值：绝对模式填 L0；相对/圈数模式填 6 个 0。"""
        if self.fk_input_mode.currentIndex() in (1, 2):
            vals = [0.0] * 6
        else:
            vals = self.solver.ref_lengths
        for e, v in zip(self.fk_edits, vals):
            e.setText(f"{v:.6f}")
        self.fk_input_cache[self.fk_input_mode.currentIndex()] = list(vals)

    def _read_fk_inputs(self):
        vals = parse_floats(self.fk_edits, "缸长输入")
        mode = self.fk_input_mode.currentIndex()
        if mode == 1:
            vals = self.solver.ref_lengths + vals
        elif mode == 2:
            lead = self._motor_lead()
            if lead <= 0:
                raise ValueError("电机圈数模式需要丝杆导程(screw_lead) > 0")
            vals = self.solver.ref_lengths + vals * lead
        return vals

    def run_fk(self):
        try:
            lengths = self._read_fk_inputs()
            tol = float(self.fk_tol.text())
            if tol <= 0: raise ValueError("收敛容差必须为正数")
            dt = float(self.fk_dt.text() or 0)
            strategy = ["home", "last", "multi"][self.fk_strategy.currentIndex()]
            solver = ["trf", "lm", "auto"][self.fk_solver.currentIndex()]
            prev_pose = getattr(self, "_prev_pose", None)
            prev_vel = getattr(self, "_prev_velocity", None)
            if dt <= 0:
                prev_pose = prev_vel = None
            res = self.solver.forward(lengths, strategy=strategy, tolerance=tol,
                                      dt=dt, prev_pose=prev_pose, prev_velocity=prev_vel,
                                      solver=solver)
            self.last_fk = res
            if dt > 0:
                self._prev_pose = res.pose.copy()
                self._prev_velocity = (res.velocity.copy()
                                       if res.velocity is not None else None)
            self._show_fk_result(res)
            self._sync_viewports(res.pose)
            self.auto_save()
        except Exception as ex:
            self.fk_status.setText(f"✗ 计算失败：{ex}")
            self._fk_ok = False
            self.apply_theme(self.app_ref.theme_mgr.theme)
            QMessageBox.warning(self, "计算失败", str(ex))

    def _workspace_colors(self, manip):
        """可操作性 → 高对比三段色带；5~95 百分位归一化解决窄区间。"""
        m = np.asarray(manip, dtype=float)
        if not len(m):
            return None
        lo, hi = float(np.percentile(m, 5)), float(np.percentile(m, 95))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
            t = np.full_like(m, 0.5)   # 退化取正中色（青绿），避免误导
        else:
            t = np.clip((m - lo) / (hi - lo), 0.0, 1.0)
        stops = np.array([[0.12, 0.36, 0.95],   # 深蓝（低）
                          [0.10, 0.86, 0.62],   # 青绿（中）
                          [1.00, 0.35, 0.12]])  # 亮橙红（高）
        colors = np.empty((len(t), 3), dtype=float)
        low = t < 0.5
        colors[low] = stops[0] + (stops[1] - stops[0]) * (t[low] / 0.5)[:, None]
        colors[~low] = stops[1] + (stops[2] - stops[1]) * ((t[~low] - 0.5) / 0.5)[:, None]
        return np.clip(colors, 0.0, 1.0)

    def _set_cloud_button(self, checked):
        if hasattr(self, "btn_cloud"):
            self.btn_cloud.blockSignals(True)
            self.btn_cloud.setChecked(checked)
            self.btn_cloud.blockSignals(False)

    def _restore_pointcloud(self, fit=False):
        """从缓存恢复点云到主视口，不重复采样。"""
        vp = self.main_viewport
        vp.show_pointcloud = True
        vp.set_pointcloud(self._ws_points6, self._ws_colors, fit=fit)

    def sample_workspace(self):
        """后台采样工作空间点云，完成后叠加到主视口（不阻塞 UI）。"""
        if getattr(self, "_ws_sampling", False):
            self.ws_status.setText("正在采样，请稍候…")
            return
        self._ws_sampling = True
        self.ws_status.setText("正在采样工作空间…")
        # 采样即视为希望看到点云：自动打开叠加开关
        self._set_cloud_button(True)
        self.main_viewport.show_pointcloud = True

        class _Worker(QObject):
            done = Signal(object)
            failed = Signal(str)

            @Slot()
            def run(self, solver):
                try:
                    res = solver.workspace_samples(axes=(0, 1, 2), grid=14)
                    self.done.emit(res)
                except Exception as ex:
                    self.failed.emit(str(ex))

        self._ws_thread = QThread(self)
        self._ws_worker = _Worker()
        self._ws_worker.moveToThread(self._ws_thread)
        self._ws_thread.started.connect(lambda: self._ws_worker.run(self.solver))
        self._ws_worker.done.connect(self._on_workspace_done)
        self._ws_worker.failed.connect(self._on_workspace_failed)
        self._ws_thread.start()

    def _on_workspace_done(self, res):
        self._ws_sampling = False
        pts = np.asarray(res["points"], dtype=float)     # (N,6) 完整位姿
        manip = np.asarray(res["manip"], dtype=float)
        self._ws_points6 = pts.copy()                     # CSV 用完整 6 列
        self._ws_colors = self._workspace_colors(manip)
        # 用户正在拖拽视口时跳过 auto-fit，避免抢镜头
        busy = getattr(self.main_viewport, "_drag_pos", None) is not None
        self.main_viewport.set_pointcloud(pts[:, :3], self._ws_colors, fit=not busy)
        self._set_cloud_button(True)
        extra = "" if not busy else "（视口操作中，未自动聚焦）"
        self.ws_status.setText(
            f"采样完成：{len(pts)} 个可行位姿（XYZ 网格，姿态固定零位）；"
            f"已叠加到主视口{extra}。")
        if hasattr(self, "_ws_thread"):
            self._ws_thread.quit(); self._ws_thread.wait(1000)

    def _on_workspace_failed(self, msg):
        self._ws_sampling = False
        self._set_cloud_button(False)
        self.ws_status.setText(f"采样失败：{msg}")
        if hasattr(self, "_ws_thread"):
            self._ws_thread.quit(); self._ws_thread.wait(1000)

    def _toggle_pointcloud(self, checked):
        """叠加开关：关闭清显示，重开复用缓存不重采样。"""
        vp = getattr(self, "main_viewport", None)
        if vp is None:
            return
        if not checked:
            vp.show_pointcloud = False
            vp.set_pointcloud(None)
            vp.update()
            self.ws_status.setText("点云已隐藏；重新打开可恢复，或点『工作空间采样』重新生成。")
            return
        vp.show_pointcloud = True
        if getattr(self, "_ws_points6", None) is not None:
            self._restore_pointcloud(fit=False)
            self.ws_status.setText("点云已恢复叠加（未重新采样）。")
        else:
            self.ws_status.setText("已启用叠加；点『工作空间采样』生成点云。")

    def export_workspace_png(self):
        vp = getattr(self, "main_viewport", None)
        if (vp is None or not getattr(vp, "show_pointcloud", False)
                or vp.pointcloud is None):
            QMessageBox.information(self, "提示", "请先采样并打开『点云』叠加。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出视口 PNG（含机构与点云）", export_path("workspace.png"),
            "PNG 图片 (*.png)")
        if path:
            vp.grab().save(path, "PNG")
            self.ws_status.setText(f"已导出当前视口 PNG（含机构与点云）：{path}")

    def export_workspace_csv(self):
        if getattr(self, "_ws_points6", None) is None:
            QMessageBox.information(self, "提示", "请先采样工作空间点云。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出工作空间 CSV", export_path("workspace.csv"),
                                              "CSV 文件 (*.csv)")
        if path:
            pts = self._ws_points6
            with open(path, "w", encoding="utf-8-sig") as fp:
                fp.write("X_mm,Y_mm,Z_mm,Roll_deg,Pitch_deg,Yaw_deg\n")
                for row in pts:
                    fp.write(",".join(f"{v:.6f}" for v in row) + "\n")
            self.ws_status.setText(f"已导出 CSV：{path}")

    def jog_move(self, dof: int, sign: int):
        """点动：沿指定自由度移动一小步 → 逆解 → 下发。"""
        try:
            step = float(self.jog_step.text()) * sign
            mode = self.jog_mode.currentIndex()
            if mode == 0 and getattr(self, "last_ik", None) is not None:
                base = self.last_ik.pose.copy()
            elif mode == 0 and getattr(self, "last_fk", None) is not None:
                base = self.last_fk.pose.copy()
            else:
                base = self.config.home_pose()
            q = base.copy(); q[dof] += step
            if not bool(self.solver.pose_limits_ok(q)):
                self.jog_status.setText(f"⚠ 该方向已到限位，未移动。")
                return
            res = self.solver.inverse(q)
            self.last_ik = res
            det = getattr(self, "link_detector", None)
            ok = False
            if det is not None and read_settings().get("debug_mode", False):
                ok = det.transmit(",".join(f"{float(v):.4f}" for v in res.lengths))
            self._sync_viewports(res.pose)
            name = ("X", "Y", "Z", "Roll", "Pitch", "Yaw")[dof]
            self.jog_status.setText(
                f"{name} {step:+.2f} → 缸长 {np.round(res.lengths[:3], 2).tolist()}… "
                f"{'已下发' if ok else '未下发（检查调试模式/链路）'}")
        except Exception as ex:
            self.jog_status.setText(f"点动失败：{ex}")

    # ---- 轨迹播放 ----
    def import_trajectory(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入轨迹 CSV", default_export_dir(),
                                              "数据文件 (*.csv *.txt *.dat)")
        if path:
            from pathlib import Path as _Path
            self.traj_input.setPlainText(_Path(path).read_text(encoding="utf-8-sig", errors="replace")[:500000])
            self.traj_status.setText(f"已导入 {path}，点『解析轨迹』生成帧队列。")

    def parse_trajectory(self):
        from collections import deque
        self._traj_frames.clear()
        self._traj_table_rows = []
        mode = self.traj_mode.currentIndex()  # 0=pose, 1=length
        for line in self.traj_input.toPlainText().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p for p in __import__("re").split(r"[,;\s]+", line) if p]
            try:
                vals = [float(v) for v in parts[:7]]
            except (ValueError, OverflowError):
                continue
            if len(vals) < 6 or not all(math.isfinite(v) for v in vals[:6]):
                continue   # 忽略 nan/inf 位姿，避免逆解产生 NaN
            try:
                dwell = int(vals[6]) if len(vals) >= 7 and math.isfinite(vals[6]) and vals[6] > 0 else 100
            except (ValueError, OverflowError):
                dwell = 100
            dwell = max(1, min(int(dwell), 600000))   # 钳制 1ms~600s，防止极端值卡死定时器
            self._traj_frames.append((np.array(vals[:6], dtype=float), dwell))
        self._traj_total = len(self._traj_frames)
        self._traj_index = 0
        self.traj_progress.setValue(0)
        # 填表预览
        self.traj_table.setRowCount(0)
        cap = effective_speed_cap(self.config)
        prev_lengths = None
        self._traj_prev_lengths = None
        for i, (vals, dwell) in enumerate(self._traj_frames):
            r = self.traj_table.rowCount(); self.traj_table.insertRow(r)
            preview_err = ""
            if mode == 0:
                try:
                    tgt = self.config.home_pose() + vals
                    res = self.solver.inverse(tgt)
                    lengths = np.asarray(res.lengths, dtype=float)
                    ok = (not str(getattr(res, "limit_warning", "") or "").startswith("⚠")
                          and not bool(np.any(getattr(res, "joint_violations", []))))
                    if not np.all(np.isfinite(lengths)):
                        ok = False; preview_err = "逆解结果非有限值"
                except Exception as ex:
                    lengths = np.full(6, np.nan)
                    ok = False; preview_err = f"逆解失败：{ex}"
            else:
                lengths = np.asarray(vals, dtype=float)
                ok = bool(np.all(lengths >= self.config.actuator_limits()[:, 0])
                          and np.all(lengths <= self.config.actuator_limits()[:, 1]))
            # 缸速 = 相邻帧缸长差 / dwell × 1000（mm/s）
            if prev_lengths is None:
                leg_spd = np.zeros(6)
            else:
                leg_spd = np.abs(lengths - prev_lengths) / max(float(dwell), 1.0) * 1000.0
            prev_lengths = np.asarray(lengths, dtype=float).copy()
            finite_spd = np.isfinite(leg_spd)
            over_speed = bool(cap > 0 and np.any(finite_spd & (leg_spd > cap + 1e-6)))
            spd = "—" if i == 0 or not np.any(finite_spd) else f"{float(np.max(leg_spd[finite_spd])):.1f}"
            state = (preview_err[:12] if preview_err else
                     ("⚠超速" if over_speed else ("✓" if ok else "⚠超限")))
            cells = [str(i + 1),
                     f"{vals[0]:.1f}, {vals[1]:.1f}, {vals[2]:.1f}" if mode == 0 else "—",
                     f"{vals[3]:.1f}, {vals[4]:.1f}, {vals[5]:.1f}" if mode == 0 else "—",
                     ", ".join((f"{x:.1f}" if math.isfinite(x) else "—") for x in lengths),
                     spd, state, str(dwell), "待播放"]
            for c_idx, txt in enumerate(cells):
                item = QTableWidgetItem(txt)
                if over_speed and c_idx in (4, 5):
                    item.setForeground(QBrush(QColor("#e57373")))
                self.traj_table.setItem(r, c_idx, item)
        self.traj_status.setText(
            f"已解析 {self._traj_total} 帧（{'位姿→逆解' if mode == 0 else '缸长直发'}）。"
            f"点『播放』逐帧求解并下发，dt/dwell 自动按帧间隔。")

    def play_trajectory(self):
        if not self._traj_frames:
            self.traj_status.setText("请先解析轨迹。")
            return
        if not read_settings().get("debug_mode", False):
            QMessageBox.information(self, "提示", "逐帧下发需先开启调试模式。")
            return
        self._traj_timer.start()

    def pause_trajectory(self):
        self._traj_timer.stop()
        self.traj_status.setText(f"已暂停（第 {self._traj_index}/{self._traj_total} 帧）。")

    def stop_trajectory(self):
        self._traj_timer.stop()
        self._traj_frames.clear()
        self._traj_prev_lengths = None
        self.traj_progress.setValue(0)
        self.traj_status.setText("已停止并清空帧队列。")

    def _traj_next_frame(self):
        from collections import deque
        if not self._traj_frames:
            self._traj_timer.stop()
            self.traj_status.setText(f"播放完成（共 {self._traj_total} 帧）。")
            return
        vals, dwell = self._traj_frames.popleft()
        self._traj_index += 1
        mode = self.traj_mode.currentIndex()
        try:
            if mode == 0:
                tgt = self.config.home_pose() + vals
                res = self.solver.inverse(tgt)
                lengths = res.lengths
                limit = str(getattr(res, "limit_warning", "") or "")
                joint_bad = bool(np.any(getattr(res, "joint_violations", [])))
            else:
                lengths = vals
                # 缸长直发也要做行程校验，避免超程帧误下发
                lim = self.config.actuator_limits()
                over = np.where((lengths < lim[:, 0] - 1e-6) | (lengths > lim[:, 1] + 1e-6))[0]
                limit = ("⚠ 缸 " + ",".join(str(i + 1) for i in over) + " 超出行程") if len(over) else ""
                joint_bad = False
            # 逐帧缸速与超速提示（仅提示，不拦截下发）
            cap = effective_speed_cap(self.config)
            prev = getattr(self, "_traj_prev_lengths", None)
            if prev is None:
                leg_spd = np.zeros(6)
            else:
                leg_spd = np.abs(lengths - prev) / max(float(dwell), 1.0) * 1000.0
            self._traj_prev_lengths = np.asarray(lengths, dtype=float).copy()
            finite_spd = np.isfinite(leg_spd)
            over_speed = bool(cap > 0 and np.any(finite_spd & (leg_spd > cap + 1e-6)))
            spd_txt = ("—" if prev is None or not np.any(finite_spd)
                       else f"{float(np.max(leg_spd[finite_spd])):.1f}")
            state_txt = "⚠超速" if over_speed else ("⚠超限" if (limit.startswith("⚠") or joint_bad) else "✓")
            row = self._traj_index - 1
            if 0 <= row < self.traj_table.rowCount():
                for c_idx, txt in ((4, spd_txt), (5, state_txt), (7, "已播放")):
                    item = QTableWidgetItem(txt)
                    if over_speed and c_idx in (4, 5):
                        item.setForeground(QBrush(QColor("#e57373")))
                    self.traj_table.setItem(row, c_idx, item)
            over_note = (f"（缸速 {spd_txt} mm/s 超过最大线速度 {cap:g}，仍按你的设置下发）"
                         if over_speed else "")
            if limit.startswith("⚠") or joint_bad:
                self.traj_status.setText(f"第 {self._traj_index} 帧超限，已跳过下发。{over_note}")
            else:
                det = getattr(self, "link_detector", None)
                ok = det.transmit(",".join(f"{float(v):.4f}" for v in lengths)) if det else False
                self.traj_status.setText(
                    f"第 {self._traj_index}/{self._traj_total} 帧 "
                    f"{'已下发' if ok else '下发失败（链路未就绪）'}。{over_note}")
            self.traj_progress.setValue(int(self._traj_index / max(self._traj_total, 1) * 100))
            if self._traj_index < self._traj_total:
                self._traj_timer.setInterval(dwell)
        except Exception as ex:
            self.traj_status.setText(f"第 {self._traj_index} 帧出错：{ex}")
            self._traj_timer.stop()

    def fill_kin_frame(self, idx: int):
        """把当前正解结果的绝对位姿写入运动学页第 idx 个帧输入框。

        idx: 2=帧 k, 1=帧 k−1, 0=帧 k−2（与 self.kin_frame_edits 顺序一致）。
        绝对位姿是运动学页 jacobian(q2)/leg_lengths(q2) 的要求，不能填相对偏移。
        """
        name = {2: "帧 k", 1: "帧 k−1", 0: "帧 k−2"}.get(idx, "该帧")
        res = getattr(self, "last_fk", None)
        if res is None or getattr(res, "pose", None) is None:
            QMessageBox.information(self, "提示", "请先计算一次姿态正解，再填入运动学帧。")
            return
        try:
            pose = np.asarray(res.pose, dtype=float).reshape(-1)
        except Exception:
            QMessageBox.warning(self, "数据异常", "当前正解位姿数据异常，无法填入。")
            return
        if pose.size < 6 or not bool(np.all(np.isfinite(pose[:6]))):
            QMessageBox.warning(self, "数据异常", "当前正解位姿包含无效数值，无法填入。")
            return
        edits = getattr(self, "kin_frame_edits", None)
        if not edits or idx < 0 or idx >= len(edits):
            return
        edits[idx].setText(", ".join(f"{v:.6f}" for v in pose[:6]))
        self.fk_status.setText(
            (self.fk_status.text() + "\n" if self.fk_status.text() else "") +
            f"✓ 已将当前绝对位姿填入运动学页 {name}（X,Y,Z,Roll,Pitch,Yaw，mm/°）；dt 请手动填写。")

    def _linkify_warning(self, text: str) -> str:
        """把限位/摆角警告里的『缸 N』转成可点击链接，便于跳转高亮。"""
        import re
        if not text:
            return ""
        esc = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        def repl(m):
            n = int(m.group(1))
            return (f"<a href='leg:{n-1}' style='color:#E08B5A;text-decoration:underline;'>"
                    f"缸{n}</a>")

        # 匹配 "缸 1" / "缸 [1, 2]" 里的每个数字：先处理方括号列表
        def repl_list(m):
            nums = re.findall(r"\d+", m.group(1))
            return "缸 [" + ", ".join(
                f"<a href='leg:{int(x)-1}' style='color:#E08B5A;text-decoration:underline;'>{x}</a>"
                for x in nums) + "]"

        esc = re.sub(r"缸\s*\[([\d,\s]+)\]", repl_list, esc)
        esc = re.sub(r"缸\s*(\d+)", repl, esc)
        # 非缸类关键词链接到参数页
        for kw in ("平移", "倾角", "摆角"):
            esc = esc.replace(kw, f"<a href='param' style='color:#E08B5A;text-decoration:underline;'>{kw}</a>")
        return esc

    def _on_limit_link(self, url: str):
        """点击报错链接：跳到对应页面并高亮相关控件。"""
        if url.startswith("leg:"):
            n = int(url.split(":")[1])
            # 切到正解页（导航组索引2）
            self.nav_group.button(2).setChecked(True)
            self._on_nav(2)
            if 0 <= n < len(self.fk_edits):
                self._flash_widget(self.fk_edits[n])
        elif url == "param":
            self.nav_group.button(1).setChecked(True)
            self._on_nav(1)
            self._flash_widget(getattr(self, "min_edit", None))

    def _flash_widget(self, widget, color="#E08B5A", times=3, interval=220):
        """让控件边框闪烁数次以高亮提醒。"""
        if widget is None:
            return
        original = widget.styleSheet()
        from PySide6.QtCore import QTimer
        state = {"n": 0}

        def step():
            if state["n"] >= times * 2:
                widget.setStyleSheet(original)
                return
            if state["n"] % 2 == 0:
                widget.setStyleSheet(original + f"; border: 2px solid {color};")
            else:
                widget.setStyleSheet(original)
            state["n"] += 1
            QTimer.singleShot(interval, step)

        step()

    def _torque_summary(self, leg_force):
        """把 6 缸推力换算成电机扭矩摘要（N·m），并与额定转矩对比。"""
        lead = self._motor_lead()
        torques = motor_torque(leg_force, lead)
        if torques is None:
            return ""
        parts = [f"{t:+.3f}" for t in torques]
        rated = float(getattr(self.config, "rated_torque", 0) or 0)
        cap = f" / 额定 {rated:.3f} N·m" if rated > 0 else ""
        over = [i + 1 for i, t in enumerate(torques) if rated > 0 and abs(t) > rated]
        warn = f"　⚠ 缸 {','.join(map(str, over))} 扭矩超额定" if over else ""
        return f"\n电机扭矩 [{', '.join(parts)}] N·m{cap}{warn}"

    def _show_fk_result(self, res):
        d = res.delta_pose
        for i in range(6):
            self.pose_value_labels[i].setText(f"{d[i]:+.3f}")
        self._fk_ok = res.converged
        self.fk_status.setText(
            f"{'✓ 收敛' if res.converged else '⚠ 未收敛'} · 迭代 {res.iterations} 次 · "
            f"最大残差 {res.max_error:.3e} mm · RMS {res.rms_error:.3e} mm\n"
            f"绝对位姿：X={res.pose[0]:.3f} Y={res.pose[1]:.3f} Z={res.pose[2]:.3f} mm · "
            f"Roll={res.pose[3]:.4f}° Pitch={res.pose[4]:.4f}° Yaw={res.pose[5]:.4f}°")
        self._joint_bad = bool(np.any(res.joint_violations))
        jw = res.joint_warning
        jump = "　<a href='param' style='color:#E08B5A;text-decoration:underline;'>【点击定位问题】</a>"
        self.joint_status.setText(
            self._linkify_warning(jw) + "  ·  各缸摆角：" +
            ", ".join(f"{x:.1f}°" for x in res.joint_angles) +
            (jump if self._joint_bad else ""))
        self._limit_bad = bool(res.limit_warning and res.limit_warning.startswith("⚠"))
        limit_text = res.limit_warning or "限位未设置"
        # 调试模式：缸速不得超过电缸管理栏的最大速度（无峰值则用额定值）
        speed_cap = effective_speed_cap(self.config)
        if res.leg_velocity is not None and speed_cap > 0:
            over = [i + 1 for i in range(6) if abs(res.leg_velocity[i]) > speed_cap + 1e-6]
            if over:
                limit_text += (f"　⚠ 超速：缸 {','.join(map(str, over))} 所需缸速超过"
                               f"电缸最大线速度 {speed_cap:g} mm/s")
                self._limit_bad = True
        self.limit_status.setText(
            self._linkify_warning(limit_text) + (jump if self._limit_bad else ""))
        if res.velocity is not None:
            v, a = res.velocity, res.acceleration
            self.dyn_status.setText(
                f"速度  X{v[0]:+.2f} Y{v[1]:+.2f} Z{v[2]:+.2f} mm/s · "
                f"Roll{v[3]:+.2f} Pitch{v[4]:+.2f} Yaw{v[5]:+.2f} °/s\n"
                f"加速度  X{a[0]:+.2f} Y{a[1]:+.2f} Z{a[2]:+.2f} mm/s² · "
                f"角加速度 Roll{a[3]:+.2f} Pitch{a[4]:+.2f} Yaw{a[5]:+.2f} °/s²\n"
                f"整体推力 {res.total_force:.1f} N" +
                (self._torque_summary(res.leg_force) if res.leg_force is not None else "") +
                (f"　{res.dynamic_warning}" if res.speed_violation or res.force_violation else ""))
            self._dyn_ok = not (res.speed_violation or res.force_violation)
        vals = parse_floats(self.fk_edits, "缸长输入")
        fk_mode = self.fk_input_mode.currentIndex()
        mode_offset = fk_mode in (1, 2)
        # 调试数据：相对零位位移 → 圈数 / 转速
        lead = self._motor_lead()
        dL = res.lengths - self.solver.ref_lengths
        revs, rpms = motor_revolutions(dL, lead, res.leg_velocity)
        for i in range(6):
            if fk_mode == 2:   # 圈数模式：显示输入圈数 × 导程 + 零位
                ui = vals[i] * lead + self.solver.ref_lengths[i] if lead > 0 else 0.0
            else:
                ui = vals[i] + (self.solver.ref_lengths[i] if mode_offset else 0)
            self.fk_table.setItem(i, 0, QTableWidgetItem(f"缸 {i+1}"))
            self.fk_table.setItem(i, 1, QTableWidgetItem(f"{ui:.4f}"))
            self.fk_table.setItem(i, 2, QTableWidgetItem(f"{res.lengths[i]:.4f}"))
            self.fk_table.setItem(i, 3, QTableWidgetItem(f"{res.residuals[i]:+.2e}"))
            self.fk_table.setItem(i, 4, QTableWidgetItem(
                f"{res.leg_velocity[i]:+.2f}" if res.leg_velocity is not None else "—"))
            self.fk_table.setItem(i, 5, QTableWidgetItem(
                f"{res.leg_force[i]:+.1f}" if res.leg_force is not None else "—"))
            self.fk_table.setItem(i, 6, QTableWidgetItem(
                f"{revs[i]:+.2f}" if revs is not None else "—"))
            self.fk_table.setItem(i, 7, QTableWidgetItem(
                f"{rpms[i]:.0f}" if rpms is not None else "—"))
        r = res.rotation
        detail = "旋转矩阵 R (ZYX)：\n"
        for row in r:
            detail += "  [" + "  ".join(f"{x:+.7f}" for x in row) + " ]\n"
        detail += "\n上铰点世界坐标 (mm)：\n"
        for i, pnt in enumerate(res.anchors_world):
            detail += f"  P{i+1}: ({pnt[0]:+.2f}, {pnt[1]:+.2f}, {pnt[2]:+.2f})\n"
        detail += "\n虎克铰摆角 (°) — 下铰 / 上铰 / 综合：\n"
        ab, ap, ac = self.solver.joint_angles(res.pose)
        for i in range(6):
            detail += f"  缸{i+1}: {ab[i]:6.2f} / {ap[i]:6.2f} / {ac[i]:6.2f}\n"
        self.fk_detail.setPlainText(detail)
        self.apply_theme(self.app_ref.theme_mgr.theme)

    def export_calculation_archive(self):
        """导出自包含计算存档：动力学输入 + 输出 + 整套转换公式。"""
        if self.last_fk is None:
            QMessageBox.information(self, "无结果", "请先执行正解计算。"); return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存计算存档", export_path(f"{self.profile_name}_计算存档.json"),
            "JSON (*.json)")
        if not path:
            return
        try:
            res = self.last_fk
            cfg = self.config
            s = self.solver
            # ---- 动力学输入 ----
            dynam_in = {
                "结构": cfg.structure,
                "上铰点摆角限_deg": cfg.platform_joint_limit,
                "下铰点摆角限_deg": cfg.base_joint_limit,
                "零位定义": "中间位" if cfg.zero_reference == "middle" else "最低位",
                "行程下限_mm": cfg.min_length, "行程上限_mm": cfg.max_length,
                "额定推力_N": cfg.max_force, "重力_m_s2": cfg.gravity,
                "负载质量_kg": cfg.payload_mass, "平台自重_kg": cfg.platform_mass,
                "采样间隔_dt_s": float(self.fk_dt.text() or 0),
                "零位缸长L0_mm": [float(x) for x in s.ref_lengths],
                "输入缸长_mm": [float(x - r) for x, r in
                             zip(res.lengths, res.residuals)],
            }
            # ---- 输出数据 ----
            dynam_out = {
                "绝对位姿_mm_deg": [float(x) for x in res.pose],
                "相对零位_delta_mm_deg": [float(x) for x in res.delta_pose],
                "反算缸长_mm": [float(x) for x in res.lengths],
                "残差_mm": [float(x) for x in res.residuals],
                "收敛": bool(res.converged), "迭代次数": int(res.iterations),
                "最大残差_mm": float(res.max_error),
                "虎克铰摆角_deg": [float(x) for x in s.joint_angles(res.pose)[2]],
            }
            if res.velocity is not None:
                dynam_out["平台速度"] = [float(x) for x in res.velocity]
            if res.acceleration is not None:
                dynam_out["平台加速度"] = [float(x) for x in res.acceleration]
            if res.leg_velocity is not None:
                dynam_out["缸速_mm_s"] = [float(x) for x in res.leg_velocity]
            if res.leg_force is not None:
                dynam_out["缸推力_N"] = [float(x) for x in res.leg_force]
                dynam_out["整体推力_N"] = float(res.total_force)
            # ---- 整套转换公式 ----
            formulas = {
                "逆解_缸长": "L_i = || R(q) · p_i + t − b_i ||，R = Rz(yaw)·Ry(pitch)·Rx(roll)（ZYX 内旋）",
                "正解": "min_q Σ(L_i(q) − L_i^target)²，阻尼最小二乘 SciPy TRF 迭代",
                "雅可比": "J[i,:3] = u_i（单位缸轴），J[i,3:] = (R·p_i) × u_i；L̇ = J·[v, ω]（ω 单位 rad/s）",
                "速度正变换": "缸速 = J · [vx,vy,vz,ωx,ωy,ωz]",
                "速度逆变换": "[v,ω] = pinv(J) · 缸速",
                "推力_虚功": "F_leg = pinv(Jᵀ) · w，w = [m·a_x, m·a_y, m·(g+a_z), I·α_x, I·α_y, I·α_z]",
                "转动惯量近似": "I = 0.5 · m · r²（上平台薄圆盘，r = 铰点圆平均半径 m）",
                "摆角": "θ = acos(u · axis)，下铰轴 +Z、上铰轴 R·(−Z)，两端取较大值",
                "行程绝对范围": "S = 上限−下限；零位最低位→[L0, L0+S]，中间位→[L0−S/2, L0+S/2]",
                "限位判定": "L_i ∈ [下限_i, 上限_i] 且摆角 ≤ 许用摆角",
            }
            data = {
                "存档类型": "六自由度姿态解算-计算存档",
                "版本": APP_VERSION,
                "档案": self.profile_name,
                "说明": "自包含：动力学输入 + 输出数据 + 整套输入输出转换公式",
                "动力学输入": dynam_in,
                "输出数据": dynam_out,
                "转换公式": formulas,
            }
            def _conv(o):
                if isinstance(o, np.generic):
                    return o.item()
                if isinstance(o, np.ndarray):
                    return o.tolist()
                return str(o)
            Path(path).write_text(
                json.dumps(data, ensure_ascii=False, indent=2, default=_conv), encoding="utf-8")
            QMessageBox.information(self, "已保存", f"计算存档已保存到：\n{path}")
        except Exception as ex:
            QMessageBox.warning(self, "保存失败", str(ex))

    def export_fk_csv(self):
        if self.last_fk is None:
            QMessageBox.information(self, "无结果", "请先执行正解计算。"); return
        path, _ = QFileDialog.getSaveFileName(self, "导出结果", export_path("pose_result.csv"), "CSV (*.csv)")
        if not path: return
        res = self.last_fk
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["项目", "X_mm", "Y_mm", "Z_mm", "Roll_deg", "Pitch_deg", "Yaw_deg"])
            w.writerow(["绝对位姿"] + [f"{x:.8f}" for x in res.pose])
            w.writerow(["相对零位Δ"] + [f"{x:.8f}" for x in res.delta_pose])
            w.writerow([])
            w.writerow(["作动器", "输入长度mm", "反算长度mm", "残差mm", "虎克铰摆角deg"])
            ab, ap, ac = self.solver.joint_angles(res.pose)
            for i in range(6):
                w.writerow([f"缸{i+1}", f"{res.lengths[i]-res.residuals[i]:.8f}",
                            f"{res.lengths[i]:.8f}", f"{res.residuals[i]:+.3e}", f"{ac[i]:.3f}"])
            w.writerow([])
            w.writerow(["结构", self.config.structure, "状态", "收敛" if res.converged else "未收敛",
                        "最大残差mm", f"{res.max_error:.3e}", "铰点校验", res.joint_warning])
        QMessageBox.information(self, "已导出", path)

    def export_all(self):
        """导出全部数据：档案/平台参数、电缸规格、限位、输入、正逆解、动态与批量结果。"""
        path, _ = QFileDialog.getSaveFileName(
            self, "导出全部数据", export_path(f"{self.profile_name}_全部数据.json"),
            "JSON (*.json);;CSV (*.csv)")
        if not path:
            return
        try:
            data = self._collect_all_data()
            if path.lower().endswith(".csv"):
                # CSV：键值对平铺
                with open(path, "w", newline="", encoding="utf-8-sig") as f:
                    w = csv.writer(f)
                    w.writerow(["分类", "字段", "值"])
                    for section, kv in data.items():
                        for k, v in kv.items():
                            w.writerow([section, k, v])
            else:
                import json
                def _conv(o):
                    if isinstance(o, np.generic):
                        return o.item()
                    if isinstance(o, np.ndarray):
                        return o.tolist()
                    return str(o)
                Path(path).write_text(
                    json.dumps(data, ensure_ascii=False, indent=2, default=_conv),
                    encoding="utf-8")
            QMessageBox.information(self, "已导出", f"全部数据已导出到：\n{path}")
        except Exception as ex:
            QMessageBox.warning(self, "导出失败", str(ex))

    def _collect_all_data(self) -> dict:
        """汇总当前档案的参数数据为 {分类: {字段: 值}}。

        重点导出：电缸规格、平台初始坐标系、几何参数，以及算得的
        航空极限、工作空间与其他关键参数；附带最近一次解算结果。
        """
        cfg = self.config.to_dict()
        out = {}

        # ---- 1. 档案信息 ----
        out["档案信息"] = {
            "名称": self.profile_name, "备注": self.note,
            "机构结构": self.config.structure,
            "导出说明": "电缸规格 + 平台初始坐标系 + 几何参数 + 关键派生参数",
        }

        # ---- 2. 电缸规格 ----
        out["电缸规格"] = {
            "电缸名称": (self.config.actuator_names or [""])[0],
            "电缸编码": (self.config.actuator_codes or [""])[0],
            "电缸品牌": (self.config.actuator_brands or [""])[0],
            "有效行程_mm": cfg.get("stroke", 0),
            "额定推力_N": cfg.get("max_force", 0),
            "最大线速度_mm_s": effective_speed_cap(cfg),
            "峰值线速度_mm_s": float(getattr(cfg, "peak_speed", 0) or 0),
            "额定线速度_mm_s": float(getattr(cfg, "rated_speed", 0) or 0),
            "缸径_mm": cfg.get("actuator_bore", 0),
            "负载质量_kg": cfg.get("payload_mass", 0),
            "平台自重_kg": cfg.get("platform_mass", 0),
            "重力加速度_m_s2": cfg.get("gravity", 9.80665),
            "行程下限ΔL_mm": cfg.get("min_length", 0),
            "行程上限ΔL_mm": cfg.get("max_length", 0),
            "零位定义": "中间位" if cfg.get("zero_reference") == "middle" else "最低位",
            "下节点许用摆角_deg": cfg.get("base_joint_limit", 0),
            "上节点许用摆角_deg": cfg.get("platform_joint_limit", 0),
            "最大平移包络_mm": cfg.get("max_translation", 0),
            "最大倾角包络_deg": cfg.get("max_tilt", 0),
        }
        # 各缸绝对极限（由行程参数推导）
        try:
            lims = self.config.actuator_limits()
            for i in range(6):
                out["电缸规格"][f"缸{i+1}_最小长度_mm"] = lims[i, 0]
                out["电缸规格"][f"缸{i+1}_最大长度_mm"] = lims[i, 1]
        except Exception:
            pass
        # 各动作安全载荷
        try:
            safe = self.solver.safe_payloads()
            for i, nm in enumerate(["X纵荡", "Y横荡", "Z升沉", "Roll", "Pitch", "Yaw"]):
                out["电缸规格"][f"安全载荷_{nm}_kg"] = safe[i]
        except Exception:
            pass

        # ---- 3. 平台初始坐标系 ----
        try:
            base, plat = self.config.anchors()
            origin = self.config.origin()
            ref = self.solver.ref_lengths
            coord = {}
            for i in range(6):
                coord[f"下铰点B{i+1}_X_mm"] = base[i, 0]
                coord[f"下铰点B{i+1}_Y_mm"] = base[i, 1]
                coord[f"下铰点B{i+1}_Z_mm"] = base[i, 2]
                coord[f"上铰点P{i+1}_X_mm"] = plat[i, 0]
                coord[f"上铰点P{i+1}_Y_mm"] = plat[i, 1]
                coord[f"上铰点P{i+1}_Z_mm"] = plat[i, 2]
                coord[f"零位缸长L0_{i+1}_mm"] = ref[i]
            coord["零位上平台中心_X_mm"] = origin[0]
            coord["零位上平台中心_Y_mm"] = origin[1]
            coord["零位上平台中心_Z_mm"] = origin[2]
            coord["零位姿态_Roll_deg"] = 0.0
            coord["零位姿态_Pitch_deg"] = 0.0
            coord["零位姿态_Yaw_deg"] = 0.0
            coord["坐标系约定"] = "右手系；Z轴竖直向上；下平台中心为原点；欧拉角 ZYX 内旋"
            out["平台初始坐标系"] = coord
        except Exception:
            out["平台初始坐标系"] = {}

        # ---- 4. 几何参数 ----
        geom_keys = ("base_radius", "platform_radius", "base_pair_angle", "platform_pair_angle",
                     "base_phase", "platform_phase", "home_height", "base_joint_z",
                     "platform_joint_z", "pair_centers", "use_direct_points")
        cn_map = {"base_radius": "下平台铰点圆半径_mm", "platform_radius": "上平台铰点圆半径_mm",
                  "base_pair_angle": "下平台成对夹角_deg", "platform_pair_angle": "上平台成对夹角_deg",
                  "base_phase": "下平台相位角_deg", "platform_phase": "上平台相位角_deg",
                  "home_height": "零位上平台高度_mm", "base_joint_z": "下铰点Z偏移_mm",
                  "platform_joint_z": "上铰点Z偏移_mm", "pair_centers": "三组铰点中心方位角_deg",
                  "use_direct_points": "是否直接坐标建模"}
        out["几何参数"] = {cn_map.get(k, k): cfg.get(k) for k in geom_keys if k in cfg}

        # ---- 5. 关键派生参数 ----
        try:
            ws = self.solver.workspace_limits()
            derived = {}
            for i, nm in enumerate(["X纵荡", "Y横荡", "Z升沉", "Roll", "Pitch", "Yaw"]):
                unit = "mm" if i < 3 else "deg"
                derived[f"{nm}_负向最大_{unit}"] = ws[i, 0]
                derived[f"{nm}_正向最大_{unit}"] = ws[i, 1]
            derived["雅可比秩"] = int(self.solver.rank())
            out["关键派生参数"] = derived
        except Exception:
            out["关键派生参数"] = {}

        r = self.last_fk
        if r is not None:
            out["正解结果"] = {
                f"绝对位姿_{k}": v for k, v in zip(
                    ["X_mm", "Y_mm", "Z_mm", "Roll_deg", "Pitch_deg", "Yaw_deg"], r.pose)}
            out["正解结果"].update({
                f"相对零位_{k}": v for k, v in zip(
                    ["dX_mm", "dY_mm", "dZ_mm", "dRoll_deg", "dPitch_deg", "dYaw_deg"],
                    r.delta_pose)})
            out["正解结果"].update({"收敛": r.converged, "最大残差_mm": r.max_error,
                                 "RMS_mm": r.rms_error, "迭代次数": r.iterations})
        if r is not None and r.velocity is not None:
            out["动态数据"] = {
                f"速度_{k}": v for k, v in zip(
                    ["VX_mm_s", "VY_mm_s", "VZ_mm_s", "Roll_deg_s", "Pitch_deg_s", "Yaw_deg_s"],
                    r.velocity)}
            out["动态数据"].update({
                f"加速度_{k}": v for k, v in zip(
                    ["AX_mm_s2", "AY_mm_s2", "AZ_mm_s2", "Roll_deg_s2", "Pitch_deg_s2", "Yaw_deg_s2"],
                    r.acceleration)})
            out["动态数据"]["整体推力_N"] = r.total_force
            if r.leg_force is not None:
                out["动态数据"].update({f"缸{i+1}_推力_N": v for i, v in enumerate(r.leg_force)})
            if r.leg_velocity is not None:
                out["动态数据"].update({f"缸{i+1}_速度_mm_s": v for i, v in enumerate(r.leg_velocity)})
        if self.last_ik is not None:
            out["逆解结果"] = {f"缸{i+1}_长度_mm": v for i, v in enumerate(self.last_ik.lengths)}
        # 批量结果表
        bt = getattr(self, "batch_table", None)
        if bt is not None and bt.rowCount() > 0:
            rows = []
            for r_ in range(bt.rowCount()):
                rows.append([bt.item(r_, c).text() if bt.item(r_, c) else ""
                             for c in range(bt.columnCount())])
            out["批量结果"] = {f"行{i+1}": " | ".join(row) for i, row in enumerate(rows)}
        return out

    def _send_lengths_to_device(self, source, lengths, status_widget, empty_hint):
        """下发共用核心：source=正解/逆解；lengths=6 缸长度；status_widget=真实模式追加状态处。"""
        det = getattr(self, "link_detector", None)
        if det is None or not read_settings().get("debug_mode", False):
            QMessageBox.information(self, "提示", "请先开启调试模式。")
            return
        if lengths is None:
            QMessageBox.information(self, "提示", empty_hint)
            return
        data = ",".join(f"{float(v):.4f}" for v in lengths)
        ok = det.transmit(data)
        state, msg = det.state.value, self.link_indicator.toolTip()
        if not ok:
            QMessageBox.warning(self, "下发未成功",
                                f"[{source}] 链路未就绪或队列已满。\n数据帧：{data}\n"
                                f"光点状态：{state}。请检查串口/模拟信号。")
            return
        if not det.is_real:
            QMessageBox.information(self, "模拟模式",
                                    f"未配置串口，无法真实下发。\n[{source}] 数据帧：{data}\n"
                                    f"光点状态：{state}。可右键光点注入模拟信号验证绿色。")
        elif status_widget is not None:
            status_widget.setText(status_widget.text() + f"\n[{source}][{state}] {msg}")

    def _send_to_device(self):
        """正解页下发：与逆解一致，有行程/关节/超速警告时二次确认后仍可下发。"""
        if self.last_fk is None:
            self._send_lengths_to_device("正解", None, getattr(self, "fk_status", None),
                                         "请先计算一次正解。")
            return
        r = self.last_fk
        has_warning = (not getattr(r, "converged", True)
                       or bool(getattr(r, "speed_violation", False))
                       or bool(getattr(r, "force_violation", False))
                       or bool(str(getattr(r, "limit_warning", "") or "").startswith("⚠"))
                       or bool(np.any(getattr(r, "joint_violations", []) or [])))
        if has_warning:
            ans = QMessageBox.question(
                self, "正解存在警告",
                "当前正解存在行程/关节/超速警告，仍要下发吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ans != QMessageBox.Yes:
                return
        self._send_lengths_to_device(
            "正解", r.lengths, getattr(self, "fk_status", None),
            "请先计算一次正解。")

    def _send_to_device_ik(self):
        """逆解页下发：以 last_ik 为主判据；有警告时二次确认后仍可下发。"""
        if self.last_ik is None:
            QMessageBox.information(self, "提示", "请先计算一次逆解。")
            return
        has_warning = (not getattr(self, "_ik_ok", True)
                       or getattr(self, "_ik_joint_bad", False)
                       or getattr(self, "_ik_limit_bad", False))
        if has_warning:
            ans = QMessageBox.question(
                self, "逆解存在警告",
                "当前逆解存在行程/关节/超速警告，仍要下发吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ans != QMessageBox.Yes:
                return
        self._send_lengths_to_device(
            "逆解", self.last_ik.lengths,
            getattr(self, "ik_status", None),
            "请先计算一次逆解。")

    def copy_fk_result(self):
        if self.last_fk is None: return
        d = self.last_fk.delta_pose
        text = (f"X={d[0]:+.5f}mm  Y={d[1]:+.5f}mm  Z={d[2]:+.5f}mm\n"
                f"Roll={d[3]:+.6f}°  Pitch={d[4]:+.6f}°  Yaw={d[5]:+.6f}°\n"
                f"结构={self.config.structure} · {self.last_fk.joint_warning}")
        QApplication.clipboard().setText(text)

    def run_ik(self):
        try:
            offset = parse_floats(self.ik_edits, "目标位姿")
            target = self.config.home_pose() + offset
            try:
                dt = float(self.ik_dt.text() or 0) if hasattr(self, "ik_dt") else 0
            except ValueError:
                dt = 0
            prev = getattr(self, "_prev_ik_pose", None)
            if prev is None:
                prev = self.config.home_pose()
            res = self.solver.inverse(target, dt=dt if dt > 0 else 0.0,
                                      prev_pose=prev if dt > 0 else None,
                                      prev_velocity=self._prev_ik_velocity if dt > 0 else None)
            if dt > 0:
                self._prev_ik_pose = target.copy()
                self._prev_ik_velocity = (res.velocity.copy()
                                          if getattr(res, "velocity", None) is not None else None)
            self.last_ik = res
            ref = self.solver.ref_lengths
            limits = self.config.actuator_limits()
            # 调试数据：圈数 / 转速（逆解无 dt 时 leg_velocity=None，转速显示 —）
            revs, rpms = motor_revolutions(res.lengths - ref, self._motor_lead(),
                                           res.leg_velocity)
            # 标记行程超限的缸（在长度单元格加 ⚠ 并用红色文字）
            theme = self.app_ref.theme_mgr.theme
            for i in range(6):
                over = (res.lengths[i] < limits[i, 0] - 1e-6 or
                        res.lengths[i] > limits[i, 1] + 1e-6)
                self.ik_table.setItem(i, 0, QTableWidgetItem(f"缸 {i+1}"))
                c1 = QTableWidgetItem(f"{res.lengths[i]:.4f}" + ("  ⚠" if over else ""))
                c2 = QTableWidgetItem(f"{res.lengths[i]-ref[i]:+.4f}")
                if over:
                    c1.setForeground(QBrush(QColor(theme.danger)))
                    c2.setForeground(QBrush(QColor(theme.danger)))
                self.ik_table.setItem(i, 1, c1)
                self.ik_table.setItem(i, 2, c2)
                self.ik_table.setItem(i, 3, QTableWidgetItem(
                    f"{revs[i]:+.2f}" if revs is not None else "—"))
                self.ik_table.setItem(i, 4, QTableWidgetItem(
                    f"{rpms[i]:.0f}" if rpms is not None else "—"))
            self.ik_status.setText("✓ 逆解完成（闭式解）。目标偏移：" +
                                   ", ".join(f"{x:+.2f}" for x in offset) +
                                   (self._torque_summary(res.leg_force)
                                    if getattr(res, "leg_force", None) is not None else ""))
            self._ik_ok = True
            self._ik_joint_bad = bool(np.any(res.joint_violations))
            self.ik_joint_status.setText(res.joint_warning)
            # 显示行程超限警告（此前遗漏，导致超限也显示"完成"）
            self._ik_limit_bad = bool(res.limit_warning and res.limit_warning.startswith("⚠"))
            limit_text = res.limit_warning or "行程：全部在允许范围内"
            # 调试模式：缸速不得超过电缸管理栏的最大速度（无最大值则用额定值）
            speed_cap = effective_speed_cap(self.config)
            if res.leg_velocity is not None and speed_cap > 0:
                over_idx = [i + 1 for i in range(6) if abs(res.leg_velocity[i]) > speed_cap + 1e-6]
                if over_idx:
                    limit_text += (f"　⚠ 超速：缸 {','.join(map(str, over_idx))} 所需缸速超过"
                                   f"电缸最大线速度 {speed_cap:g} mm/s")
                    self._ik_limit_bad = True
                    for i in range(6):
                        if abs(res.leg_velocity[i]) > speed_cap + 1e-6:
                            cell = self.ik_table.item(i, 4)
                            if cell:
                                cell.setForeground(QBrush(QColor(theme.danger)))
            self.ik_limit_status.setText(self._linkify_warning(limit_text))
            self._ik_ok = not self._ik_limit_bad
            self.verify_label.setText("已生成目标姿态，可点下方按钮送回正解验证一致性。")
            self._sync_viewports(res.pose)
            self.apply_theme(self.app_ref.theme_mgr.theme)
            # 回填功能：每次成功逆解都记录目标偏移并持久化（与转速用的 _prev_ik_pose 解耦）
            st = read_settings()
            st["ik_last_pose_offset"] = [float(v) for v in offset]
            write_settings(st)
            q = quat_from_pose(res.pose)
            self.ik_quat_out.setText(
                f"当前姿态四元数 [w,x,y,z]：{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}")
        except Exception as ex:
            QMessageBox.warning(self, "逆解失败", str(ex))

    def apply_ik_quaternion(self):
        """把输入框的四元数 [w,x,y,z] 转成 Roll/Pitch/Yaw 填入目标位姿角度框。"""
        try:
            vals = np.asarray(parse_series(self.ik_quat_edit.text(), 4), dtype=float)
            p = pose_from_quat(vals, (0, 0, 0))
            for e, v in zip(self.ik_edits[3:6], p[3:6]):
                e.setText(f"{v:.6f}")
            self.ik_status.setText(
                f"✓ 四元数已应用：Roll={p[3]:.4f}° Pitch={p[4]:.4f}° Yaw={p[5]:.4f}°（平移请在上框另填）")
        except Exception as ex:
            QMessageBox.warning(self, "四元数无效", str(ex))

    def restore_prev_ik_pose(self):
        """把 settings.json 中保存的上一帧目标位姿偏移回填到 6 个 IK 输入框。"""
        st = read_settings()
        vals = st.get("ik_last_pose_offset")
        if not (isinstance(vals, (list, tuple)) and len(vals) == 6):
            QMessageBox.information(self, "提示", "暂无历史位姿：请先计算一次逆解。")
            return
        try:
            nums = [float(v) for v in vals]
        except (TypeError, ValueError):
            QMessageBox.warning(self, "数据异常", "历史位姿数据已损坏，无法回填。")
            return
        if not all(np.isfinite(v) for v in nums):
            QMessageBox.warning(self, "数据异常", "历史位姿包含无效数值，无法回填。")
            return
        for e, v in zip(self.ik_edits, nums):
            e.setText(f"{v:g}")
        self.ik_status.setText("已回填上一次目标位姿偏移（相对零位）。")

    def verify_ik(self):
        if self.last_ik is None:
            QMessageBox.information(self, "提示", "请先计算逆解。"); return
        res = self.solver.forward(self.last_ik.lengths, strategy="home")
        err = np.max(np.abs(res.delta_pose - self.last_ik.delta_pose))
        self.verify_label.setText(
            f"往返验证：最大位姿误差 {err:.3e}（mm/°）；正解{'收敛' if res.converged else '未收敛'}。")
        self._verify_ok = err < 1e-4
        self.apply_theme(self.app_ref.theme_mgr.theme)

    def import_batch_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入批量数据", default_export_dir(),
                                              "数据文件 (*.txt *.csv *.dat);;所有文件 (*.*)")
        if path:
            self.batch_input.setPlainText(Path(path).read_text(encoding="utf-8-sig", errors="replace")[:200000])

    def run_batch(self):
        if getattr(self, "_batch_running", False):
            if hasattr(self, "_batch_flag"):
                self._batch_flag["v"] = True
            self.batch_status.setText("正在取消…")
            return
        lines = [ln.strip() for ln in self.batch_input.toPlainText().splitlines() if ln.strip()]
        if not lines:
            QMessageBox.information(self, "提示", "请先输入或导入批量数据。"); return
        self._batch_running = True
        self._batch_cancel = False
        self.batch_progress.setVisible(True); self.batch_progress.setValue(0)
        self.batch_status.setText(f"后台计算中：0/{len(lines)} 行（再点『批量求解』可取消）")

        class _BatchWorker(QObject):
            progress = Signal(int, int)
            finished = Signal(list, list)
            cancelled = Signal()

            @Slot()
            def run(self, solver, refs, relative, cancel_flag):
                rows, errors = [], []
                for idx, line in enumerate(lines):
                    if cancel_flag["v"]:
                        self.cancelled.emit(); return
                    try:
                        vals = np.asarray(parse_series(line), float)
                        if relative:
                            vals = refs + vals
                        res = solver.forward(vals, strategy="home")
                        rows.append(([f"{x:+.4f}" for x in res.delta_pose] +
                                     [f"{res.max_error:.2e}",
                                      "⚠超限" if np.any(res.joint_violations) else "正常",
                                      "收敛" if res.converged else "未收敛"]))
                    except Exception as ex:
                        rows.append(["—"] * 6 + ["—", "—", "错误"])
                        errors.append((idx + 1, str(ex)))
                    self.progress.emit(idx + 1, len(lines))
                self.finished.emit(rows, errors)

        flag = {"v": False}
        self._batch_flag = flag
        self._batch_thread = QThread(self)
        self._batch_worker = _BatchWorker()
        self._batch_worker.moveToThread(self._batch_thread)
        self._batch_thread.started.connect(lambda: self._batch_worker.run(
            self.solver, self.solver.ref_lengths,
            self.fk_input_mode.currentIndex() == 1, flag))
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.finished.connect(self._on_batch_done)
        self._batch_worker.cancelled.connect(self._on_batch_cancelled)
        self._batch_thread.start()

    def _on_batch_progress(self, done, total):
        self.batch_progress.setValue(int(done / max(total, 1) * 100))
        self.batch_status.setText(f"后台计算中：{done}/{total} 行（再点『批量求解』可取消）")

    def _on_batch_cancelled(self):
        self._batch_running = False
        self.batch_progress.setVisible(False)
        self.batch_status.setText("已取消批量计算。")
        self._batch_thread.quit(); self._batch_thread.wait(1000)

    def _on_batch_done(self, rows, errors):
        self._batch_running = False
        self.batch_progress.setVisible(False)
        self.batch_table.setRowCount(len(rows))
        theme = self.app_ref.theme_mgr.theme
        for r, vals in enumerate(rows):
            self.batch_table.setItem(r, 0, QTableWidgetItem(str(r + 1)))
            for c, v in enumerate(vals, start=1):
                it = QTableWidgetItem(v)
                it.setForeground(QBrush(QColor(theme.input if 1 <= c <= 3 else theme.output)))
                self.batch_table.setItem(r, c, it)
        self.batch_status.setText(f"完成 {len(rows)} 行 · 收敛 {sum(1 for x in rows if x[-1]=='收敛')} 行 · "
                                  f"解析失败 {len(errors)} 行")
        self._batch_thread.quit(); self._batch_thread.wait(1000)

    def export_batch_csv(self):
        if self.batch_table.rowCount() == 0:
            QMessageBox.information(self, "提示", "请先执行批量求解。"); return
        path, _ = QFileDialog.getSaveFileName(self, "导出批量结果", export_path("batch_result.csv"), "CSV (*.csv)")
        if not path: return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow([self.batch_table.horizontalHeaderItem(c).text()
                        for c in range(self.batch_table.columnCount())])
            for r in range(self.batch_table.rowCount()):
                w.writerow([self.batch_table.item(r, c).text() if self.batch_table.item(r, c) else ""
                            for c in range(self.batch_table.columnCount())])
        QMessageBox.information(self, "已导出", path)


# ==================== 应用入口 ====================

class App:
    def __init__(self):
        self.manager = ProfileManager()
        self.theme_mgr = ThemeManager()
        self.theme_mgr.changed.connect(self.on_theme_changed)
        self.launcher = None
        self.main = None
        self.rail_lock_state = {"locked": True}    # 侧栏默认锁定展开，两窗口共享

    @property
    def theme(self):
        return self.theme_mgr.theme

    def toggle_theme(self):
        self.theme_mgr.toggle()

    def on_theme_changed(self, theme):
        style = build_style(theme)
        app = QApplication.instance()
        app.setStyleSheet(style)
        apply_palette(app, theme)   # 同步原生控件配色
        for w in QApplication.topLevelWidgets():
            if isinstance(w, (QMainWindow, QDialog)):
                w.setStyleSheet(style)
        if self.launcher:
            self.launcher.apply_theme(theme)
        if self.main:
            self.main.apply_theme(theme)
        from hints import HintDialog as _HintDialog
        for w in QApplication.topLevelWidgets():
            if isinstance(w, _HintDialog):
                w.apply_theme(theme)

    def show_launcher(self):
        self.launcher = Launcher(self)
        self.launcher.show()

    def open_profile(self, name):
        self.main = MainWindow(self, name)
        self.main.show()
        if self.launcher:
            self.launcher.hide()


def main():
    # 高 DPI：让 Qt 按显示比例缩放，避免窗口缩放/不同分辨率下文字发虚变小
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setStyle("Fusion")
    # 全局字体用 pt（随 DPI 放大），保证缩放后文字依旧清晰可读
    f = QFont("Microsoft YaHei UI", 10)
    f.setPointSize(10)
    app.setFont(f)
    icon_path = Path(__file__).resolve().parent / "app.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))   # 所有窗口统一图标
    controller = App()
    app.setStyleSheet(build_style(controller.theme))
    apply_palette(app, controller.theme)
    controller.show_launcher()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
