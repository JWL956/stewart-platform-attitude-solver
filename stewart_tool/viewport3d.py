# -*- coding: utf-8 -*-
"""三维姿态视口：基于 QPainter 的轻量软件渲染。

不依赖 OpenGL / matplotlib / pyqtgraph，打包体积几乎不增加。
支持鼠标拖拽旋转视角、滚轮缩放、双击重置，可显示零位参考与地面网格。
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt, QPointF, QTimer, QRectF, Signal
from PySide6.QtGui import (
    QColor, QPainter, QPen, QBrush, QPolygonF, QLinearGradient,
    QRadialGradient, QFont,
)
from PySide6.QtWidgets import QWidget

from kinematics import rot_matrix


def _dot(a, b):
    return float(np.dot(a, b))


def _cross(a, b):
    return np.cross(a, b)


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


class Viewport3D(QWidget):
    """可交互的 Stewart 平台三维线框/实体视口。"""

    pose_dragged = Signal(object)   # 拖拽平台模式：发出新位姿 [x,y,z,roll,pitch,yaw]

    def __init__(self, parent=None, theme=None):
        super().__init__(parent)
        self.theme = theme
        self.solver = None
        self.config = None
        self.base = None                 # (6,3) 下铰点世界坐标
        self.platform = None             # (6,3) 上铰点本地坐标
        self.home_pose = None            # (6,) 零位位姿
        self.pose = None                 # (6,) 当前显示位姿
        self.pointcloud = None           # (N,3) 工作空间点云世界坐标
        self.pointcolors = None          # (N,3) 每点 RGB (0-1)
        self.show_pointcloud = False     # 点云叠加开关：False 时完全不绘制
        self._fit_center = None          # auto-fit 相机目标；None 表示沿用默认
        self.last_pointcloud_error = None  # 绘制异常留痕（便于诊断）
        self.drag_pose_mode = False      # 拖拽平台模式：左键拖动改变上平台位姿
        self._drag_pose_start = None
        self._drag_pose_base = None
        self.ellipsoid = None            # (N,3) 力椭球点云

        self.azimuth = -52.0             # 相机方位角 (°)
        self.elevation = 24.0            # 相机仰角 (°)
        self.distance = 1500.0           # 相机距离 (mm)
        self.show_grid = True
        self.show_home = True
        self.show_axis = True
        self.orthographic = True     # 正交投影：保证真实角度/高度比例
        self.show_scale = True       # 显示真实比例标尺
        self.show_coords = False     # 是否在各铰点显示坐标 (X,Y,Z)
        # 部位高亮（帮助弹窗用）：dict(kind, index) 或 None
        self.highlight = None
        self.highlight_label = ""

        self.setMouseTracking(True)
        self.setMinimumSize(320, 280)
        self.setCursor(Qt.OpenHandCursor)
        self._drag_pos = None
        self._pan_mode = False
        self._pan = np.array([0.0, 0.0])   # 屏幕平移偏移 (px)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._rotate_step)
        self._auto = False

    # ---------- 外部接口 ----------

    def set_theme(self, theme):
        self.theme = theme
        self.update()

    def set_model(self, solver, config):
        """绑定求解器与配置，提取铰点几何。"""
        self.solver = solver
        self.config = config
        if solver is not None:
            self.base = solver.base.copy()
            self.platform = solver.platform.copy()
        else:
            self.base, self.platform = config.anchors()
        self.home_pose = config.home_pose().copy()
        if self.pose is None:
            self.pose = self.home_pose.copy()
        # 距离自适应平台尺寸
        span = max(float(np.max(np.abs(self.base))), float(np.max(np.abs(self.platform))),
                   float(self.home_pose[2]), 1.0)
        self.distance = max(700.0, span * 3.0)
        self._fit_center = None     # 模型/参数变更：旧聚焦中心失效
        self.update()

    def set_pose(self, pose):
        """设置当前显示位姿 [x,y,z,roll,pitch,yaw]。"""
        if pose is None:
            return
        arr = np.asarray(pose, dtype=float).reshape(-1)
        if arr.size == 6:
            self.pose = arr.copy()
            self.update()

    def set_drag_pose_mode(self, on: bool):
        """开启/关闭拖拽平台模式：左键拖动直接改变上平台 X/Y 位置并发出 pose_dragged。"""
        self.drag_pose_mode = bool(on)
        if not self.drag_pose_mode:
            self._drag_pose_start = None
            self._drag_pose_base = None
        self.setCursor(Qt.OpenHandCursor if self.drag_pose_mode else Qt.ArrowCursor)
        self.setToolTip("拖拽平台模式：左键拖动平移上平台，Shift+拖动改高度" if self.drag_pose_mode else "")

    def set_force_ellipsoid(self, points=None):
        """设置力椭球点云（世界坐标 (N,3)）；传 None 清除。"""
        self.ellipsoid = None if points is None else np.asarray(points, dtype=float).reshape(-1, 3)
        self.update()

    def set_pointcloud(self, points=None, colors=None, fit=False):
        """设置工作空间点云。

        points 接受 (N,3) 位置或 (N,6) 绝对位姿（自动截取前 3 列）。
        严禁 reshape(-1,3)：那会把 (N,6) 拆成 2N 个点，角度列全 0
        会叠出一堆原点伪点，并与 (N,3) 颜色错位。
        colors 为 (N,3) RGB 0-1，数量须与点数一致。
        fit=True 时按『点云+机构』联合包围盒自动聚焦相机。
        """
        if points is None:
            self.pointcloud = None
            self.pointcolors = None
            self._fit_center = None
            self.last_pointcloud_error = None
            self.update()
            return

        arr = np.asarray(points, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError("点云必须是 (N,3) 或 (N,6)")
        arr3 = np.ascontiguousarray(arr[:, :3])
        finite = np.isfinite(arr3).all(axis=1)
        arr3 = arr3[finite]

        carr = None
        if colors is not None:
            carr = np.asarray(colors, dtype=float).reshape(-1, 3)
            if len(carr) != len(arr):
                raise ValueError("颜色数量与点数不一致")
            carr = np.ascontiguousarray(carr[finite])

        self.pointcloud = arr3
        self.pointcolors = carr
        self.last_pointcloud_error = None
        if fit:
            self.fit_camera_to_cloud()
        self.update()

    def fit_camera_to_cloud(self, margin=1.25):
        """按『点云 + 机构』联合包围盒聚焦相机，保证两者都在视野内。

        只改 _fit_center / distance / _pan，不改 azimuth / elevation / _auto，
        因此不与自动旋转、拖拽视角冲突（拖拽改的是 pose）。
        """
        if self.pointcloud is None or not len(self.pointcloud):
            return
        boxes = [self.pointcloud]
        if self.base is not None:
            boxes.append(self.base)
        if self.platform is not None:
            r = rot_matrix(*[float(x) for x in self.pose[3:6]]) \
                if self.pose is not None else np.eye(3)
            origin = self.pose[:3] if self.pose is not None else np.zeros(3)
            boxes.append(self.platform @ r.T + origin)
        if self.home_pose is not None:
            boxes.append(np.asarray(self.home_pose[:3], dtype=float).reshape(1, 3))
        allp = np.vstack(boxes)
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        span = float(np.max(hi - lo))
        if not np.isfinite(span) or span <= 0:
            return
        self._fit_center = (lo + hi) * 0.5
        self._pan[:] = 0.0
        scene = max(float(self._scene_scale()), 1.0)
        target_d = 1500.0 * span * margin / (scene * 2.0)
        self.distance = float(np.clip(target_d, 350.0, 12000.0))
        self.update()

    def set_highlight(self, key: str | None, label: str = ""):
        """设置朱红色部位高亮。key 为帮助系统里的 diagram 标识。"""
        self.highlight_label = label or ""
        if not key:
            self.highlight = None
            self.update()
            return
        h = {"kind": None, "index": -1}
        if key.startswith("leg") and key[3:].isdigit():
            h = {"kind": "leg", "index": int(key[3:]) - 1}
        elif key.startswith("base_point") and key[10:].isdigit():
            h = {"kind": "base_point", "index": int(key[10:]) - 1}
        elif key.startswith("platform_point") and key[14:].isdigit():
            h = {"kind": "platform_point", "index": int(key[14:]) - 1}
        elif key in ("base_radius", "base_pair_angle", "base_phase", "pair_centers",
                     "base_joint_z", "base_points", "base_joint_limit"):
            h = {"kind": "base_all", "index": -1}
        elif key in ("platform_radius", "platform_pair_angle", "platform_phase",
                     "platform_joint_z", "platform_points", "platform_joint_limit"):
            h = {"kind": "platform_all", "index": -1}
        elif key in ("home_height",):
            h = {"kind": "height", "index": -1}
        elif key in ("axis_x", "axis_y", "axis_z"):
            h = {"kind": "axis", "index": "xyz".index(key[-1])}
        elif key in ("legs", "stroke", "zero_lengths"):
            h = {"kind": "legs", "index": -1}
        self.highlight = h if h["kind"] else None
        self.update()

    def set_auto_rotate(self, enabled: bool):
        self._auto = bool(enabled)
        if self._auto:
            self._timer.start(50)
        else:
            self._timer.stop()

    @property
    def auto_rotate(self) -> bool:
        return self._auto

    def reset_view(self):
        self.azimuth, self.elevation = -52.0, 24.0
        self._pan = np.array([0.0, 0.0])
        span = max(float(np.max(np.abs(self.base))) if self.base is not None else 600.0,
                   float(self.home_pose[2]) if self.home_pose is not None else 600.0, 1.0)
        self.distance = max(700.0, span * 3.0)
        self._fit_center = None     # 重置视角：恢复默认取景中心
        self.update()

    def _rotate_step(self):
        self.azimuth = (self.azimuth + 0.35) % 360.0
        self.update()

    # ---------- 鼠标交互 ----------

    def mousePressEvent(self, ev):
        if ev.button() in (Qt.LeftButton, Qt.MiddleButton):
            self._drag_pos = ev.position()
            self._pan_mode = (ev.button() == Qt.MiddleButton)
            if self.drag_pose_mode and ev.button() == Qt.LeftButton and self.pose is not None:
                self._drag_pose_start = ev.position()
                self._drag_pose_base = self.pose.copy()
            self.setCursor(Qt.ClosedHandCursor)
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._drag_pos is not None:
            d = ev.position() - self._drag_pos
            self._drag_pos = ev.position()
            self.set_auto_rotate(False)
            if self.drag_pose_mode and self._drag_pose_base is not None:
                # 拖拽平台：按视口比例把像素位移换算成 mm（左键 X/Y，Shift+拖动改 Z）
                proj = self._camera(self.width(), self.height())
                # proj() 返回 (QPointF, depth)；必须保留 QPointF，
                # np.asarray(QPointF) 会得到 0 维 object 数组，.x() 抛 AttributeError
                o = proj(np.array([0.0, 0.0, 0.0]))[0]
                px = proj(np.array([10.0, 0.0, 0.0]))[0]
                # 阻尼 0.15：默认换算 mm_per_px≈5.58，拖 100px≈558mm 会瞬间超摆角限位
                mm_per_px = 10.0 / max(float(px.x() - o.x()), 1e-6) * 0.15
                total = ev.position() - self._drag_pose_start
                q = self._drag_pose_base.copy()
                if ev.modifiers() & Qt.ShiftModifier:
                    q[2] += float(total.y()) * mm_per_px
                else:
                    q[0] += float(total.x()) * mm_per_px
                    q[1] -= float(total.y()) * mm_per_px
                self.pose_dragged.emit(q)
            elif self._pan_mode:
                self._pan[0] += d.x()
                self._pan[1] += d.y()
            else:
                self.azimuth = (self.azimuth - d.x() * 0.55) % 360.0
                self.elevation = float(np.clip(self.elevation + d.y() * 0.45, -85.0, 85.0))
            self.update()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() in (Qt.LeftButton, Qt.MiddleButton):
            self._drag_pos = None
            self._pan_mode = False
            self._drag_pose_start = None
            self._drag_pose_base = None
            self.setCursor(Qt.OpenHandCursor if self.drag_pose_mode else Qt.ArrowCursor)
            ev.accept()
            return
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        self.reset_view()
        super().mouseDoubleClickEvent(ev)

    def wheelEvent(self, ev):
        delta = ev.angleDelta().y()
        if delta == 0:
            return
        factor = math.exp(-delta * 0.0012)
        self.distance = float(np.clip(self.distance * factor, 350.0, 12000.0))
        self.update()
        ev.accept()

    # ---------- 投影 ----------

    def _scene_scale(self):
        """真实场景尺寸（mm），用于等比例缩放与标尺。"""
        if self.base is None:
            return 1000.0
        span_xy = float(np.max(np.abs(self.base[:, :2])))
        span_p = float(np.max(np.abs(self.platform))) if self.platform is not None else span_xy
        height = float(self.home_pose[2]) if self.home_pose is not None else 600.0
        return max(span_xy, span_p, height, 1.0)

    def _camera(self, w, h):
        if self._fit_center is not None:
            target = self._fit_center.copy()
        else:
            target = np.array([0.0, 0.0,
                               float(self.home_pose[2]) * 0.5
                               if self.home_pose is not None else 300.0])
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)
        cam = target + self.distance * np.array([
            math.cos(el) * math.cos(az),
            math.cos(el) * math.sin(az),
            math.sin(el),
        ])
        forward = _norm(target - cam)
        right = _norm(_cross(forward, np.array([0.0, 0.0, 1.0])))
        up = _cross(right, forward)
        cx, cy = w * 0.5 + self._pan[0], h * 0.52 + self._pan[1]

        # 正交投影：固定比例，长度/高度/角度严格按真实几何等比缩放，不受深度影响
        if getattr(self, "orthographic", True):
            # 让真实场景 span 映射到视口约 78% 高度
            scale = (min(w, h) * 0.78) / (self._scene_scale() * 2.0)
            scale *= (1500.0 / max(self.distance, 1.0))  # 滚轮缩放仍生效

            def project(p):
                v = np.asarray(p, dtype=float) - target
                x, y, z = _dot(v, right), _dot(v, up), _dot(v, forward)
                return QPointF(cx + x * scale, cy - y * scale), z + self.distance

            return project

        # 透视投影（可选）
        focal = min(w, h) * 1.05

        def project(p):
            v = np.asarray(p, dtype=float) - cam
            x, y, z = _dot(v, right), _dot(v, up), _dot(v, forward)
            depth = max(z, 1.0)
            f = focal / depth
            return QPointF(cx + x * f, cy - y * f), depth

        return project

    # ---------- 绘制 ----------

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        t = self.theme
        if t is None:
            from theme import LIGHT
            t = LIGHT

        # 渐变背景
        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0.0, QColor(t.surface_alt))
        grad.setColorAt(1.0, QColor(t.bg))
        p.fillRect(self.rect(), QBrush(grad))

        if self.base is None or self.pose is None:
            p.setPen(QColor(t.text_3))
            f = QFont(); f.setPointSize(11); p.setFont(f)
            p.drawText(self.rect(), Qt.AlignCenter, "请先应用平台参数")
            return

        proj = self._camera(w, h)
        r = rot_matrix(float(self.pose[3]), float(self.pose[4]), float(self.pose[5]))
        tips = self.platform @ r.T + self.pose[:3]
        home_tips = self.platform + self.home_pose[:3]

        c_base = QColor(t.input)
        c_top = QColor(t.output)
        c_line = QColor(t.text_2)
        c_home = QColor(t.text_3)
        c_axis_x = QColor(t.danger)
        c_axis_y = QColor(t.success)
        c_axis_z = QColor(t.accent)
        # 部位高亮色：朱红（帮助系统专用，优先级高于语义色）
        c_hot = QColor(t.highlight)
        hl = self.highlight
        self._callouts = []   # 本帧的引线标注 (anchor, text)

        # 点云与力椭球移到机构绘制之后（见本函数末尾），确保不被机构线框遮挡。

        # 深度排序的绘制任务
        tasks = []

        # 地面网格
        if self.show_grid:
            span = max(float(np.max(np.abs(self.base[:, :2]))), 200.0) * 1.35
            grid_pen = QPen(QColor(t.grid)); grid_pen.setWidthF(0.7)
            p.setPen(grid_pen)
            for k in np.linspace(-span, span, 9):
                q1, _ = proj([k, -span, 0]); q2, _ = proj([k, span, 0])
                q3, _ = proj([-span, k, 0]); q4, _ = proj([span, k, 0])
                p.drawLine(q1, q2); p.drawLine(q3, q4)
            # 中心十字稍亮
            cp = QPen(QColor(t.border_strong)); cp.setWidthF(1.1)
            p.setPen(cp)
            for a, b in (([-span, 0, 0], [span, 0, 0]), ([0, -span, 0], [0, span, 0])):
                q1, _ = proj(a); q2, _ = proj(b)
                p.drawLine(q1, q2)

        # 零位参考平台（半透明虚线）
        if self.show_home:
            poly_home = QPolygonF([proj(x)[0] for x in home_tips])
            hp = QPen(c_home); hp.setWidthF(1.2); hp.setStyle(Qt.DashLine)
            p.setPen(hp); p.setBrush(Qt.NoBrush); p.drawPolygon(poly_home)
            hc, _ = proj(self.home_pose[:3])
            p.setPen(QColor(t.text_3))
            f = QFont(); f.setPointSize(9); p.setFont(f)
            p.drawText(hc + QPointF(6, -6), "零位")

        # 基座多边形（下节点/下平台相关部位高亮为红色）
        base_hot = hl is not None and hl["kind"] in ("base_all", "base_point")
        poly_b = QPolygonF([proj(x)[0] for x in self.base])
        p.setBrush(QBrush(QColor(c_base.red(), c_base.green(), c_base.blue(), 28)))
        bp = QPen(c_hot if base_hot else c_base)
        bp.setWidthF(3.6 if base_hot else 2.0)
        p.setPen(bp); p.drawPolygon(poly_b)
        if base_hot:
            # 半径引线：底座中心 → 第 1 铰点
            ctr = proj([0, 0, 0])[0]; ring = proj(self.base[0])[0]
            pen = QPen(c_hot); pen.setWidthF(2.4); pen.setStyle(Qt.DashLine)
            p.setPen(pen); p.drawLine(ctr, ring)
            self._callouts.append((QPointF((ctr.x()+ring.x())/2, (ctr.y()+ring.y())/2),
                                   self.highlight_label or "Rb 底座半径"))

        # 缸体（按深度排序，远的先画）
        leg_items = []
        for i in range(6):
            a, da = proj(self.base[i])
            b, db = proj(tips[i])
            leg_items.append(((da + db) / 2.0, i, a, b, da, db))
        leg_items.sort(key=lambda x: -x[0])

        for _, i, a, b, da, db in leg_items:
            is_hot = hl is not None and (
                (hl["kind"] == "leg" and hl["index"] == i) or hl["kind"] == "legs")
            # 缸筒（下 60%）
            mid = QPointF(a.x() + (b.x() - a.x()) * 0.58, a.y() + (b.y() - a.y()) * 0.58)
            pen = QPen(c_hot if is_hot else c_line)
            pen.setWidthF(7.0 if is_hot else 5.2); pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen); p.drawLine(a, mid)
            # 活塞杆（上 42%）
            rod = QColor(c_hot if is_hot else c_top)
            rod.setAlpha(210)
            pen = QPen(rod); pen.setWidthF(3.8 if is_hot else 2.6); pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen); p.drawLine(mid, b)
            # 端点球铰
            self._draw_ball(p, a, 6.4 if is_hot else 5.0, c_hot if is_hot else c_base)
            self._draw_ball(p, b, 6.0 if is_hot else 4.6, c_hot if is_hot else c_top)
            # 编号
            p.setPen(c_hot if is_hot else QColor(t.text_2))
            f = QFont(); f.setPointSize(9 if is_hot else 8); f.setBold(is_hot); p.setFont(f)
            p.drawText(QPointF((a.x() + b.x()) / 2 + 5, (a.y() + b.y()) / 2 - 4), f"L{i+1}")
            if is_hot and hl["kind"] == "leg":
                self._callouts.append((QPointF((a.x()+b.x())/2, (a.y()+b.y())/2),
                                       self.highlight_label or f"缸 {i+1}"))

        # 上平台（当前姿态）；上节点相关部位高亮为红色
        plat_hot = hl is not None and hl["kind"] in ("platform_all", "platform_point")
        poly_t = QPolygonF([proj(x)[0] for x in tips])
        p.setBrush(QBrush(QColor(c_top.red(), c_top.green(), c_top.blue(), 40)))
        tp = QPen(c_hot if plat_hot else c_top)
        tp.setWidthF(3.6 if plat_hot else 2.4); p.setPen(tp)
        p.drawPolygon(poly_t)
        for i, pt in enumerate(tips):
            q, _ = proj(pt)
            one_hot = plat_hot and (hl["kind"] == "platform_all" or hl["index"] == i)
            self._draw_ball(p, q, 6.0 if one_hot else 4.0, c_hot if one_hot else c_top)
            p.setPen(c_hot if one_hot else QColor(t.text_2))
            f = QFont(); f.setPointSize(9 if one_hot else 8); f.setBold(one_hot); p.setFont(f)
            label_text = f"P{i+1}"
            if self.show_coords:
                label_text += f"({pt[0]:.0f},{pt[1]:.0f},{pt[2]:.0f})"
            p.drawText(q + QPointF(5, -6), label_text)
        if plat_hot:
            ctr = proj(self.home_pose[:3])[0]; ring = proj(tips[0])[0]
            pen = QPen(c_hot); pen.setWidthF(2.4); pen.setStyle(Qt.DashLine)
            p.setPen(pen); p.drawLine(ctr, ring)
            self._callouts.append((QPointF((ctr.x()+ring.x())/2, (ctr.y()+ring.y())/2),
                                   self.highlight_label or "Rp 平台半径"))

        # 基座编号
        for i, pt in enumerate(self.base):
            q, _ = proj(pt)
            one_hot = base_hot and (hl["kind"] == "base_all" or hl["index"] == i)
            self._draw_ball(p, q, 6.0 if one_hot else 4.0, c_hot if one_hot else c_base)
            p.setPen(c_hot if one_hot else QColor(t.text_2))
            f = QFont(); f.setPointSize(9 if one_hot else 8); f.setBold(one_hot); p.setFont(f)
            label_text = f"B{i+1}"
            if self.show_coords:
                label_text += f"({pt[0]:.0f},{pt[1]:.0f},{pt[2]:.0f})"
            p.drawText(q + QPointF(-20, 4), label_text)

        # 坐标轴
        if self.show_axis:
            o, _ = proj([0, 0, 0])
            ax_len = max(float(np.max(np.abs(self.base[:, :2]))) * 0.42, 120.0)
            for idx, (vec, col, name) in enumerate((
                (np.array([ax_len, 0, 0]), c_axis_x, "X"),
                (np.array([0, ax_len, 0]), c_axis_y, "Y"),
                (np.array([0, 0, ax_len]), c_axis_z, "Z"),
            )):
                q, _ = proj(vec)
                is_hot = hl is not None and hl["kind"] == "axis" and hl["index"] == idx
                use = c_hot if is_hot else col
                pen = QPen(use); pen.setWidthF(3.4 if is_hot else 1.8); p.setPen(pen)
                p.drawLine(o, q)
                self._draw_ball(p, q, 5.4 if is_hot else 3.2, use)
                f = QFont(); f.setBold(True); f.setPointSize(11 if is_hot else 9); p.setFont(f)
                p.setPen(use)
                p.drawText(q + QPointF(4, -4), name)
                if is_hot:
                    self._callouts.append((q, self.highlight_label or f"{name} 轴"))

        # 高度 H0 高亮（底座中心 → 上平台中心）
        if hl is not None and hl["kind"] == "height":
            a = proj([0, 0, 0])[0]; b = proj(self.home_pose[:3])[0]
            pen = QPen(c_hot); pen.setWidthF(2.6); pen.setStyle(Qt.DashLine); p.setPen(pen)
            p.drawLine(a, b)
            self._draw_ball(p, a, 5.0, c_hot); self._draw_ball(p, b, 5.0, c_hot)
            self._callouts.append((QPointF((a.x()+b.x())/2, (a.y()+b.y())/2),
                                   self.highlight_label or "H0 零位高度"))

        # 引线标注（朱红色，功能指示）
        for anchor, text in getattr(self, "_callouts", []):
            self._callout(p, anchor, text, c_hot, w, h)

        # 真实比例标尺（正交模式下长度与真实 mm 等比）
        mode_text = "正交等比" if self.orthographic else "透视"
        if self.show_scale and self.orthographic:
            scale = (min(w, h) * 0.78) / (self._scene_scale() * 2.0)
            scale *= (1500.0 / max(self.distance, 1.0))
            # 选取 50/100/200mm 中长度适中的一档
            target_px = 70.0
            for mm in (50, 100, 200, 500):
                if mm * scale >= target_px:
                    break
            bar_px = mm * scale
            x0, y0 = 14, h - 38
            pen = QPen(QColor(t.text)); pen.setWidthF(2)
            p.setPen(pen)
            p.drawLine(QPointF(x0, y0), QPointF(x0 + bar_px, y0))
            p.drawLine(QPointF(x0, y0 - 4), QPointF(x0, y0 + 4))
            p.drawLine(QPointF(x0 + bar_px, y0 - 4), QPointF(x0 + bar_px, y0 + 4))
            f = QFont(); f.setPointSize(8); p.setFont(f)
            p.drawText(QPointF(x0, y0 - 7), f"{mm} mm")

        # ---- 工作空间点云：最后绘制 + 自身深度远→近 + 描边，确保明显可见 ----
        if (getattr(self, "show_pointcloud", False)
                and self.pointcloud is not None and len(self.pointcloud)):
            try:
                pts = self.pointcloud
                cols = self.pointcolors
                items = []
                for idx, v in enumerate(pts):
                    qp, depth = proj(v)
                    if np.isfinite(qp.x()) and np.isfinite(qp.y()):
                        items.append((depth, idx, qp))
                items.sort(key=lambda it: -it[0])   # 远的先画，近的覆盖

                # 主题感知描边：浅色主题深色描边，深色主题白描边
                bg = QColor(t.bg)
                outline = QColor("#0B1020") if bg.lightness() > 128 else QColor("#FFFFFF")
                outline.setAlpha(235)
                p.setPen(QPen(outline, 7.0, Qt.SolidLine))
                for _, _, qp in items:
                    p.drawPoint(qp)                  # 描边层 7px

                for _, idx, qp in items:
                    if cols is not None and idx < len(cols):
                        cr = cols[idx]
                        col = QColor(int(np.clip(cr[0], 0, 1) * 255),
                                     int(np.clip(cr[1], 0, 1) * 255),
                                     int(np.clip(cr[2], 0, 1) * 255), 205)
                    else:
                        col = QColor(t.accent); col.setAlpha(205)
                    p.setPen(QPen(col, 5.0, Qt.SolidLine))
                    p.drawPoint(qp)                  # 实心层 5px
                self.last_pointcloud_error = None
            except Exception as ex:
                self.last_pointcloud_error = str(ex)

        # 力椭球：紧跟点云之后，保持不透明橙色参照
        if self.ellipsoid is not None and len(self.ellipsoid):
            try:
                p.setPen(QPen(QColor(t.warning), 3.0, Qt.SolidLine))
                for v in self.ellipsoid:
                    qp = proj(v)[0]
                    p.drawPoint(qp)
            except Exception:
                pass

        # HUD
        p.setPen(QColor(t.text_2))
        f = QFont(); f.setPointSize(9); p.setFont(f)
        p.drawText(QRectF(10, 8, w - 20, 20), Qt.AlignLeft,
                   f"{mode_text} · 视角 方位 {self.azimuth:.0f}° 仰角 {self.elevation:.0f}°  "
                   f"零位高度 H0={self.home_pose[2]:.0f}mm")
        p.drawText(QRectF(10, h - 24, w - 20, 18), Qt.AlignRight,
                   "拖拽旋转 · 滚轮缩放 · 双击重置")

        # 边框
        p.setPen(QPen(QColor(t.border), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(QRectF(0.5, 0.5, w - 1, h - 1))

    def _callout(self, p: QPainter, anchor: QPointF, text: str,
                 color: QColor, w: int, h: int):
        """朱红引线 + 标签框，指向高亮部位。"""
        if not text:
            return
        # 引线折点：从锚点向右上引出
        tx = min(max(anchor.x() + 24, 10), w - 168)
        ty = min(max(anchor.y() - 34, 12), h - 52)
        tip = QPointF(tx, ty)
        pen = QPen(color); pen.setWidthF(1.4); p.setPen(pen)
        p.drawLine(anchor, tip)
        self._draw_ball(p, anchor, 3.4, color)
        box = QRectF(tx, ty - 20, min(160, w - tx - 8), 22)
        p.setBrush(QBrush(QColor(self.theme.surface if self.theme else "#FFFFFF")))
        p.setPen(QPen(color, 1.2))
        p.drawRoundedRect(box, 5, 5)
        p.setPen(color)
        f = QFont(); f.setPointSize(9); f.setBold(True); p.setFont(f)
        p.drawText(box, Qt.AlignCenter, text)

    def _draw_ball(self, p: QPainter, center: QPointF, radius: float, color: QColor):
        g = QRadialGradient(center, radius)
        light = QColor(color); light.setAlpha(230)
        dark = QColor(color); dark.setAlpha(110)
        g.setColorAt(0.0, light)
        g.setColorAt(1.0, dark)
        p.setBrush(QBrush(g))
        p.setPen(Qt.NoPen)
        p.drawEllipse(center, radius, radius)
