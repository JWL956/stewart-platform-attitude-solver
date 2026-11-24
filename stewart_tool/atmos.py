# -*- coding: utf-8 -*-
"""大气背景层 —— 印象派色光。

在纯色底上叠加几团极柔和的径向色斑（暖金 / 灰蓝 / 鼠尾草绿），
边界被刻意虚化（"boundary suggested rather than defined"），
为上层半透明玻璃卡片提供可透出的光感，而不是平板纯色。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QColor, QPainter, QRadialGradient, QLinearGradient, QBrush
from PySide6.QtWidgets import QWidget


def paint_atmosphere(p: QPainter, rect, theme, w: int, h: int):
    """在 painter 上绘制印象派大气色光。"""
    p.save()
    t = theme

    # 基底：极轻的垂直渐变（大气纵深）
    base = QLinearGradient(0, 0, 0, max(h, 1))
    base.setColorAt(0.0, QColor(t.bg))
    base.setColorAt(1.0, QColor(t.surface_alt))
    p.fillRect(rect, QBrush(base))

    # 三团大气色斑，透明度低、半径大、互相交融
    boost = 1.25 if t.key == "light" else 1.0
    blobs = [
        (0.08, 0.00, 0.78, t.atmo_warm, min(0.9, 0.55 * boost)),
        (0.92, 0.08, 0.70, t.atmo_cool, min(0.9, 0.50 * boost)),
        (0.55, 0.92, 0.85, t.atmo_green, min(0.9, 0.42 * boost)),
    ]
    for fx, fy, fr, color, alpha in blobs:
        cx, cy = w * fx, h * fy
        radius = max(w, h) * fr
        g = QRadialGradient(QPointF(cx, cy), radius)
        c = QColor(color); c.setAlphaF(alpha)
        g.setColorAt(0.0, c)
        c0 = QColor(color); c0.setAlphaF(0.0)
        g.setColorAt(1.0, c0)
        p.fillRect(rect, QBrush(g))

    # 极淡中心提亮，强化玻璃后的光感
    glow = QRadialGradient(QPointF(w * 0.42, h * 0.35), max(w, h) * 0.6)
    gc = QColor(t.raised); gc.setAlphaF(0.28 if t.key == "light" else 0.10)
    glow.setColorAt(0.0, gc)
    gc0 = QColor(t.raised); gc0.setAlphaF(0.0)
    glow.setColorAt(1.0, gc0)
    p.fillRect(rect, QBrush(glow))
    p.restore()


class FluidPage(QWidget):
    """自带印象派大气背景的容器。直接作为内容区使用。"""

    def __init__(self, theme, parent=None):
        super().__init__(parent)
        self.theme = theme
        self.setObjectName("foreground")

    def set_theme(self, theme):
        self.theme = theme
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        paint_atmosphere(p, self.rect(), self.theme, self.width(), self.height())
        super().paintEvent(ev)
