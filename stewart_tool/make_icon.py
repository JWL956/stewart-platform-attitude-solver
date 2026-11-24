# -*- coding: utf-8 -*-
"""生成应用图标 app.ico（六自由度平台抽象标识）。

用 PySide6 的 QImage/QPainter 绘制，不依赖 Pillow。
输出 256/128/64/48/32/16 多尺寸 ICO。
"""

from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QImage, QPainter, QColor, QPen, QBrush, QPolygonF, QFont

OUT = Path(__file__).resolve().parent / "app.ico"


def render(size: int) -> QImage:
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing, True)
    s = size / 100.0
    p.scale(s, s)

    # 深蓝底圆角方块
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(QColor("#25313B")))
    p.drawRoundedRect(2, 2, 96, 96, 20, 20)

    # 鼠尾草绿外六边形
    def hexpoly(cx, cy, r, rot=-math.pi / 2):
        return QPolygonF([QPointF(cx + r * math.cos(rot + i * math.pi / 3),
                                  cy + r * math.sin(rot + i * math.pi / 3))
                          for i in range(6)])

    p.setBrush(Qt.NoBrush)
    pen = QPen(QColor("#A0C0A0")); pen.setWidthF(3.2); p.setPen(pen)
    p.drawPolygon(hexpoly(50, 50, 34))

    # 灰蓝内六边形
    pen = QPen(QColor("#6B8FAB")); pen.setWidthF(2.6); p.setPen(pen)
    p.drawPolygon(hexpoly(50, 50, 19))

    # 中心赭金点
    p.setBrush(QBrush(QColor("#C8A96E"))); p.setPen(Qt.NoPen)
    p.drawEllipse(QPointF(50, 50), 6, 6)

    # 6 根支链（外六角顶点 → 内六角对应顶点）
    pen = QPen(QColor("#E8D5A3")); pen.setWidthF(1.8); p.setPen(pen)
    outer = hexpoly(50, 50, 34); inner = hexpoly(50, 50, 19)
    for i in range(6):
        p.drawLine(outer[i], inner[i])

    p.end()
    return img


def write_ico(path: Path, images: list[QImage]):
    """手写 ICO 容器（PNG 压缩帧），避免平台相关图标库。"""
    import struct
    import tempfile
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries, blobs = [], []
    for img in images:
        sz = img.width()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tmp = tf.name
        img.save(tmp, "PNG")
        data = Path(tmp).read_bytes()
        Path(tmp).unlink()
        entries.append(struct.pack("<BBBBHHII", sz % 256, sz % 256, 0, 0, 1, 32,
                                   len(data), offset))
        blobs.append(data)
        offset += len(data)
    path.write_bytes(header + b"".join(entries) + b"".join(blobs))


if __name__ == "__main__":
    imgs = [render(s) for s in (256, 128, 64, 48, 32, 16)]
    write_ico(OUT, imgs)
    print("图标已生成:", OUT, OUT.stat().st_size, "bytes")
