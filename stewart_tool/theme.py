# -*- coding: utf-8 -*-
"""主题系统 —— 印象派大气色光 + 毛玻璃层叠。

设计来源：
- Impeccable UX/UI：带色调的中性色、克制强调色、层级靠空间与字重。
- Taste Impressionism：边界"被暗示而非切分"、大气色光替代固有色、
  柔和自然色（鼠尾草绿 #4A6741 / 灰蓝 #6B8FAB / 赭金 #C8A96E）。

落地方式：
- 背景为柔和渐变 + 大色斑（见 atmos.py），卡片为半透明玻璃层，
  叠加后自然产生"纯色背景 + 模块轻微色差"的效果。
- 每个功能卡有独立 tint（暖金/冷蓝/青绿/中性），差异极轻微。
- 输入=冷蓝、输出=绿、部位高亮=朱红，语义色贯穿两模式。

关键修复延续：所有容器显式着色，避免透出系统深色底；同步 QPalette。
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QColor, QPalette

from profile_manager import app_data_dir

SETTINGS_FILE = app_data_dir() / "settings.json"


@dataclass(frozen=True)
class Theme:
    key: str
    name: str

    # ---- 基底与大气 ----
    bg: str
    atmo_warm: str          # 大气色斑：暖金
    atmo_cool: str          # 大气色斑：灰蓝
    atmo_green: str         # 大气色斑：鼠尾草绿

    # ---- 玻璃层 ----
    glass: str              # 卡片半透明底色 rgba
    glass_border: str       # 玻璃高光边框 rgba
    glass_hover: str
    surface: str            # 不透明表面（下拉/弹窗）
    surface_alt: str        # 次级填充 / 输入框
    raised: str
    border: str
    border_strong: str

    # ---- 模块微色差 ----
    tint_param: str         # 参数 → 暖金
    tint_input: str         # 输入 → 冷蓝
    tint_output: str        # 输出 → 青绿
    tint_view: str          # 三维 → 灰蓝

    # ---- 导航轨 ----
    sidebar: str
    sidebar_text: str
    sidebar_active: str

    # ---- 文字 ----
    text: str
    text_2: str
    text_3: str
    on_accent: str

    # ---- 语义 ----
    accent: str
    accent_hover: str
    accent_soft: str
    input: str
    output: str
    success: str
    warning: str
    danger: str
    highlight: str          # 部位指示 = 朱红
    grid: str


LIGHT = Theme(
    key="light", name="浅色",
    bg="#F7F8F9",                                   # 近白带微冷灰阶（印纸感）
    atmo_warm="#F0EBE1", atmo_cool="#E6EDF2", atmo_green="#E9F0E9",
    glass="rgba(255,255,255,0.62)", glass_border="rgba(255,255,255,0.78)",
    glass_hover="rgba(255,255,255,0.80)",
    surface="#FFFFFF", surface_alt="#F0F2F4", raised="#FFFFFF",
    border="#E4E8EB", border_strong="#CBD3D8",
    tint_param="rgba(200,169,110,0.07)", tint_input="rgba(107,143,171,0.08)",
    tint_output="rgba(74,131,103,0.07)", tint_view="rgba(96,126,150,0.06)",
    sidebar="#EFF2F4", sidebar_text="#5C6B75", sidebar_active="#1E2A33",
    text="#1A252D", text_2="#53616B", text_3="#8896A0", on_accent="#FFFFFF",
    accent="#4E7482", accent_hover="#3A5B68", accent_soft="#E2EBEE",
    input="#2E6E9E", output="#2F7D5B", success="#2F7D5B",
    warning="#A97628", danger="#B84940", highlight="#D63B32", grid="#E2E7EA",
)

DARK = Theme(
    key="dark", name="深色",
    bg="#0E1012",                                   # 近黑带微冷阶（暗场感）
    atmo_warm="#1B1916", atmo_cool="#141A20", atmo_green="#131C19",
    glass="rgba(28,33,38,0.60)", glass_border="rgba(150,175,190,0.10)",
    glass_hover="rgba(38,45,52,0.78)",
    surface="#181D22", surface_alt="#222A31", raised="#262F36",
    border="#2C353C", border_strong="#45535C",
    tint_param="rgba(200,169,110,0.05)", tint_input="rgba(107,143,171,0.07)",
    tint_output="rgba(74,131,103,0.07)", tint_view="rgba(96,126,150,0.06)",
    sidebar="#13181C", sidebar_text="#93A3AD", sidebar_active="#EDF3F6",
    text="#E8EEF2", text_2="#A5B4BE", text_3="#788992", on_accent="#FFFFFF",
    accent="#7BA3B2", accent_hover="#98BCC8", accent_soft="#243940",
    input="#85B8DC", output="#77C9A4", success="#77C9A4",
    warning="#DCAF63", danger="#E88078", highlight="#F0574C", grid="#252E35",
)

THEMES = {"light": LIGHT, "dark": DARK}


# 玻璃质感三模式：推荐(明显+柔光) / 强(更透更亮) / 极简(近纯色无阴影)
GLASS_MODES = ("recommended", "strong", "minimal")
GLASS_MODE_NAMES = {"recommended": "推荐 · 液态玻璃", "strong": "强 · 通透玻璃", "minimal": "极简 · 扁平"}

# 每模式：卡片背景透明度覆写、阴影、卡片圆角、侧栏透明度
_GLASS_PARAMS = {
    "recommended": {"card_alpha": None, "shadow": True, "radius": 14, "side_alpha": None},
    "strong": {"card_alpha": 0.42, "shadow": True, "radius": 16, "side_alpha": 0.72},
    "minimal": {"card_alpha": 0.97, "shadow": False, "radius": 10, "side_alpha": 0.98},
}


def _rgba(hex_color: str, alpha: float) -> str:
    """把 #RRGGBB 转成 rgba(...) 字符串。"""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def current_glass_mode() -> str:
    """从 settings.json 读取当前玻璃模式（默认 recommended）。"""
    try:
        if SETTINGS_FILE.exists():
            mode = json.loads(SETTINGS_FILE.read_text(encoding="utf-8")).get("glass_mode")
            if mode in GLASS_MODES:
                return mode
    except Exception:
        pass
    return "recommended"


def build_style(t: Theme, mode: str = None) -> str:
    """完整 QSS：显式背景防串色 + 液态玻璃卡片 + 模块微色差。

    mode: recommended / strong / minimal；None 时自动从设置读取。
    """
    if mode is None:
        mode = current_glass_mode()
    gp = _GLASS_PARAMS.get(mode, _GLASS_PARAMS["recommended"])
    # 按模式推导颜色
    if gp["card_alpha"] is None:
        card_bg, side_bg = t.glass, t.sidebar
    else:
        # 用 base 色 + 指定透明度重建玻璃层
        base = "#FFFFFF" if t.key == "light" else "#1C2329"
        side_base = "#FFFFFF" if t.key == "light" else "#13181C"
        card_bg = _rgba(base, gp["card_alpha"])
        side_bg = _rgba(side_base, gp["side_alpha"])
    # 阴影与顶部柔光：三模式差异明显（推荐=柔光+中影，强=强高光+深影，极简=无光无影）
    if mode == "strong":
        shadow = ("box-shadow: 0 10px 34px rgba(0,0,0,0.34);" if t.key == "dark"
                  else "box-shadow: 0 10px 30px rgba(30,42,51,0.20);")
        top_glow = f"border-top: 2px solid {t.glass_border};"
        card_border = f"1px solid {t.glass_border}"
    elif mode == "recommended":
        shadow = ("box-shadow: 0 6px 22px rgba(0,0,0,0.18);" if t.key == "dark"
                  else "box-shadow: 0 6px 20px rgba(30,42,51,0.12);")
        top_glow = f"border-top: 1px solid {t.glass_border};"
        card_border = f"1px solid {t.glass_border}"
    else:  # minimal
        shadow = ""
        top_glow = ""
        card_border = "none"
    radius = gp["radius"]
    return f"""
* {{
    font-family: 'Microsoft YaHei UI', 'Microsoft YaHei', 'Segoe UI', sans-serif;
    font-size: 13px;
}}

/* 所有容器显式着色（浅色发黑的根因修复） */
QMainWindow, QDialog, QWidget, QFrame, QStackedWidget, QScrollArea {{
    background: {t.bg};
    color: {t.text};
}}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QWidget#atmos {{ background: {t.bg}; }}
/* 前景层与页面透明，让大气色光透过玻璃卡片 */
QWidget#foreground, QWidget#page, QStackedWidget,
QScrollArea, QScrollArea > QWidget {{ background: transparent; }}

/* ---------- 液态玻璃卡片：边界被暗示，不被切分 ---------- */
QFrame#card, QFrame#hero {{
    background: {card_bg};
    border: {card_border};
    {top_glow}
    border-radius: {radius}px;
    {shadow}
}}
QFrame#card:hover {{ background: {t.glass_hover}; }}
QFrame#hero {{ border-left: 3px solid {t.accent}; }}
/* 模块微色差（叠在玻璃上，同色系极轻微） */
QFrame#tintParam {{ background: {t.tint_param}; border-radius: {radius}px; {shadow} }}
QFrame#tintInput {{ background: {t.tint_input}; border-radius: {radius}px; {shadow} }}
QFrame#tintOutput {{ background: {t.tint_output}; border-radius: {radius}px; {shadow} }}
QFrame#tintView {{ background: {t.tint_view}; border-radius: {radius}px; {shadow} }}

QFrame#divider {{ background: {t.border}; max-height: 1px; min-height: 1px; border: none; }}
QFrame#segContainer {{
    background: {t.surface_alt}; border: none;
    border-radius: 8px; padding: 2px;
}}

/* ---------- 导航轨（同色系玻璃，无高亮边框） ---------- */
QFrame#sidebar {{ background: {side_bg}; border: none; }}
QLabel#brandName {{ color: {t.sidebar_active}; font-size: 15px; font-weight: 700; }}
QLabel#brandSub {{ color: {t.sidebar_text}; font-size: 11px; }}
QLabel#navSection {{ color: {t.sidebar_text}; font-size: 10px; font-weight: 700; }}
QPushButton#navItem {{
    background: transparent; color: {t.sidebar_text};
    border: none; border-left: 3px solid transparent; border-radius: 0;
    padding: 11px 14px 11px 16px; text-align: left;
    font-size: 13px; font-weight: 500;
}}
QPushButton#navItem:hover {{ background: {t.surface_alt}; color: {t.text}; }}
QPushButton#navItem:checked {{
    background: {t.accent}; color: {t.on_accent};
    border-left: 3px solid {t.output}; font-weight: 700;
}}

/* ---------- 顶栏 ---------- */
QFrame#topbar {{
    background: {card_bg}; border: none;
    border-bottom: 1px solid {t.border};
}}
/* ---------- 自绘标题栏（无边框窗口） ---------- */
QFrame#titlebar {{
    background: {side_bg}; border: none; border-bottom: 1px solid {t.border};
}}
QFrame#titleDot {{ background: {t.accent}; border-radius: 4px; }}
QLabel#titleText {{ color: {t.text_2}; font-size: 12px; font-weight: 600; }}
QPushButton#winBtn, QPushButton#winBtnClose {{
    background: transparent; border: none; border-radius: 6px;
    color: {t.text_2}; font-size: 17px; font-weight: 400;
    padding: 0; min-width: 28px; max-width: 28px; min-height: 26px; max-height: 26px;
}}
QPushButton#winBtn:hover {{ background: {t.glass_hover}; color: {t.text}; }}
QPushButton#winBtnClose:hover {{ background: {t.danger}; color: #FFFFFF; }}

QLabel#pageTitle {{ color: {t.text}; font-size: 18px; font-weight: 700; }}
QLabel#pageSub {{ color: {t.text_2}; font-size: 11px; }}
QLabel#badge {{
    color: {t.accent}; background: {t.accent_soft};
    border: none; border-radius: 11px;
    padding: 4px 10px; font-size: 11px; font-weight: 700;
}}
QLabel#debugBadge {{
    color: {t.warning}; background: {t.accent_soft};
    border: 1px solid {t.warning}; border-radius: 11px;
    padding: 4px 10px; font-size: 11px; font-weight: 700;
}}

/* ---------- 文字层级 ---------- */
QLabel {{ color: {t.text}; background: transparent; }}
QLabel#largeTitle {{ color: {t.text}; font-size: 25px; font-weight: 700; }}
QLabel#title {{ color: {t.text}; font-size: 16px; font-weight: 700; }}
QLabel#headline {{ color: {t.text}; font-size: 13px; font-weight: 700; }}
QLabel#body {{ color: {t.text}; font-size: 13px; }}
QLabel#caption {{ color: {t.text_2}; font-size: 11px; }}
QLabel#sectionTitle {{ color: {t.text_3}; font-size: 10px; font-weight: 700; }}
QLabel#inValue {{ color: {t.input}; font-weight: 700; font-family: 'Consolas', monospace; }}
QLabel#outValue {{ color: {t.output}; font-size: 21px; font-weight: 700; font-family: 'Consolas', monospace; }}
QLabel#unit {{ color: {t.text_3}; font-size: 10px; }}

/* ---------- 输入控件（玻璃上的凹槽）---------- */
QLineEdit, QPlainTextEdit, QComboBox {{
    background: {t.surface_alt}; border: none;
    border-radius: 7px; padding: 7px 10px; color: {t.input}; font-size: 13px;
    min-height: 22px;
    font-family: 'Consolas', 'Microsoft YaHei', monospace;
    selection-background-color: {t.accent}; selection-color: {t.on_accent};
}}
QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus {{
    background: {t.surface};
}}
/* 参与『各动作最大值』计算的输入项：淡色底 + 粗体（无高亮边框） */
QLineEdit#calcInput {{
    background: {t.accent_soft};
    color: {t.input};
    font-weight: 700;
}}
QLineEdit::placeholder, QPlainTextEdit::placeholder {{ color: {t.text_3}; }}
QComboBox {{ font-family: 'Microsoft YaHei UI', 'Microsoft YaHei', sans-serif; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background: {t.raised}; border: none;
    border-radius: 7px; padding: 4px;
    selection-background-color: {t.accent}; selection-color: {t.on_accent}; outline: none;
}}

/* ---------- 按钮 ---------- */
QPushButton {{
    background: {t.glass}; color: {t.text}; border: none;
    border-radius: 7px; padding: 8px 14px; font-size: 12px; font-weight: 600;
    min-height: 20px;
}}
QPushButton:hover {{ background: {t.glass_hover}; }}
QPushButton:pressed {{ background: {t.accent_soft}; }}
QPushButton:disabled {{ color: {t.text_3}; }}
QPushButton#primary {{
    background: {t.accent}; color: {t.on_accent}; border: none;
    border-radius: 8px; padding: 11px 20px; font-size: 13px; font-weight: 700;
}}
QPushButton#primary:hover {{ background: {t.accent_hover}; }}
QPushButton#danger {{ color: {t.danger}; border-color: {t.border}; }}
QPushButton#ghost {{ background: transparent; border: none; color: {t.accent}; padding: 6px 9px; font-size: 12px; }}
QPushButton#segment {{ background: transparent; color: {t.text_2}; border: none; border-radius: 6px; padding: 7px 13px; font-size: 12px; }}
QPushButton#segment:checked {{ background: {t.surface}; color: {t.accent}; font-weight: 700; }}
QPushButton#themeBtn {{
    background: {t.glass}; border: 1px solid {t.border};
    border-radius: 7px; padding: 6px 11px; font-size: 11px;
}}

/* ---------- 问号 ---------- */
QToolButton#hint {{
    background: transparent; color: {t.accent};
    border: none; border-radius: 8px;
    font-size: 11px; font-weight: 700;
    min-width: 17px; max-width: 17px; min-height: 17px; max-height: 17px; padding: 0;
}}
QToolButton#hint:hover {{ background: {t.accent_soft}; color: {t.accent}; }}

/* ---------- 表格（玻璃层，无高亮边框） ---------- */
QTableWidget {{
    background: {card_bg}; border: none;
    border-radius: 9px; gridline-color: {t.border};
}}
QHeaderView::section {{
    background: {t.surface_alt}; color: {t.text_2}; font-size: 11px; font-weight: 700;
    border: none; padding: 8px 4px;
}}
QTableWidget::item {{ padding: 6px 5px; border-bottom: 1px solid {t.border}; }}
QTableWidget::item:selected {{ background: {t.accent_soft}; color: {t.text}; }}

QCheckBox {{ font-size: 12px; color: {t.text}; spacing: 7px; }}
QCheckBox::indicator {{
    width: 15px; height: 15px; border-radius: 4px;
    border: 1px solid {t.border_strong}; background: {t.surface};
}}
QCheckBox::indicator:checked {{ background: {t.accent}; border-color: {t.accent}; }}

QScrollBar:vertical {{ background: transparent; width: 9px; }}
QScrollBar::handle:vertical {{ background: {t.border_strong}; border-radius: 4px; min-height: 32px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 9px; }}
QScrollBar::handle:horizontal {{ background: {t.border_strong}; border-radius: 4px; min-width: 32px; }}
QToolTip {{
    background: {t.raised}; color: {t.text}; border: 1px solid {t.border_strong};
    border-radius: 6px; padding: 6px 9px; font-size: 12px;
}}
"""


def apply_palette(app, theme: Theme):
    """同步 QPalette，让原生对话框 / 菜单跟随主题。"""
    p = QPalette()
    for role, color in (
        (QPalette.Window, theme.bg), (QPalette.WindowText, theme.text),
        (QPalette.Base, theme.surface), (QPalette.AlternateBase, theme.surface_alt),
        (QPalette.Text, theme.text), (QPalette.Button, theme.surface),
        (QPalette.ButtonText, theme.text), (QPalette.Highlight, theme.accent),
        (QPalette.HighlightedText, theme.on_accent),
        (QPalette.ToolTipBase, theme.raised), (QPalette.ToolTipText, theme.text),
    ):
        p.setColor(role, QColor(color))
    p.setColor(QPalette.Disabled, QPalette.Text, QColor(theme.text_3))
    app.setPalette(p)


class ThemeManager(QObject):
    """主题管理器：深浅双主题 + 三种玻璃质感，均持久化到 settings.json。"""

    changed = Signal(object)

    def __init__(self):
        super().__init__()
        self._key = "dark"
        self._glass_mode = "recommended"
        try:
            if SETTINGS_FILE.exists():
                data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
                if data.get("theme") in THEMES:
                    self._key = data["theme"]
                if data.get("glass_mode") in GLASS_MODES:
                    self._glass_mode = data["glass_mode"]
        except Exception:
            pass

    @property
    def theme(self) -> Theme:
        return THEMES[self._key]

    @property
    def key(self) -> str:
        return self._key

    @property
    def glass_mode(self) -> str:
        return self._glass_mode

    def set(self, key: str):
        if key in THEMES and key != self._key:
            self._key = key
            self._persist()
            self.changed.emit(self.theme)

    def set_glass_mode(self, mode: str):
        if mode in GLASS_MODES and mode != self._glass_mode:
            self._glass_mode = mode
            self._persist()
            self.changed.emit(self.theme)

    def toggle(self):
        self.set("light" if self._key == "dark" else "dark")

    def _persist(self):
        try:
            data = {}
            if SETTINGS_FILE.exists():
                data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            data["theme"] = self._key
            data["glass_mode"] = self._glass_mode
            SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
