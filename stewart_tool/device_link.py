# -*- coding: utf-8 -*-
"""开发板连接检测（可插拔）。

默认假设：串口 COM + 心跳包，超时 2 秒。
- 装了 pyserial 且有串口 → 真实检测：后台线程周期发心跳字节，收到应答=已连接，超时=未连接。
- 缺 pyserial / 未配置串口 → 模拟模式：默认无信号(黄)；需 inject_signal(True) 注入信号才显示绿色。
  绿色(CONNECTED/TRANSMITTING)只由真实心跳应答或注入信号驱动，手动演示切态不产生绿色。

线程模型：串口是独占长驻资源，所有 write/read 集中在单一 QThread（SerialWorker）内，
由 worker 自己的 QTimer 驱动心跳与下发队列；主线程只通过信号投递请求/接收状态，
不再在 QTimer 回调里 read() 阻塞 UI。对外 API 保持不变。

真实协议可替换点：SerialWorker._heartbeat_once() —— 换成你开发板的实际握手/心跳即可，
其余状态机、UI、信号机制不用动。
"""

from __future__ import annotations

import time
from collections import deque
from enum import Enum

from PySide6.QtCore import QObject, Signal, Slot, QTimer, QThread, Qt


class LinkState(Enum):
    UNCONNECTED = "未连接开发板"      # 黄
    CONNECTED = "已连接"              # 绿常亮
    TRANSMITTING = "传输正常"         # 绿呼吸
    FAILED = "传输失败"               # 红呼吸


DEFAULT_HEARTBEAT = b"?\n"       # 默认心跳查询（可替换）
HEARTBEAT_TIMEOUT = 2.0          # 秒
DEFAULT_BAUD = 115200
HEARTBEAT_INTERVAL = 600         # ms
QUEUE_MAX = 512                  # 下发队列上限，防止轨迹播放堆积


class SerialWorker(QObject):
    """串口 IO 工作线程：心跳与下发全部在此串行执行，不阻塞 UI。"""

    opened = Signal(bool, str)            # (成功, 错误信息)
    replied = Signal(float)               # 心跳应答时间戳（monotonic）
    sent = Signal(bool, str)              # (成功, 错误信息)
    queueSignal = Signal(bytes)           # 主线程投递下发帧
    enqueueResult = Signal(bool)          # 入队结果回传主线程

    def __init__(self, port: str = "", baud: int = DEFAULT_BAUD):
        super().__init__()
        self.port = port
        self.baud = baud
        self.serial = None
        self._queue = deque()
        self._timer = None            # 工作线程内的心跳定时器
        self.queueSignal.connect(self._on_enqueue, Qt.QueuedConnection)

    @property
    def is_real(self) -> bool:
        return self.serial is not None

    @Slot()
    def setup(self):
        """在工作线程内执行：打开串口并启动心跳定时器。"""
        # 恢复时可能已有旧定时器（stop 已 teardown，这里再清理一次确保幂等）
        if self._timer is not None:
            self._timer.stop()
            self._timer.deleteLater()
            self._timer = None
        if not self.port:
            self.opened.emit(False, "未配置串口（模拟模式）")
            return
        try:
            import serial                # 惰性导入，缺库不崩
            self.serial = serial.Serial(self.port, self.baud, timeout=0.2)
            self.opened.emit(True, "")
        except ImportError:
            self.serial = None
            self.opened.emit(False, "未安装 pyserial（模拟模式）")
        except Exception as ex:
            self.serial = None
            self.opened.emit(False, f"串口打开失败：{ex}")
        # 定时器必须在工作线程内创建，timeout 回调才在工作线程执行
        self._timer = QTimer(self)
        self._timer.setInterval(HEARTBEAT_INTERVAL)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    @Slot()
    def teardown(self):
        """在工作线程内执行：停定时器、关串口。"""
        if self._timer is not None:
            self._timer.stop()
        if self.serial is not None:
            try:
                self.serial.close()
            except Exception:
                pass
            self.serial = None
        self._queue.clear()

    def _tick(self):
        """工作线程节拍：心跳 + 排空下发队列。"""
        if self.serial is not None:
            self._heartbeat_once()
        self._drain_queue()

    def _heartbeat_once(self):
        """真实协议替换点：发心跳、看应答。仅在工作线程调用。"""
        if self.serial is None:
            return
        try:
            self.serial.write(DEFAULT_HEARTBEAT)
            reply = self.serial.read(self.serial.in_waiting or 1)
            if reply:
                self.replied.emit(time.monotonic())
        except Exception:
            pass

    def _on_enqueue(self, payload: bytes):
        """工作线程内入队（由主线程 queueSignal 投递）。"""
        if len(self._queue) >= QUEUE_MAX:
            self.enqueueResult.emit(False)
            return
        self._queue.append(payload)
        self.enqueueResult.emit(True)

    def _drain_queue(self):
        """串行发送队列中的帧。仅在工作线程调用。"""
        while self._queue:
            payload = self._queue.popleft()
            if self.serial is None:
                self.sent.emit(False, "串口未连接")
                continue
            try:
                self.serial.write(payload)
                reply = self.serial.read(self.serial.in_waiting or 1)
                if reply:
                    self.replied.emit(time.monotonic())
                self.sent.emit(True, "")
            except Exception as ex:
                self.sent.emit(False, f"传输失败：{ex}")


class DeviceLinkDetector(QObject):
    """连接状态机 + 后台串口线程。状态变化通过 state_changed 发给 UI。

    对外 API 与旧版一致：start/stop/transmit/inject_signal/simulate/state/is_real/timer。
    """

    state_changed = Signal(object, str)   # (LinkState, message)

    def __init__(self, port: str = "", baud: int = DEFAULT_BAUD, parent=None):
        super().__init__(parent)
        self.port = port
        self.baud = baud
        self._state = LinkState.UNCONNECTED
        self._sim_has_signal = False   # 仅模拟模式：绿色只由此标志或真实心跳驱动
        self._last_reply = 0.0
        self._serial_err = ""

        # 兼容旧代码：app.py 的 _set_debug_visible 用 det.timer.isActive()
        self.timer = QTimer(self)
        self.timer.setInterval(HEARTBEAT_INTERVAL)
        self.timer.timeout.connect(self._on_heartbeat_tick)

        # 工作线程与 worker
        self._thread = QThread(self)
        self.worker = SerialWorker(port, baud)
        self.worker.moveToThread(self._thread)
        self._thread.started.connect(self.worker.setup)
        self.worker.opened.connect(self._on_opened)
        self.worker.replied.connect(self._on_replied)
        self.worker.sent.connect(self._on_sent)
        self.worker.enqueueResult.connect(self._on_enqueue_result)

    @property
    def is_real(self) -> bool:
        return self.worker is not None and self.worker.is_real

    @property
    def state(self) -> LinkState:
        return self._state

    def _set_state(self, state: LinkState, msg: str = ""):
        if state != self._state:
            self._state = state
            self.state_changed.emit(state, msg or state.value)

    # ---- 主线程槽：只做状态判定，不碰串口 ----
    def _on_opened(self, ok: bool, err: str):
        self._serial_err = err if not ok else ""

    def _on_replied(self, ts: float):
        self._last_reply = ts
        if self._state in (LinkState.UNCONNECTED, LinkState.FAILED):
            self._set_state(LinkState.CONNECTED, "开发板心跳正常")

    def _on_sent(self, ok: bool, err: str):
        if ok:
            self._set_state(LinkState.TRANSMITTING, "正在传输运动数据…")
        else:
            self._set_state(LinkState.FAILED, err)

    def _on_enqueue_result(self, ok: bool):
        if not ok:
            self._set_state(LinkState.FAILED, "下发队列已满")

    def _on_heartbeat_tick(self):
        """主线程节拍：只做超时判定（真实心跳/下发由 worker 线程定时器驱动）。"""
        now = time.monotonic()
        if self.is_real:
            connected = (now - self._last_reply) < HEARTBEAT_TIMEOUT
        else:
            connected = self._sim_has_signal
        if connected:
            if self._state in (LinkState.UNCONNECTED, LinkState.FAILED):
                self._set_state(LinkState.CONNECTED,
                                "开发板心跳正常" if self.is_real else "模拟信号正常")
        else:
            if self._state not in (LinkState.UNCONNECTED, LinkState.FAILED):
                self._set_state(LinkState.UNCONNECTED,
                                self._serial_err or "开发板无应答（超时 2 秒）")

    # ---- 对外 API ----
    def start(self):
        # 线程只启动一次（QThread 不可重启）；stop/start 是"暂停/恢复"语义。
        if not self._thread.isRunning():
            self._thread.start()
        else:
            from PySide6.QtCore import QMetaObject
            QMetaObject.invokeMethod(self.worker, "setup", Qt.QueuedConnection)
        self.timer.start()

    def stop(self):
        self.timer.stop()
        if self._thread.isRunning():
            # 只停定时器和串口，不退出线程，保证再次 start() 可恢复
            from PySide6.QtCore import QMetaObject
            QMetaObject.invokeMethod(self.worker, "teardown", Qt.QueuedConnection)
        self._set_state(LinkState.UNCONNECTED, "检测已停止")

    def transmit(self, data=None) -> bool:
        """下发一帧数据：入队即返回；实际串口写在工作线程串行执行。

        返回 True 表示已成功入队（真实模式）或有模拟信号（模拟模式），
        False 表示链路未就绪/队列满，调用方据此提示用户。
        """
        payload = (str(data) + "\n").encode("utf-8") if data is not None else DEFAULT_HEARTBEAT
        if self.is_real:
            # 通过信号跨线程投递入队请求，不阻塞主线程
            self.worker.queueSignal.emit(payload)
            self._set_state(LinkState.TRANSMITTING, "正在传输运动数据…")
            return True
        # 模拟模式：只有注入了信号才认为可下发
        if self._sim_has_signal:
            self._set_state(LinkState.TRANSMITTING, "模拟传输中（有信号）")
            return True
        self._set_state(LinkState.FAILED, "无信号，无法传输")
        return False

    def inject_signal(self, on: bool = True):
        """仅模拟模式：注入/撤销『有信号』。绿色只由此或真实心跳驱动。"""
        if self.is_real:
            return
        self._sim_has_signal = bool(on)
        self._set_state(LinkState.CONNECTED if on else LinkState.UNCONNECTED,
                        "模拟信号已注入" if on else "模拟信号已撤销")

    def simulate(self, state: LinkState, message: str = ""):
        """纯演示切态：只改展示状态，不改 _sim_has_signal（因此不会产生假绿）。"""
        if not self.is_real:
            self._set_state(state, message or state.value)

    def shutdown(self):
        """彻底退出工作线程（程序关闭时调用，不可再 start）。"""
        self.timer.stop()
        if self._thread.isRunning():
            from PySide6.QtCore import QMetaObject
            QMetaObject.invokeMethod(self.worker, "teardown", Qt.QueuedConnection)
            self._thread.quit()
            self._thread.wait(3000)

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
