# 六自由度平台姿态演算

面向 Stewart 六自由度并联平台的姿态解算与调试上位机。闭式逆解 + 阻尼最小二乘正解，带三维姿态预览、参数档案、批量解算、电机学参数派生、工作空间分析、CSV 轨迹播放、调试模式与开发板下发。MIT 开源。

> 项目仓库：<https://github.com/JWL956/stewart-platform-attitude-solver>

---

## 功能说明

软件侧边栏共九个页面，下面按页面顺序说明每页做什么。所有参数旁的 **?** 按钮点开可看大白话解释、在机构上的位置与计算公式。

### 01 电缸管理

逐根电缸（6 根）填写名称、编码、品牌，以及整机级的 22 项规格参数：有效行程、**最大线速度（峰值）**、额定线速度、额定/最大推力、缸径、负载质量、平台自重、丝杆导程、额定电压/功率/转矩/电流、最高转速、转矩常数、反电势常数等。

- 派生项（额定转速、额定功率、额定转矩、额定电流等）按公式自动填入，无需手算。
- 超速校核统一以「最大线速度」为准；未填时回退额定线速度；填 0 表示不校核。额定语义的派生字段仍按额定线速度计算。
- 调试模式需要的 6 项必要参数（额定推力、额定线速度、导程、额定电压、额定转矩、转矩常数）在此页补全。

### 02 平台参数

设置平台几何与限位：上下平台铰点圆半径、成对夹角、相位角、铰点 Z 偏移、三组铰点中心方位角、零位高度、虎克铰许用摆角、零位基准缸长、行程上下限。

- **各动作最大值**：按当前行程与摆角约束实时算出 X/Y/Z/Roll/Pitch/Yaw 六个方向的理论可达范围，改参数即刷新，无需试算。
- **性能诊断**：归一化雅可比 SVD 的 6 个奇异值、条件数、可操作性、各向同性度，用于判断各方向的运动能力。
- **力椭球**：开关在「05 三维姿态」页，数值面板（半轴比）显示在此页。

### 03 姿态正解

输入 6 路作动器数据求平台姿态，三种输入模式各自独立保存：绝对缸长 L1~L6、相对零位位移 ΔL1~ΔL6、电机圈数（调试）。提供「从文本填充 / 从文件导入」以及「填入运动学帧 k / k−1 / k−2」按钮。

- **求解器**：TRF 阻尼最小二乘（推荐）、LM Levenberg-Marquardt、AUTO 两者竞争取最优。
- 结果区给出位姿六量、反算长度、残差、缸速、推力、电机圈数与所需转速；超行程、超摆角、超最大线速度、超额定推力会高亮告警。
- 下方 1:1 三维预览 + 旋转矩阵/铰点坐标/摆角详细数据。

### 04 逆解验证

输入目标位姿偏移（相对零位的 X/Y/Z/Roll/Pitch/Yaw），闭式求 6 缸目标长度，并做往返精度验证。支持四元数 `[w,x,y,z]` 输入输出，便于与 ROS/Unity 交换姿态。填写采样间隔 dt 后可输出逆解缸速与电机转速（连续计算两次才会求加速度）。结果区含电机圈数、转速、残差、铰点与限位校验，可下发到开发板。

### 05 三维姿态

纯 QPainter 软件渲染的三维视口，不依赖 OpenGL，正交等比 / 透视可切换。

- **视图控制**：重置视角、重置位置、地面网格、零位参考、坐标轴、自动旋转、点位坐标、正交等比。
- **拖拽平台**：开启后左键拖动上平台改 X/Y，Shift+拖动改 Z，实时反解缸长并按行程/摆角限位钳制；「重置位置」让平台回到零位。
- **力椭球**：叠加等缸力下平台各方向的响应椭球。
- **点云开关**：把工作空间点云叠加到主视口；采样后自动聚焦点云与机构，颜色按可操作性的 5~95 百分位归一化（深蓝→青绿→亮橙红）。
- **工作空间采样**：按当前行程/摆角撒点并逐点判限位，可导出 PNG（当前视口，含机构与点云）或 CSV（完整 XYZ+Roll/Pitch/Yaw 六列）。参数变更后旧点云自动清除。

### 06 运动学计算

输入三帧绝对位姿（帧 k、k−1、k−2）与采样间隔 dt，求线速度、角速度、加速度、角加速度以及各缸线速度；结合导程给出电机圈数与转速。各缸速度与「最大线速度」实时比对，超限缸标红并在状态栏告警。

### 07 批量计算

一次粘贴或导入多行缸长（每行 6 个），后台线程批量正解，进度条显示完成情况，再点一次「批量求解」可取消。结果含各轴偏移、最大残差、铰点校验与状态，可导出 CSV。

### 08 轨迹播放

导入 CSV 轨迹：每行 6 列（pose 模式 [X,Y,Z,Roll,Pitch,Yaw] 或缸长模式 L1~L6），可选第 7 列 dwell(ms)。解析后逐帧逆解 / 下发，可播放 / 暂停 / 停止。缸速列按相邻帧位移与 dwell 估算，超最大线速度的帧高亮提示但不拦截下发。

- **点动控制**：六向 ± 小步移动，步长可调，相对当前位姿或零位，逆解后下发，用于实机对零微调。

### 09 使用说明

软件内完整帮助：参数怎么填、单位约定、配色与问号说明、动态量与导出、开源许可与版权声明，以及全部参考项目链接（与下方「参考项目」一致）。

---

## 快速开始

**方式 A — 运行已打包程序（推荐）**

直接双击 `dist/六自由度姿态解算.exe`（单文件，无需 Python）。

**方式 B — 源码运行**

```bash
pip install -r requirements.txt
python stewart_tool/app.py
```

**重新打包 exe**

```bash
cd stewart_tool
pyinstaller 六自由度姿态解算.spec --noconfirm
# spec 已配置 scipy 子模块与 Qt 模块排除项
```

## 目录结构

```
六自由度驱动/
├─ README.md / LICENSE / requirements.txt / .gitignore
├─ profiles/                        # 参数档案（JSON）
├─ dist/
│  ├─ 六自由度姿态解算.exe           # 单文件可执行程序
│  └─ profiles/                     # exe 运行目录配套档案
└─ stewart_tool/
   ├─ app.py                        # 主程序：PySide6 UI / 页面 / 信号槽
   ├─ kinematics.py                 # 运动学核心：结构 / 正逆解 / 雅可比 / SVD / 工作空间
   ├─ actuator_specs.py             # 电缸字段元数据 / 派生公式 / 圈数扭矩换算
   ├─ device_link.py                # 开发板连接状态机（可插拔协议）
   ├─ hints.py                      # 问号帮助（模型位置 / 计算公式）
   ├─ theme.py                      # 主题与玻璃质感
   ├─ viewport3d.py                 # 轻量三维视口（纯 QPainter）
   ├─ profile_manager.py            # 档案持久化
   ├─ atmos.py                      # 印象派背景层
   ├─ make_icon.py                  # 图标生成工具
   └─ 六自由度姿态解算.spec          # PyInstaller 配置
```

## 编程导向

### 分层架构

`app.py` 负责 UI 与交互；`kinematics.py` 是纯计算核心（不依赖 UI）；`actuator_specs.py` 是电缸参数的单一真相源（字段定义、公式、前置校验）；`device_link.py` 封装连接状态机与协议；其余模块提供主题、帮助、三维预览、档案与背景。计算与界面分离，便于单独验算与替换。

### 数据流

```
位姿 q[6] ──leg_lengths()──> 6 缸长度 L[6] ──forward()──> 位姿 q'[6]
                                 │
                                 └──inverse() / kinematics_dynamics()──> SolveResult
```

`SolveResult`（`kinematics.py`）关键字段：`pose`、`lengths`、`residuals`、`converged`、`rms_error`、`joint_warning`、`limit_warning`、`leg_velocity`、`leg_force`。

### 坐标与单位约定

- 世界系原点在基座中心，Z 向上；上平台本地系原点在平台中心
- 零位：`(0, 0, H0)`，旋转矩阵为单位阵
- 姿态顺序 `[X, Y, Z, Roll, Pitch, Yaw]`，长度单位 mm、角度单位 °
- 旋转矩阵 ZYX 内旋：`R = Rz(yaw)·Ry(pitch)·Rx(roll)`

### 扩展方式

1. **换机构**：修改 `kinematics.py` 的 `PlatformConfig`、`STRUCTURES`、上下铰点坐标与限位。
2. **换正解算法**：改 `forward()` 中的 `least_squares` 调用，或新增 `strategy` 分支。
3. **换驱动协议（核心可插拔点）**：只改 `device_link.py` 的 `_open_serial()` / `_heartbeat_once()` / `transmit()` 三个函数，状态机、UI 与信号机制无需改动。当前默认串口心跳 + 2 秒超时；可选 pysoem 走 EtherCAT 路线。
4. **加页面**：新增 `_build_xxx_tab()` 并挂到导航与页面栈（参照 `_build_fk_tab()`）。
5. **加电缸字段**：在 `actuator_specs.py` 的 `FIELDS` / `EXTRA_FIELDS` / `UI_ORDER_KEYS` 中登记，并在 `hints.py` 补公式说明。

### 调试模式

`settings.json` 的 `debug_mode: true` 统一控制：顶栏 ACTIVE 徽章、正/逆解调试列、逆解 dt 输入、圈数输入模式、下发按钮与右下角连接光点。开启前需补全 6 项必要电机参数。

## 公式表

| 量 | 公式 | 代码位置 |
|---|---|---|
| 旋转矩阵 | `R = Rz(yaw)·Ry(pitch)·Rx(roll)`（ZYX 内旋） | `kinematics.rot_matrix` |
| 逆解缸长 | `Lᵢ = ‖(t + R·pᵢ) − aᵢ‖` | `StewartKinematics.leg_lengths` |
| 正解 | `min_q Σ(‖t(q)+R(q)pᵢ − aᵢ‖ − Lᵢ)²`（SciPy TRF / LM / AUTO） | `StewartKinematics.forward` |
| 相对行程 | `ΔLᵢ = Lᵢ − L0ᵢ` | `app.py` 结果渲染 |
| 电机圈数 | `rev = ΔL / 导程L` | `actuator_specs.motor_revolutions` |
| 电机转速 | `n = 60·|v| / 导程L` (r/min) | `actuator_specs.motor_revolutions` |
| 额定转速 | `n_rated = 60·v_rated / 导程L` | `actuator_specs.recompute` |
| 额定转矩 | `T = F·L / (2π·η·1000)`，η=0.9 | `actuator_specs.recompute` |
| 额定电流 | `I = T / Kt` | `actuator_specs.recompute` |
| 额定功率 | `P = F·v / 1000` (W) | `actuator_specs.recompute` |
| 线反电势 | `Ke = Kt·2π·1000 / L` (V·s/m) | `actuator_specs.recompute` |
| 缸推力 | `F = pinv(Jᵀ)·wrench`，wrench 转动分量为 N·mm | `kinematics.kinematics_dynamics` |
| 归一化雅可比 | `Jₙ = J·diag(1,1,1,1/L,1/L,1/L) = U·Σ·Vᵀ` | `StewartKinematics.normalized_svd` |
| 条件数 / 可操作性 | `cond = σmax/σmin`，`manip = ∏σᵢ` | `StewartKinematics.normalized_svd` |
| 超速校核口径 | `v_cap = peak_speed → rated_speed → max_speed` | `kinematics.effective_speed_cap` |

**验算结果**（200 组随机位姿）：闭式 IK 独立复算最大差 0.0；雅可比投影幂等性 < 1e-12；IK→FK 往返最大误差约 1e-8（软件阈值 1e-4）；软件内"送回正解验证"最大位姿误差 5.31e-11。电机换算手算：ΔL=12.5、导程 5 → 2.5 圈；v=30 → 360 r/min；F=1000、导程 5 → 0.884194 N·m。

## 驱动协议说明

调试模式下，正解与逆解页均可下发当前 6 路目标缸长（逗号分隔、4 位小数）。右下角光点反映连接状态：

| 状态 | 颜色 | 触发 |
|---|---|---|
| 未连接 | 黄 | 默认 / 心跳超时 2 秒无应答 |
| 已连接 | 绿常亮 | 收到心跳应答（或模拟模式注入信号） |
| 传输正常 | 绿呼吸 | 下发后仍在线 |
| 传输失败 | 红呼吸 | 下发超时 / 无应答 |

- **真实模式**：设置中填写串口号与波特率并安装 `pyserial`，状态由心跳应答决定，光点不可手动切换。
- **模拟模式**：未配置串口时默认黄色无信号；右键光点"注入模拟信号"可验证绿色与传输态。绿色只由真实心跳或注入信号驱动，不会因点击而变绿。
- **EtherCAT 路线**：需要时可用 `pysoem`（MIT）作为主站，设备规范采用 CiA 402 over CoE；替换点集中在 `device_link.py`。

---

## 参考项目

本项目独立实现算法思路，参考下列开源项目（与软件「使用说明 → 参考项目与算法来源」完全一致）。未复制其源码，版权归各自作者所有。

### ① 同类软件 / 上位机

| 项目 | 参考内容 | 链接 |
|---|---|---|
| nairck/Hexapod IK & Workspace Calculator | MIT，Win/Linux 免安装包；IK + 可达/定向工作空间绘图 + PNG 导出 + 增量调节表（不含正解） | https://github.com/nairck/Hexapod-Inverse-Kinematics-and-Workspace-Calculator |
| stewart_platform_control | CustomTkinter + Matplotlib GUI；手动控制 / 实时 IK / CSV 轨迹 / 限位保护 / 3D 可视化 | https://github.com/stewart-platform-control/stewart_platform_control |

### ② 算法库（可移植 / 嵌入式）

| 项目 | 参考内容 | 链接 |
|---|---|---|
| Yeok-c/Stewart_Py | Python，`calculate()` 直接返回 6 舵机角，带逐步推导 Notebook | https://github.com/Yeok-c/Stewart_Py |
| rawify/Stewart.js | JS/MIT，IK 库，支持舵机/缸长、四元数、p5.js 可视化 | https://github.com/rawify/Stewart.js |
| conroy-cheers/hexapod_kinematics | 现代 C++（Eigen3），正解 + 逆解，源自 LinuxCNC genhexkins | https://github.com/conroy-cheers/hexapod_kinematics |
| duccuongvu/StewartPlatformForwardKinematics | 纯 C99 正解，LM / Trust Region 双求解器，可跑 STM32 | https://github.com/duccuongvu/StewartPlatformForwardKinematics |
| Sherin-SEF-AI/robot-kinematics | MIT，通用机器人运动学库（含 Stewart、DH/URDF、奇异检测、workspace） | https://github.com/Sherin-SEF-AI/robot-kinematics |

### ③ 分析 / 性能指标

| 项目 | 参考内容 | 链接 |
|---|---|---|
| AsioOtus055/StewartPlatformClass_py | 工作空间、雅可比、奇异值、条件数、驱动力矩 | https://github.com/AsioOtus055/StewartPlatformClass_py |
| Flamisell/StewartPlatformClass_py | SVD 可操作性、局部条件指标、力椭球、工作空间分析 | https://github.com/Flamisell/StewartPlatformClass_py |
| ravichand13/Stewart-platform | 逆解几何 | https://github.com/ravichand13/Stewart-platform |
| matthewmhong/stewart-platform | 运动学框架 | https://github.com/matthewmhong/stewart-platform |
| martyushev/fkSGP | 正解数值方法 | https://github.com/martyushev/fkSGP |
| coelacant1/StewartPlatformIK | C++17 Windows CLI，6 参数 IK | https://github.com/coelacant1/StewartPlatformIK |

### ④ 仿真 / 实机控制

| 项目 | 参考内容 | 链接 |
|---|---|---|
| wei-hsuan-cheng/stewart_platform_sim | ROS 2 + RViz2，IK + 微分运动学（缸速/雅可比） | https://github.com/wei-hsuan-cheng/stewart_platform_sim |
| s-ritwik/Parallel-Manipulator-Stewart-Platform- | ROS 2 Humble + Gazebo，IK 节点 + 正弦测试运动 | https://github.com/s-ritwik/Parallel-Manipulator-Stewart-Platform- |
| garethnisbet/Robot | Web 3D 可视化，Newton-Raphson 正解 + 拖拽平台 + 限位钳制 | https://github.com/garethnisbet/Robot |
| JD-Patten/stewart_platform | ROS2 + Arduino 串口控制实物 | https://github.com/JD-Patten/stewart_platform |
| NicHub/stewart-platform-esp32 | ESP32 硬件落地 | https://github.com/NicHub/stewart-platform-esp32 |
| daniel-s-ingram/stewart | Gazebo + ROS + PS4 手柄（经典参考，2019 后未更新） | https://github.com/daniel-s-ingram/stewart |

### 运行与打包依赖

| 组件 | 用途 | 许可证 | 链接 |
|---|---|---|---|
| NumPy | 矩阵运算与线性代数 | BSD-3-Clause | https://numpy.org/ |
| SciPy | 最小二乘优化与旋转变换 | BSD-3-Clause | https://scipy.org/ |
| PySide6-Essentials / Shiboken6 | 图形界面（The Qt Company） | LGPL-3.0 或 GPL-2.0/3.0 | https://doc.qt.io/qtforpython/ |
| PyInstaller | 单文件 exe 打包（仅构建期使用） | GPL-2.0 + 例外条款 | https://pyinstaller.org/ |

### 可选驱动与标准（当前版本未启用）

| 名称 | 用途 | 许可证 | 链接 |
|---|---|---|---|
| pysoem | EtherCAT 主站 Python 绑定 | MIT（v1.0.2 起） | https://github.com/bnjmnp/pysoem |
| Npcap | Windows 网卡驱动，pysoem 前置依赖 | Npcap License（BSD 衍生） | https://npcap.com/ |
| SOEM | pysoem 底层 C 主站栈 | GPLv3 + 商业双许可 | https://github.com/OpenEtherCATsociety/SOEM |
| CiA 402 / EtherCAT / CANopen | 伺服驱动器对象字典与状态机标准 | 公开技术标准 | https://www.can-cia.org/can-knowledge/canopen/cia402/ · https://www.ethercat.org/ |

个人调试不触发 GPL 的分发义务；如需商业闭源分发，请自行核实许可并购买相应商业授权。

## 许可证

MIT License，详见 `LICENSE`。第三方组件的版权与许可声明见软件「使用说明 → 开源许可与版权声明」。

> 使用过程中如有问题，可联系邮箱：**jwl2425591445@163.com**
