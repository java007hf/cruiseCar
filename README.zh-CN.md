# CruiseCar

[English](README.md) | [简体中文](README.zh-CN.md)

CruiseCar 是一个基于 ESP32 小车底盘和两台 Android 手机的智能巡航小车项目。一台 Android 手机可以固定在车上作为接收端摄像头和控制桥，另一台 Android 手机作为发送端遥控器。项目还包含用于远程控制的 Python server，以及用于目标识别实验的 ML 训练目录。

## 项目能力概览

- 发送端 Android 手机提供屏幕摇杆控制。
- 接收端 Android 手机通过 Classic Bluetooth SPP 将控制包转发给 ESP32。
- 支持局域网 UDP 发现 + TCP 直连控制。
- 支持通过轻量 Python relay server 做远程控制。
- 支持可选全量服务模式：账号登录、接收端注册、设备列表和简单 Web 管理页。
- 支持 WebRTC 实时视频；server 只转发信令，媒体流尽量走 WebRTC ICE/P2P。
- 接收端支持本地智能跟随 / 目标识别实验。
- `ml/` 下提供基于视频生成 YOLO/TFLite 模型的训练工作区。

更详细的功能逻辑、协议、架构说明和维护规范见 [`agents.md`](agents.md)。

## 硬件清单

| 设备/模块 | 数量 | 用途说明 |
| --- | ---: | --- |
| ESP32 开发板 | 1 | 小车主控，接收控制包并输出电机/舵机控制信号。 |
| TB6612 双路电机驱动模块 | 1 | 驱动左右直流减速电机，支持正反转和 PWM 调速。 |
| 520 减速电机 | 2 | 左右驱动轮动力来源；带编码器版本可接入反馈输入。 |
| 双层/多孔小车底盘 | 1 | 固定 ESP32、电机驱动、电池、手机支架和接线。 |
| 车轮 | 2 | 与 520 减速电机连接。 |
| 万向轮/支撑轮 | 1 | 保持底盘平衡。 |
| 电池/电池盒 | 1 | 给电机驱动和 ESP32 供电；按模块要求接线并共地。 |
| Android 手机（接收端） | 1 | 固定在车上，用于摄像头采集、检测/跟随和转发控制指令。 |
| Android 手机（发送端） | 1 | 作为遥控端，显示摇杆/视频画面。 |
| 手机支架/固定件 | 1 | 将接收端手机固定在车体前方。 |
| 杜邦线/排线/螺丝铜柱 | 若干 | 接线和结构固定。 |
| 舵机（如 SG90） | 1（可选） | 固件预留 GPIO18 50Hz PWM，可用于摄像头俯仰或其他结构。 |
| 蓝牙手柄 | 1（可选） | ESP32 HID Host 实验可直接接入真实手柄。 |

## 项目外观

当前实物方案：接收端 Android 手机固定在小车前部作为车载摄像头，ESP32、电机驱动、电池和接线模块固定在多孔底盘上，左右两侧为 520 减速电机驱动车轮，支撑轮用于保持平衡。

![CruiseCar 项目外观](docs/images/cruisecar-appearance.jpeg)

## 项目结构

```text
.
├── android-app/                 # Android Kotlin App：发送端 / 接收端 / 调试台
│   └── app/src/main/
│       ├── assets/              # 目标识别 Demo 使用的 TFLite 模型和标签
│       └── java/com/cruisecar/app/
├── esp32-firmware/              # 主 ESP-IDF 固件：SPP + 电机/舵机/编码器控制
│   ├── main/
│   └── scripts/
├── esp32-gamepad-hid-demo/      # ESP32 蓝牙 HID 手柄实验工程
│   ├── main/
│   └── scripts/
├── server/                      # Python 远程控制服务
│   ├── control_server/          # TCP 控制转发 + WebRTC 信令转发
│   ├── manager_api/             # 账号 HTTP API + 配置/协议/存储
│   └── manager_web/             # 账号 Web 管理页，独立启动
├── ml/                          # 训练、自动标注、模型导出和运行产物目录
├── docs/images/                 # README 图片
├── README.md                    # 英文概览
├── README.zh-CN.md              # 中文概览
└── agents.md                    # 详细架构与维护指南
```

## Android App 模式

应用首页提供四个入口，覆盖两类网络模式下的发送端/接收端角色：

- **局域网发送端 / 接收端**：发送端通过 UDP 广播发现接收端，再直连接收端手机。
- **服务器发送端 / 接收端**：发送端和接收端登录同一个账号。接收端加入账号成为设备，发送端从设备列表中选择接收端。

接收端设备 ID 会根据 Android 设备信息和安装 ID 自动生成，用户不需要手动编造。Android App 内置服务器 `http://116.62.32.90/`，并会在本地保存上次填写的账号、token 和发送端 ID，避免每次重复输入。密码只用于登录请求，不会持久化保存。

## Server 模式

从 `server/` 目录启动：

```bash
cd server

# 账号部署：控制转发 + WebRTC 信令转发 + manager-api + manager-web。
python3 -m control_server.server
```

`server/` 目录也提供 Docker 配置：

```bash
cd server
docker compose up server
```

默认端口：

| 端口 | 用途 |
| ---: | --- |
| 42100 | 局域网 UDP 发现 |
| 42101 | 局域网 TCP 控制 |
| 42102 | 局域网 WebRTC 信令 |
| 42110 | 服务器 TCP 控制转发 |
| 42112 | 服务器 WebRTC 信令转发 |
| 8088 | 账号 manager-api |
| 8089 | 账号 manager-web |

## Debug Trace 延迟日志

Debug trace 用于测量控制命令从发送端到 ESP32 的各阶段延迟。正常模式和 release 包仍发送原有 10 字节控制包；打开 debug trace 后，发送端会把原始 10 字节控制包包在 `0xF0` trace 帧里，各阶段只追加时间戳，ESP32 最终仍执行内部的原始控制包。

- **手机发送端**：安装/运行 Android debug APK，例如 `cd android-app && ./gradlew :app:assembleDebug`。debug 构建会通过 `BuildConfig.DEBUG` 自动打开 trace；release 构建仍发送普通 10 字节包。
- **Web 发送端**：打开 `/send/` 或 `/web_send/`，选择接收端设备后勾选 `Debug trace`。只有勾选期间发送的命令会携带 trace 时间戳。
- **Server**：不需要单独开关。`control_server` 和 `manager-api` 会自动识别 trace 帧，追加 `server_received` / `server_forward`，普通包保持原样转发。
- **ESP32**：不需要运行时开关。刷入本仓库固件后，ESP32 收到 SPP trace 帧会执行内部控制包，并通过低优先级 FreeRTOS 日志任务异步输出一行 `trace seq=...`，避免在 SPP 回调里直接打日志影响延迟。

后续把 ESP32 这类日志贴回来即可分析：

```text
trace seq=123 type=0x01 esp_rx_ms=456789 sender_created=... sender_tcp_write=... server_received=... server_forward=... receiver_tcp_received=... receiver_bt_enqueue=... receiver_bt_write=...
```

`esp_rx_ms` 是 ESP32 本地单调时间，不能直接和手机/server 的 epoch 毫秒相减。优先看同一设备内的差值，例如 `server_forward - server_received`、`receiver_bt_write - receiver_tcp_received`、`sender_tcp_write - sender_created`。

## ESP32 接线

TB6612 电机驱动：

```text
STBY -> GPIO27

Left motor A:
  PWMA -> GPIO13
  AIN1 -> GPIO14
  AIN2 -> GPIO12

Right motor B:
  PWMB -> GPIO33
  BIN1 -> GPIO25
  BIN2 -> GPIO26
```

编码器输入：

```text
E1A -> GPIO22 / D22
E1B -> GPIO23 / D23
E2A -> GPIO21 / D21
E2B -> GPIO19 / D19
```

ESP32 GPIO 输入仅支持 3.3V。如果编码器板输出 5V，请使用电平转换，或确认模块支持 3.3V 逻辑。

## 构建

Android：

```bash
cd android-app
JAVA_HOME=/path/to/jdk17 ./gradlew :app:assembleDebug
```

Python server：

```bash
cd server
python3 -m compileall .
python3 -m control_server.server
```

ESP32：

```powershell
cd esp32-firmware
.\scripts\build-esp32.ps1
```

ML Web 训练平台：

```powershell
cd ml
.\.venv\Scripts\python.exe train_server.py
```

## 运行产物

训练输出、下载模型、上传视频、抽帧图片、本地数据库和临时文件不应提交到 Git。规范运行目录统一放在 `ml/` 下，包括 `uploads/`、`extractions/`、`datasets/`、`outputs/`、`weights/`、`_tmp/`。历史根目录 `runs/`、`weights/`、`yolo*.pt` 不属于标准目录结构。
