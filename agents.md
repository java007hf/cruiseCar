# CruiseCar Agent Guide

本文档面向后续参与本仓库维护的 AI Agent / 开发者，说明项目结构、核心功能、开发规范、注意事项和关键组件。

## 项目定位

CruiseCar 是一个智能巡航小车项目，当前由 Android App、ESP32 固件、Python 远程控制服务和 ML 训练工具组成：

- Android 手机可作为 **发送端遥控器** 或安装在车上的 **接收端**。
- 接收端 Android 负责摄像头、目标识别/跟随、WebRTC 媒体采集，并将控制包转发给 ESP32。
- ESP32 通过 Classic Bluetooth SPP 接收控制包，驱动 TB6612 电机和预留舵机 PWM。
- Python server 支持公网/虚拟组网下的远程控制，分轻量部署和全量部署。
- `ml/` 提供基于视频 + 多模态 LLM 自动标注 + YOLO 训练 + TFLite/LiteRT 导出的训练链路。

## 目录结构

```text
.
├── android-app/                 # Android Kotlin App：发送端 / 接收端共用
│   └── app/src/main/java/com/cruisecar/app/
├── esp32-firmware/              # 主小车 ESP-IDF 固件：SPP + 电机/舵机/编码器
├── esp32-gamepad-hid-demo/      # ESP32 蓝牙 HID 手柄接入实验工程
├── server/                      # Python 远程控制服务，运行时从该目录启动
│   ├── control_server/          # TCP 控制转发 + WebRTC 信令转发，统一启动入口
│   ├── manager_api/             # Full 模式 HTTP API + 配置/协议/SQLite 存储
│   │   ├── config/              # 环境变量配置
│   │   ├── protocol/            # 10 字节控制协议解析/生成
│   │   └── storage/             # SQLite 持久化
│   └── manager_web/             # Full 模式 Web 管理页，独立于 manager_api
├── ml/                          # 训练、自动标注、模型导出、训练产物规范目录
├── docs/images/                 # 文档图片
├── README.md                    # 英文说明
└── README.zh-CN.md              # 中文说明
```

## 核心功能

### Android App

入口与主要逻辑位于 `android-app/app/src/main/java/com/cruisecar/app/MainActivity.kt`。

支持三类网络模式：

1. **局域网模式**
   - 发送端通过 UDP 发现接收端。
   - 发送端直连接收端 TCP `42101` 控制端口。
   - WebRTC 信令直连接收端 TCP `42102`。

2. **服务器 Light 模式**
   - 发送端和接收端都手动填写 server IP/域名。
   - 接收端根据 Android 设备信息自动生成稳定 `device_id` 和可读设备名。
   - 发送端填写接收端展示的 `device_id` 后连接 `server:42110`。
   - 不依赖账号、manager-api 或 manager-web。

3. **服务器 Full 模式**
   - 发送端和接收端使用同一账号登录 manager-api。
   - 接收端用自动生成的 `device_id` 和可读设备名加入账号成为设备。
   - 发送端查询设备列表并选择设备控制。
   - manager-web 只负责管理 UI，实际数据通过 manager-api。

控制模式包括：

- 手动控制
- 实时视频
- 智能跟随
- 智能巡航（预留）
- OpenCV / YOLO 目标识别 Demo

### ESP32 固件

主固件在 `esp32-firmware/main/`：

- `main.c`：SPP / HID / 主任务初始化。
- `car_control.c` / `car_control.h`：TB6612 电机控制、舵机 PWM、编码器相关逻辑。
- `esp_hid_gap.c` / `esp_hid_gap.h`：蓝牙 GAP / HID 辅助逻辑。

控制输入统一归一为手柄状态：`LX/LY/RX/RY/buttons`。

### Python Server

server 按控制层、管理 API 和管理 Web 分层拆分，不保留顶层 `app.py`，从 `server/` 目录内启动：

```bash
cd server

# Light：仅控制转发 + WebRTC 信令转发
CRUISECAR_DEPLOYMENT=light python3 -m control_server.server

# Full：Light 能力 + manager-api + manager-web
CRUISECAR_DEPLOYMENT=full python3 -m control_server.server
```

Docker 部署：`server/` 下已提供 `Dockerfile` 与 `docker-compose.yml`（light/full 两种 profile）。镜像基于 `python:3.11-slim`，纯标准库无需 `pip install`；容器内工作目录为 `/app/server`，`CRUISECAR_DB` 默认 `/data/cruisecar.db`，需挂载卷才能持久化 SQLite。

端口约定：

```text
42110  TCP 控制转发，发送端/接收端连接
42112  TCP WebRTC 信令转发，仅转发 offer/answer/ICE，不转发媒体
8088   HTTP manager-api，仅 Full 模式
8089   HTTP manager-web，仅 Full 模式
```

server 内部职责：

- `control_server/server.py`
  - 控制转发核心。
  - 统一启动入口。
  - Light 模式启动 `control_server` + `webrtc_signal`。
  - Full 模式额外启动 `manager_api` + `manager_web`。

- `control_server/webrtc_signal.py`
  - WebRTC 信令 relay。
  - 按 `room_id` 连接 caller/answerer。
  - 媒体流仍走 WebRTC ICE/P2P；严格 NAT 环境需要 TURN。

- `manager_api/server.py`
  - 账号登录/注册。
  - 设备加入、列表查询、事件查询。
  - HTTP 下发控制命令，在线则即时发送，离线则进入 SQLite 队列。

- `manager_web/server.py`
  - 独立 Web 管理页。
  - 页面默认调用同 host 的 `manager_api` 端口。

- `manager_api/config/settings.py`
  - 读取 `CRUISECAR_*` 环境变量。

- `manager_api/protocol/control_protocol.py`
  - 10 字节控制包解析和生成。

- `manager_api/storage/store.py`
  - SQLite 表结构和设备/用户/事件/命令队列操作。

### ML 训练

ML 相关内容统一放在 `ml/`：

- `train_server.py`：Web 训练平台。
- `auto_label_orange.py`：早期辅助标注脚本。
- `export_to_tflite.py`：模型导出辅助脚本。
- `uploads/`：上传原始视频。
- `extractions/`：抽帧结果。
- `datasets/`：YOLO 标签和数据集。
- `outputs/`：训练输出和最终 `.pt` 模型。
- `weights/`：预训练权重。

根目录历史 `runs/`、`weights/`、`yolo*.pt` 属于旧训练残留，不应作为规范目录使用。

## 控制协议

Android、Python server、ESP32 共享 10 字节控制包模型。

手柄帧：

```text
AA 55 01 LX LY RX RY BTN_L BTN_H SUM
```

模式帧：

```text
AA 55 02 MODE 00 00 00 00 00 SUM
```

其他帧：

- `0x03`：舵机帧。
- `0x04`：接收端状态帧，接收端周期性上报 ESP32 连接态和当前模式。
- `0x05`：命令帧，例如远程触发接收端扫描并连接 ESP32。

协议实现位置：

- Android：`ControlPacket.kt`、`TcpControl.kt`
- Python：`server/manager_api/protocol/control_protocol.py`
- ESP32：`esp32-firmware/main/`

修改协议时必须同步三端实现和 README。

## 开发规范

### 通用规范

- 保持 README.md 和 README.zh-CN.md 内容同步。
- 不要把运行产物、缓存、训练中间文件提交到 Git。
- 不要重新引入顶层 `server/app.py`、`server/common/`、`server/config/`、`server/protocol/`、`server/storage/`。
- Python server 运行时从 `server/` 目录启动，不依赖顶层 `server` 作为 Python package。
- Full 模式下 `manager_api` 和 `manager_web` 必须保持独立服务，不能把 Web HTML 再内嵌回 API 根路径。
- WebRTC server 只做信令转发，不处理媒体流。

### Android 规范

- 主要 UI 当前是 Kotlin 代码动态创建，不是 XML 布局。
- 网络模式入口、发送端/接收端流程优先在 `MainActivity.kt` 中维护。
- 控制通道相关修改优先看 `TcpControl.kt`。
- WebRTC 相关修改优先看 `WebRtcCall.kt`。
- Full 模式 HTTP API 客户端在 `RemoteApi.kt`。
- 接收端服务器模式的 `device_id` 由 `data/local/ReceiverIdentityStore.kt` 调用 `utils/DeviceIdUtils.kt` 自动生成并持久化，不要恢复成默认 `car-001` 手填流程。
- 构建 Android 时需要 JDK 17。

### ESP32 规范

- 电机控制相关逻辑集中在 `car_control.c`。
- ESP32 GPIO 是 3.3V 逻辑；编码器若输出 5V，需要电平转换或确认模块支持 3.3V。
- 固件控制包解析应保持和 Android/Python 的 10 字节协议一致。

### Python Server 规范

- 使用 Python 标准库实现，当前不依赖第三方包。
- Light 部署不能依赖 manager-api 的 HTTP 服务；设备直连 `control_server` 必须可工作。
- 允许 `control_server` 复用 `manager_api` 下的 config/protocol/storage 代码，但 Light 模式不应启动 HTTP API。
- 修改认证、设备归属、命令队列时要同时考虑 Light 和 Full 两种部署。
- SQLite 默认路径由 `CRUISECAR_DB` 控制，未配置时位于 `server/cruisecar.db`。

### ML 规范

- 新训练流程应优先使用 `ml/train_server.py`。
- 中间产物放入 `ml/extractions/`、`ml/datasets/`、`ml/outputs/`。
- 不要在项目根目录生成新的 `runs/` 或 `weights/`。
- `.pt`、`.tflite`、视频、帧图片等二进制内容较大，提交前必须确认是否确实需要纳入版本管理。

## 常用命令

Android 构建：

```bash
cd android-app
JAVA_HOME=/path/to/jdk17 ./gradlew :app:assembleDebug
```

Python server 编译检查：

```bash
cd server
python3 -m compileall .
```

Python server 启动：

```bash
cd server
CRUISECAR_DEPLOYMENT=light python3 -m control_server.server
CRUISECAR_DEPLOYMENT=full python3 -m control_server.server
```

ESP32 构建：

```powershell
cd esp32-firmware
.\scripts\build-esp32.ps1
```

ML Web 训练平台：

```powershell
cd ml
.\.venv\Scripts\python.exe train_server.py
```

## 关键注意事项

- 当前 `ml/beibingyang_yolo/` 是历史/本地训练数据目录，默认不要主动纳入提交，除非明确需要保留该数据集。
- `android-app/app/src/main/assets/target_object.jpg` 是运行时/调试资产，已被 gitignore 忽略。
- `server/*.db`、`server/*.db-wal`、`server/*.db-shm` 是本地运行数据库产物，不能提交。
- 如果修改 server 启动方式，必须同步 README 中英文和本文档。
- 如果修改端口或环境变量，必须同步 Android、server 和 README。
- 如果修改 WebRTC 信令格式，必须同步 Android `WebRtcCall.kt` 与 Python `webrtc_signal.py`。
- 如果修改控制帧格式，必须同步 Android、Python、ESP32 三端。
