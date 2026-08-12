# CruiseCar

[English](README.md) | [简体中文](README.zh-CN.md)

智能巡航小车项目，运行时主要包含三个子项目：

- `android-app`：Android 6.0+ Kotlin 应用，可作为发送端遥控器，也可作为安装在车上的接收端。
- `esp32-firmware`：ESP-IDF 固件。ESP32 暴露 Classic Bluetooth SPP 服务，解析控制包，并通过 TB6612 电机驱动控制小车电机，同时支持编码器反馈。
- `server`：Python 远程控制网关。提供轻量级 TCP 控制转发和 WebRTC 信令转发，也支持可选的全量 `manager_api` + 独立 `manager_web`，用于账号化设备管理。

## 硬件清单

| 设备/模块 | 数量 | 用途说明 |
| --- | ---: | --- |
| ESP32 开发板 | 1 | 小车主控，负责接收 Android 接收端转发的控制包，并输出电机/舵机控制信号。 |
| TB6612 双路电机驱动模块 | 1 | 驱动左右两路直流减速电机，支持正反转和 PWM 调速。 |
| 520 减速电机 | 2 | 左右驱动轮动力来源；如使用带编码器版本，可接入 E1/E2 编码器反馈。 |
| 双层/多孔小车底盘 | 1 | 固定 ESP32、电机驱动、电池、手机支架等硬件。 |
| 车轮 | 2 | 与 520 减速电机连接，提供小车行走能力。 |
| 万向轮/支撑轮 | 1 | 用于保持底盘平衡。 |
| 电池/电池盒 | 1 | 给电机驱动、ESP32 等硬件供电；电机电源建议与逻辑电源按模块要求正确接线并共地。 |
| Android 手机（接收端） | 1 | 安装在车体上，负责视频采集、目标检测/跟随，并通过蓝牙或 TCP 链路把控制指令转发给 ESP32。 |
| Android 手机（发送端） | 1 | 作为遥控端，显示摇杆/视频画面并发送控制指令。 |
| 手机支架/固定件 | 1 | 将接收端手机固定在车体前部，作为车载摄像头使用。 |
| 杜邦线/排线/螺丝铜柱 | 若干 | 用于模块接线与结构固定。 |
| 舵机（如 SG90/兼容 180° 舵机） | 1（可选） | 当前固件预留 GPIO18 输出 50Hz PWM，可用于摄像头俯仰或其他机械结构控制。 |
| 蓝牙手柄 | 1（可选） | ESP32 HID Host 路径可直接接入手柄控制，Android SPP 路径也可同时保留。 |

## 项目外观

当前项目外观如下：接收端 Android 手机固定在小车前部作为车载摄像头，ESP32、电机驱动、电池和接线模块固定在多孔底盘上，左右两侧为 520 减速电机驱动车轮，后部/底部使用支撑轮保持平衡。

![CruiseCar 项目外观](docs/images/cruisecar-appearance.jpeg)

## 当前架构

```text
发送端 Android
  局域网模式：UDP 广播发现 + 直连 TCP 控制 + 直连 WebRTC 信令
  服务器模式：经 Python control_server 转发 TCP 控制 + WebRTC 信令中继
接收端 Android
  局域网模式：本机 TCP 控制服务 + 本机 WebRTC 信令服务
  服务器模式：主动连接 Python control_server + WebRTC 信令中继
  Classic Bluetooth SPP
蓝牙 HID 手柄
  Classic Bluetooth / BLE HID
ESP32
  SPP 控制包解析
  HID Host 解析
  TB6612 电机控制
  E1/E2 编码器反馈
```

Android 6.0 无法通过官方 API 直接模拟蓝牙 HID 手柄，因此第一版使用接收端手机与 ESP32 之间的 Classic Bluetooth SPP 通道。控制包仍然按“手柄状态”建模，便于后续 ESP32 直接接入真实蓝牙 HID 手柄。

ESP32 当前可同时支持两种输入路径：接收端 Android 应用通过 Classic Bluetooth SPP 保持连接，同时 ESP32 也可以扫描并连接蓝牙 HID 手柄。两条路径最终都会归一化为相同的 `LX/LY/RX/RY/buttons` 手柄状态再进入电机控制逻辑。最近输入优先；断开其中一路时，只要还有另一条控制路径在线，小车不会误停。

## Android 模式

发送端和接收端应用共用 10 字节 TCP 控制帧模型：

- 手动控制：发送端显示屏幕摇杆，并把手柄帧发送给接收端；接收端再透明转发到 ESP32 的 SPP 链路。
- 实时视频：发送端发送模式帧后，连接接收端 `42102` 端口上的 WebRTC 信令服务；接收端发布摄像头和麦克风，发送端渲染远端媒体。
- 智能跟随：发送端发送模式帧；接收端本地启动相机预览和 OpenCV 目标跟踪，并直接向 ESP32 发送生成的手柄帧，因此发送端无需持续手动控制。
- 智能巡航：预留模式帧和 UI 入口；在路线规划完成前，接收端会停止运动并记录当前模式。
- OpenCV 目标识别 Demo：相机预览帧输入 YOLO TFLite 检测器，检测结果以矩形框覆盖在预览画面上；检测器使用 `frameProvider` 抽象，后续可与 WebRTC 发送链路复用同一相机帧路径。

应用首页现在提供三类网络模式：

- **局域网发送端 / 接收端**：保留原有局域网流程。发送端通过 UDP 广播发现接收端，再直连接收端手机的 TCP `42101` 控制端口；WebRTC 信令也直连接收端手机的 TCP `42102` 端口。
- **服务器 Light 发送端 / 接收端**：两台手机分别手动输入同一个服务器 IP/域名和接收端 `device_id`。接收端与发送端都会主动连接 `server:42110`；不需要账号、manager-api、数据库 UI 或 manager-web 登录。适合 Tailscale/ZeroTier/ngrok 或最小化公网 VPS。
- **服务器 Full 发送端 / 接收端**：两台手机使用同一个账号登录 manager-api。接收端加入该账号成为受管理设备；发送端查询设备列表、选择接收端后通过 `control_server` 控制它。该模式会单独启动 `manager_web` 服务，和 xiaozhi 的 `manager-api` / `manager-web` 拆分方式保持一致。

服务器模式下，WebRTC **媒体流**仍尽量走两台 Android 设备之间的 ICE/P2P 链路；Python server 只在 TCP `42112` 上转发信令（`offer`、`answer` 和 ICE candidates）。如果遇到严格 NAT 或防火墙环境，建议后续给 WebRTC 增加 TURN 服务器，让媒体流可回退到 TURN 中继。

## Python Server

Python server 支持类似 `xiaozhi-esp32-server` 的轻量部署和全量部署：

```bash
cd server

# 轻量部署：仅启动控制转发 + WebRTC 信令转发。
CRUISECAR_DEPLOYMENT=light python3 -m control_server.server

# 全量部署：轻量能力 + manager-api + manager-web。
CRUISECAR_DEPLOYMENT=full python3 -m control_server.server
```

环境变量：

```text
CRUISECAR_DEPLOYMENT=light|full   # 默认 light
CRUISECAR_HOST=0.0.0.0
CRUISECAR_CONTROL_PORT=42110      # 发送端/接收端 TCP 控制转发
CRUISECAR_WEBRTC_PORT=42112       # WebRTC 信令转发
CRUISECAR_MANAGER_PORT=8088       # full 模式下 manager-api
CRUISECAR_MANAGER_WEB_PORT=8089   # full 模式下 manager-web
CRUISECAR_DB=server/cruisecar.db  # SQLite 数据库路径
CRUISECAR_AUTH_TOKEN=             # 可选全局 token，适合轻量部署
```

Full 模式 manager API：

```text
GET  /                         manager-api 服务信息
POST /api/auth/login           登录或自动注册账号
POST /api/receivers            在当前账号下加入/更新接收端
GET  /api/receivers            查询当前账号可见接收端
POST /api/receivers/{id}/commands
GET  /api/senders
GET  /api/events?limit=100
```

浏览器打开 `http://server-ip:8089/` 可使用 `manager_web` 进行登录、加入接收端、查看设备列表和事件；页面会调用 `http://server-ip:8088` 上的 `manager_api`。

## 目录结构

```text
.
├── android-app/
│   ├── app/build.gradle
│   └── app/src/main/
│       ├── assets/
│       │   ├── detect.tflite        # Android 目标识别 Demo 使用的 YOLO/LiteRT 模型
│       │   └── labels.txt           # 检测类别标签
│       └── java/com/cruisecar/app/
│           ├── MainActivity.kt
│           └── ObjectRecognitionDemo.kt
├── esp32-firmware/
│   ├── main/
│   └── scripts/
├── esp32-gamepad-hid-demo/
│   ├── main/
│   └── scripts/
├── server/
│   ├── control_server/         # TCP 控制转发 + WebRTC 信令转发；服务入口
│   ├── manager_api/            # full 模式 HTTP API + config/protocol/storage
│   │   ├── config/             # 服务配置
│   │   ├── protocol/           # 10 字节控制包协议工具
│   │   └── storage/            # SQLite 存储
│   └── manager_web/            # full 模式 Web UI，和 manager_api 独立启动
└── ml/
    ├── train_server.py          # Web 训练平台（Flask + 多模态 LLM + YOLO）
    ├── auto_label_orange.py     # 用于从橙色罐区域快速生成初始标签的辅助脚本
    ├── beibingyang.yaml         # Ultralytics 数据集配置模板
    #
    # 规范子目录（通过 .gitkeep 保留目录，运行产物被 gitignore 忽略）：
    ├── weights/                 # 基础预训练权重（yolo11n.pt 等）
    ├── uploads/                 # 上传的原始视频
    ├── extractions/             # 每次运行抽取的 JPG 帧
    ├── datasets/                # 每次运行生成的数据集、标签和 yaml
    ├── outputs/                 # 训练输出和最终 .pt 模型
    └── _tmp/                    # 临时文件目录
```

训练环境和运行产物默认不提交到 Git；规范目录通过 `.gitkeep` 占位，因此新 clone 仓库后仍保留目录骨架。

TCP 端口：

```text
局域网模式：
  42100 UDP 发现
  42101 TCP 控制帧
  42102 TCP WebRTC 信令

服务器模式：
  42110 TCP 控制转发
  42112 TCP WebRTC 信令转发
  8088  HTTP manager-api（仅 full 模式）
  8089  HTTP manager-web（仅 full 模式）
```

## 控制包

手柄帧，10 字节：

```text
AA 55 01 LX LY RX RY BTN_L BTN_H SUM
```

- `AA 55`：帧头
- `01`：协议版本
- `LX`、`LY`、`RX`、`RY`：摇杆轴，范围 `0..255`，中心值 `128`
- `BTN_L`、`BTN_H`：16 位按钮位图，小端序
- `SUM`：前 9 个字节求和后的低 8 位

模式帧，10 字节：

```text
AA 55 02 MODE 00 00 00 00 00 SUM
```

- `MODE=00`：手动控制
- `MODE=01`：实时视频
- `MODE=02`：智能跟随
- `MODE=03`：智能巡航

按钮位与当前 ESP32 HID 手柄 Demo 日志保持一致：

```text
bit 0: A
bit 1: B
bit 3: X
bit 4: Y
bit 6: L1
bit 7: R1
bit 8: L2
bit 9: R2
```

Android 发送端显示屏幕手柄，并通过 TCP 将同样的控制包发送给接收端 Android 应用。接收端应用再通过 Classic Bluetooth SPP 透明转发给 `CruiseCar-ESP32`。

在接收端应用中可以使用 `自动扫描并连接 ESP32`，避免手动输入蓝牙地址。应用会先检查已配对的 Classic Bluetooth 设备，再扫描名为 `CruiseCar-ESP32` 的设备并通过 SPP 连接。

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

ESP32 GPIO 输入仅支持 3.3V。如果编码器板输出 5V，请使用电平转换，或在编码器模块支持的前提下使用 3.3V 逻辑供电。

## 构建

Android：

```powershell
cd android-app
.\gradlew.bat assembleDebug
```

macOS/Linux 下需要确保 Android Gradle Plugin 使用 Java 17，例如：

```bash
JAVA_HOME=/path/to/jdk17 ./gradlew :app:assembleDebug
```

Python server：

```bash
cd server
python3 -m control_server.server
```

ESP32：

```powershell
cd esp32-firmware
.\scripts\build-esp32.ps1
```

## YOLO/TFLite 目标识别训练

当前目标识别 Demo 使用单类别 YOLO 检测器，默认标签为 `beibingyang_can`。训练数据流程：

1. 从源视频抽帧到 `ml/extractions/{run_id}/frame_XXXX.jpg`。
2. 将每一帧和目标文字描述（支持任意语言）发送给本地多模态 LLM：`http://127.0.0.1:12345`，LLM 直接返回归一化目标框。
3. 将标签以 YOLO 格式写入 `ml/datasets/{run_id}/labels/`，并按 90% / 10% 拆分训练集和验证集。
4. 使用生成的 `ml/datasets/{run_id}/dataset.yaml` 训练 YOLO；Ultralytics 日志位于 `ml/outputs/{run_id}/`。
5. 将最佳 checkpoint 复制为 `ml/outputs/{class_name}_{run_id}.pt`，导出为 LiteRT/TFLite，并复制到 `android-app/app/src/main/assets/detect.tflite`。

常用命令（在 `ml/` 目录执行）：

```bash
# 从 YOLO11 nano checkpoint 训练（Web UI 不需要手动执行）。
.\.venv\Scripts\yolo detect train `
  model=weights/yolo11n.pt `
  data=beibingyang.yaml `
  imgsz=416 `
  epochs=60 `
  batch=4 `
  device=cpu `
  project=outputs `
  name=beibingyang_yolo11n `
  exist_ok=True

# 导出最佳 checkpoint。Ultralytics 会把 tflite export 映射到 LiteRT。
.\.venv\Scripts\yolo export `
  model=outputs/beibingyang_yolo11n/weights/best.pt `
  format=litert `
  imgsz=416

# 安装导出的模型到 Android app。
cp outputs/beibingyang_yolo11n/weights/best.tflite `
  ..\android-app\app\src\main\assets\detect.tflite
```

当前生成模型的张量约定：

```text
input:  [1, 3, 416, 416] float32 RGB, NCHW, 归一化到 0..1
output: [1, 5, 3549] float32 YOLO boxes/classes
```

`ObjectRecognitionDemo.kt` 同时支持 NCHW 和 NHWC RGB TFLite 输入，并能解析 `[1, boxes, attrs]` 与 `[1, attrs, boxes]` 两类 YOLO 输出布局。

初始训练集只包含来自一个视频的 53 帧，适合验证链路和检测相似桌面场景中的同类罐子。若要提高鲁棒性，请补充不同距离、角度、光照、背景和遮挡条件下的短视频，再重复抽帧、标注、训练和导出流程。

## Web 训练平台

`ml/train_server.py` 是一个单文件 Flask 应用，把“抽帧 → 标注 → 训练 → 导出”流程封装为浏览器 UI。你只需要上传一个或多个视频，描述想识别/跟随的目标（支持中文，例如“红色盖子的罐子”），然后点击按钮开始训练。

### 前置条件

`ml/` 下的 `.venv` 需要安装 `ultralytics`、`opencv-python`、`torch` 和 `flask`。应用默认使用 `yolo11n.pt` 作为基础训练权重（缺失时可自动下载）。

**必须**：本地需要运行多模态 LLM 服务，例如带视觉模型的 `llama.cpp` server，地址为 `http://127.0.0.1:12345`，并提供兼容 OpenAI 的 `/v1/chat/completions` 图片输入 API。标注阶段会把每帧图片和文字描述发送给 LLM，LLM 直接返回归一化目标框。

### 启动服务

```powershell
cd ml
.\.venv\Scripts\python.exe train_server.py
```

浏览器打开 `http://127.0.0.1:5000`。

### 使用方式

1. **上传视频**：拖拽或点击上传区域，支持 MP4、AVI、MOV、MKV、WEBM 等格式；当前支持一次上传多个视频。
2. **输入目标描述**：任意语言均可，例如 `beibingyang_can`、`bottle with red cap`、`橙色带拉环的饮料罐`、`红色盖子的罐子`。视觉细节越明确，LLM 框选越稳定。
3. **高级设置（可选）**：调整抽帧 FPS、训练/验证比例、epochs、图像尺寸、batch size 和 device。
4. **点击“开始训练”**：后台执行完整流程并实时展示进度：抽帧、LLM 自动标注、人工补标/复核、数据集拆分、YOLO 训练、导出 TFLite。
5. **下载模型**：训练完成后点击下载按钮保存 `.pt` 模型文件。

### 流水线架构

```text
Browser UI (HTML/CSS/JS)
  │
  ├─ POST /api/upload      → 保存上传视频到 uploads/
  ├─ POST /api/process     → 启动后台训练线程
  ├─ GET  /api/status      → 轮询进度（阶段、百分比、日志）
  ├─ GET  /api/llm_status  → 检查本地 LLM 可用性
  └─ GET  /api/download    → 下载训练好的 .pt 文件

Backend (Flask, single file)
  │
  ├─ extract_frames()      → cv2.VideoCapture → JPG 帧
  ├─ auto_label_frames()   → 逐帧调用多模态 LLM
  │    └─ llm_detect_boxes()  → 输入：图片 + 文字描述；输出：[0,1] 归一化坐标框
  ├─ split_dataset()       → 训练/验证拆分 + dataset.yaml
  └─ YOLO.train()          → Ultralytics 训练 API
```

### 输出位置

每次运行 `{run_id}` 是时间戳 `YYYYmmdd_HHMMSS`，产物分布在以下目录：

```text
ml/
  uploads/{run_id}_NN.{mp4,webm,...}     → 上传视频
  extractions/{run_id}/                  → 抽取的原始帧 + 训练/验证拆分
  datasets/{run_id}/                     → labels/、dataset/{images,labels}/{train,val}/、dataset.yaml
  outputs/
    {run_id}/                            → Ultralytics 训练日志、events、tensorboard
    {run_id}/weights/best.pt             → 训练过程中的最佳 checkpoint
    {class_name}_{run_id}.pt             → 最终可下载模型
```

最终可下载模型保存为 `outputs/{class_name}_{run_id}.pt`。
