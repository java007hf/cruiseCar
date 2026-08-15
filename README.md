# CruiseCar

[English](README.md) | [简体中文](README.zh-CN.md)

CruiseCar is a smart cruise car project built around an ESP32 chassis and two Android phones. One Android phone can be mounted on the car as the receiver camera/control bridge, while another Android phone works as the sender controller. The project also includes a Python server for remote control and an ML training workspace for object detection experiments.

## What It Can Do

- On-screen joystick control from the sender Android phone.
- Receiver Android phone forwards control packets to ESP32 over Classic Bluetooth SPP.
- LAN control through UDP discovery + direct TCP connection.
- Remote control through a lightweight Python relay server.
- Optional full server mode with account login, receiver registration, device list, and a simple manager web page.
- WebRTC real-time video mode; the server only relays signaling, while media still uses WebRTC ICE/P2P whenever possible.
- Local smart-follow / object-detection experiments on the receiver phone.
- Optional ML workflow under `ml/` for generating YOLO/TFLite models from videos.

For detailed feature logic, protocol details, architecture notes, and maintenance rules, see [`agents.md`](agents.md).

## Hardware Requirements

| Device / Module | Qty | Purpose |
| --- | ---: | --- |
| ESP32 development board | 1 | Main controller. Receives control packets and outputs motor/servo control signals. |
| TB6612 dual-channel motor driver | 1 | Drives the left and right DC geared motors with direction and PWM speed control. |
| 520 geared DC motor | 2 | Left/right drive motors. Encoder versions can be connected to feedback inputs. |
| Double-layer / perforated car chassis | 1 | Mounts ESP32, motor driver, battery, phone holder, and wiring. |
| Wheels | 2 | Connected to the 520 geared motors. |
| Caster / support wheel | 1 | Keeps the chassis balanced. |
| Battery / battery holder | 1 | Powers the motor driver and ESP32. Follow module voltage requirements and share ground where required. |
| Android phone as receiver | 1 | Mounted on the car for camera capture, detection/following, and forwarding commands to ESP32. |
| Android phone as sender | 1 | Remote controller with joystick/video UI. |
| Phone holder / mounting parts | 1 | Holds the receiver phone at the front of the car. |
| Dupont wires / ribbon cables / screws / standoffs | Several | Wiring and mechanical mounting. |
| Servo, e.g. SG90 | 1 optional | Firmware reserves GPIO18 for 50Hz PWM, usable for camera tilt or other mechanisms. |
| Bluetooth gamepad | 1 optional | ESP32 HID Host experiments can use a real gamepad directly. |

## Project Appearance

The current build uses the receiver Android phone as the onboard camera. ESP32, motor driver, battery, and wiring are fixed on a perforated chassis; two side wheels are driven by 520 geared motors; a support wheel keeps the chassis balanced.

![CruiseCar project appearance](docs/images/cruisecar-appearance.jpeg)

## Project Structure

```text
.
├── android-app/                 # Android Kotlin app: sender / receiver / debug UI
│   └── app/src/main/
│       ├── assets/              # TFLite model and labels used by object demo
│       └── java/com/cruisecar/app/
├── esp32-firmware/              # Main ESP-IDF firmware: SPP + motor/servo/encoder control
│   ├── main/
│   └── scripts/
├── esp32-gamepad-hid-demo/      # ESP32 Bluetooth HID gamepad experiment
│   ├── main/
│   └── scripts/
├── server/                      # Python remote control server
│   ├── control_server/          # TCP control relay + WebRTC signaling relay
│   ├── manager_api/             # Account HTTP API + config/protocol/storage
│   └── manager_web/             # Account web UI, served separately
├── ml/                          # Training, auto-labeling, model export, runtime artifacts
├── docs/images/                 # README images
├── README.md                    # English overview
├── README.zh-CN.md              # Chinese overview
└── agents.md                    # Detailed architecture and maintenance guide
```

## Android App Modes

The app home screen provides four entries, covering sender/receiver roles in two network modes:

- **LAN sender / LAN receiver**: sender discovers the receiver through UDP broadcast, then connects directly to the receiver phone.
- **Server sender / receiver**: sender and receiver log in with the same account. The receiver joins the account as a device, and the sender selects it from the device list.

Receiver-side device IDs are generated automatically from Android device information plus an install ID, so users do not need to manually invent one. The Android app uses the built-in server `http://116.62.32.90/` and stores the last account, token, and sender ID locally to avoid repeated input. Passwords are only used for the login request and are not persisted.

## Server Modes

Start from the `server/` directory:

```bash
cd server

# Account deployment: control relay + WebRTC signaling relay + manager-api + manager-web.
python3 -m control_server.server
```

Docker files are also provided under `server/`:

```bash
cd server
docker compose up server
```

Default ports:

| Port | Usage |
| ---: | --- |
| 42100 | LAN UDP discovery |
| 42101 | LAN TCP control |
| 42102 | LAN WebRTC signaling |
| 42110 | Server TCP control relay |
| 42112 | Server WebRTC signaling relay |
| 8088 | account manager-api |
| 8089 | account manager-web |

## Debug Trace Latency Logging

Debug trace is used to measure control latency from sender to ESP32 without changing the normal 10-byte control packet in release/normal mode. When enabled, the sender wraps the original control packet in a `0xF0` trace frame and each stage appends its timestamp. ESP32 still executes the original inner 10-byte packet.

- **Phone sender**: install/run a debug Android APK, for example `cd android-app && ./gradlew :app:assembleDebug`. Android debug builds enable trace automatically through `BuildConfig.DEBUG`; release builds keep sending plain 10-byte packets.
- **Web sender**: open `/send/` or `/web_send/`, select the receiver, then check `Debug trace`. Only commands sent while this checkbox is enabled include trace timestamps.
- **Server**: no separate switch is required. `control_server` and `manager-api` automatically recognize trace frames, append `server_received` / `server_forward`, and pass normal packets unchanged.
- **ESP32**: no runtime switch is required. Flash firmware built from this repo; when a trace frame arrives over SPP, ESP32 executes the inner control packet and asynchronously prints one `trace seq=...` log line through a low-priority FreeRTOS log task.

Send ESP32 lines like the following back for latency analysis:

```text
trace seq=123 type=0x01 esp_rx_ms=456789 sender_created=... sender_tcp_write=... server_received=... server_forward=... receiver_tcp_received=... receiver_bt_enqueue=... receiver_bt_write=...
```

`esp_rx_ms` is ESP32 local monotonic time, so it should not be directly subtracted from phone/server epoch timestamps. Use same-device differences such as `server_forward - server_received`, `receiver_bt_write - receiver_tcp_received`, and `sender_tcp_write - sender_created`.

## ESP32 Wiring

TB6612 motor driver:

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

Encoder inputs:

```text
E1A -> GPIO22 / D22
E1B -> GPIO23 / D23
E2A -> GPIO21 / D21
E2B -> GPIO19 / D19
```

ESP32 GPIO inputs are 3.3 V only. If an encoder board outputs 5 V, use level shifting or confirm that the module supports 3.3 V logic.

## Build

Android:

```bash
cd android-app
JAVA_HOME=/path/to/jdk17 ./gradlew :app:assembleDebug
```

Python server:

```bash
cd server
python3 -m compileall .
python3 -m control_server.server
```

ESP32:

```powershell
cd esp32-firmware
.\scripts\build-esp32.ps1
```

ML web training platform:

```powershell
cd ml
.\.venv\Scripts\python.exe train_server.py
```

## Runtime Artifacts

Training outputs, downloaded model checkpoints, uploaded videos, extracted frames, local databases, and temporary files should not be committed. Canonical runtime directories live under `ml/` (`uploads/`, `extractions/`, `datasets/`, `outputs/`, `weights/`, `_tmp/`). Historical root-level `runs/`, `weights/`, and `yolo*.pt` files are not part of the standard layout.
