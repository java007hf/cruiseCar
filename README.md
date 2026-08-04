# CruiseCar

Smart cruise car project with two subprojects:

- `android-app`: Android 6.0+ Kotlin app. It can run as a sender controller or a receiver mounted on the car.
- `esp32-firmware`: ESP-IDF firmware. ESP32 exposes a Classic Bluetooth SPP service, parses controller packets, and drives the TB6612 motor controller with encoder feedback.

## Current Architecture

```text
Sender Android
  UDP broadcast discovery
  TCP control channel
Receiver Android
  Classic Bluetooth SPP
Bluetooth HID Gamepad
  Classic Bluetooth / BLE HID
ESP32
  SPP packet parser
  HID Host parser
  TB6612 motor control
  E1/E2 encoder feedback
```

Android 6.0 cannot act as a Bluetooth HID gamepad through the official Android API, so the first version uses Classic Bluetooth SPP between the receiver phone and ESP32. The packet model is still gamepad-shaped so ESP32 can later add a Bluetooth HID Host input path for real controllers.

ESP32 now supports both input paths at the same time: the receiver Android app can stay connected over Classic Bluetooth SPP while ESP32 also scans for and connects to a Bluetooth HID gamepad. Both paths are normalized to the same `LX/LY/RX/RY/buttons` gamepad state before reaching the motor controller. The most recent input wins; disconnecting one path only stops the car if no other control path is connected.

## Android Modes

The sender and receiver apps now share a 10-byte TCP control frame model:

- Manual control: sender shows the on-screen gamepad and sends gamepad frames to the receiver. The receiver transparently forwards those frames to ESP32 over SPP.
- Real-time video: sender sends a mode frame, then connects to the receiver WebRTC signaling server on port `42102`. The receiver publishes camera + microphone through WebRTC, and the sender renders the remote media.
- Smart follow: sender sends a mode frame. The receiver starts camera preview + OpenCV color tracking locally and sends generated gamepad frames directly to ESP32, so the sender does not need to keep driving.
- Smart patrol: reserved mode frame and UI entry. The receiver stops motion and logs the selected mode until route planning is added.
- OpenCV object demo: app camera preview feeds frames into a YOLO TFLite detector, detection results are rendered as rectangle overlays on top of the camera view. The detector consumes a `frameProvider` abstraction so the same camera frame path can later be shared with WebRTC sending.

## Directory Layout

```text
.
├── android-app/
│   ├── app/build.gradle
│   └── app/src/main/
│       ├── assets/
│       │   ├── detect.tflite        # YOLO/LiteRT model used by the Android object demo
│       │   └── labels.txt           # detection class labels
│       └── java/com/cruisecar/app/
│           ├── MainActivity.kt
│           └── ObjectRecognitionDemo.kt
├── esp32-firmware/
│   ├── main/
│   └── scripts/
├── esp32-gamepad-hid-demo/
│   ├── main/
│   └── scripts/
└── ml/
    └── beibingyang_yolo/
        ├── auto_label_orange.py     # helper used to bootstrap labels from the orange can region
        ├── train_server.py          # web-based training platform (Flask + YOLO-World + LLM)
        ├── beibingyang.yaml         # Ultralytics dataset config
        ├── dataset/
        │   ├── images/train/
        │   ├── images/val/
        │   ├── labels/train/
        │   └── labels/val/
        ├── contact/                 # contact sheets and label-check previews
        ├── frames/                  # local generated frames, ignored by Git
        ├── labels/                  # local generated labels before train/val split, ignored by Git
        └── preview/                 # per-frame label preview images
```

Training environments, intermediate extraction outputs, and run outputs are ignored by Git:

```text
ml/**/.venv*/
ml/beibingyang_yolo/frames/
ml/beibingyang_yolo/labels/
ml/beibingyang_yolo/preview/
ml/beibingyang_yolo/_uploads/
ml/beibingyang_yolo/_dataset/
ml/beibingyang_yolo/_outputs/
ml/beibingyang_yolo/_tmp/
ml/**/runs*/
ml/**/__pycache__/
runs/
weights/
yolo*.pt
```

TCP ports:

```text
42100 UDP discovery
42101 TCP control frames
42102 TCP WebRTC signaling
```

## Control Packet

Gamepad frame, 10 bytes:

```text
AA 55 01 LX LY RX RY BTN_L BTN_H SUM
```

- `AA 55`: header
- `01`: protocol version
- `LX`, `LY`, `RX`, `RY`: joystick axes, `0..255`, center `128`
- `BTN_L`, `BTN_H`: 16-bit button bitmask, little-endian
- `SUM`: low 8 bits of the sum of the first 9 bytes

Mode frame, 10 bytes:

```text
AA 55 02 MODE 00 00 00 00 00 SUM
```

- `MODE=00`: manual control
- `MODE=01`: real-time video
- `MODE=02`: smart follow
- `MODE=03`: smart patrol

Button bits are aligned with the current ESP32 HID gamepad demo logs:

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

The Android sender exposes an on-screen gamepad and sends this same packet to the receiver Android app over TCP. The receiver app transparently forwards each packet to `CruiseCar-ESP32` over Classic Bluetooth SPP.

On the receiver app, use `自动扫描并连接 ESP32` to avoid manually entering the Bluetooth address. The app first checks paired Classic Bluetooth devices, then scans for a device named `CruiseCar-ESP32`, and connects over SPP.

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

ESP32 GPIO inputs are 3.3 V only. If the encoder board outputs 5 V, use level shifting or power the encoder logic from 3.3 V if supported.

## Build

Android:

```powershell
cd android-app
.\gradlew.bat assembleDebug
```

ESP32:

```powershell
cd esp32-firmware
.\scripts\build-esp32.ps1
```

## YOLO/TFLite Object Detection Training

The current object demo uses a one-class YOLO detector for the `beibingyang_can` label. The first dataset was generated from:

```text
/Users/bytedance/Downloads/飞书20260804-120018.mp4
```

Training data flow:

1. Extract frames from the source video into `ml/beibingyang_yolo/frames/`.
2. Bootstrap labels with `ml/beibingyang_yolo/auto_label_orange.py`.
3. Check `ml/beibingyang_yolo/contact/labels_sheet.jpg` and the individual files in `ml/beibingyang_yolo/preview/`.
4. Split labeled images into `ml/beibingyang_yolo/dataset/images/train`, `images/val`, `labels/train`, and `labels/val`.
5. Train YOLO with `ml/beibingyang_yolo/beibingyang.yaml`.
6. Export the best checkpoint to LiteRT/TFLite.
7. Copy the exported model to `android-app/app/src/main/assets/detect.tflite`.

Useful commands from this training run:

```bash
# Train from the YOLO11 nano checkpoint.
ml/beibingyang_yolo/.venv311/bin/yolo detect train \
  model=yolo11n.pt \
  data=ml/beibingyang_yolo/beibingyang.yaml \
  imgsz=416 \
  epochs=60 \
  batch=4 \
  device=cpu \
  project=ml/beibingyang_yolo/runs \
  name=beibingyang_yolo11n \
  exist_ok=True

# Export the best checkpoint. Ultralytics now maps tflite export to LiteRT.
ml/beibingyang_yolo/.venv311/bin/yolo export \
  model=runs/detect/ml/beibingyang_yolo/runs/beibingyang_yolo11n/weights/best.pt \
  format=litert \
  imgsz=416

# Install the exported model into the Android app.
cp runs/detect/ml/beibingyang_yolo/runs/beibingyang_yolo11n/weights/best.tflite \
  android-app/app/src/main/assets/detect.tflite
```

The generated model currently has this tensor contract:

```text
input:  [1, 3, 416, 416] float32 RGB, NCHW, normalized to 0..1
output: [1, 5, 3549] float32 YOLO boxes/classes
```

`ObjectRecognitionDemo.kt` supports both NCHW and NHWC RGB TFLite inputs and parses YOLO output layouts `[1, boxes, attrs]` and `[1, attrs, boxes]`.

The initial training set contains 53 frames from one video, so it is good for validating the pipeline and detecting the same can in similar desk scenes. For robust detection, add more short videos with different distances, rotations, lighting, backgrounds, and partial occlusion, then repeat the same extract-label-train-export cycle.

## Web-Based Training Platform

`ml/beibingyang_yolo/train_server.py` is a one-file Flask app that wraps the entire extract-label-train pipeline into a browser UI. Instead of running CLI commands manually, you upload a video, type the object name, and click one button.

### Prerequisites

The existing `.venv` already has `ultralytics`, `opencv-python`, `torch` (CUDA), and `flask` installed. The app also downloads `yolov8s-worldv2.pt` (YOLO-World) and `yolo11n.pt` on first run.

Optional: a local LLM server (e.g. `llama.cpp` with `qwen3.5`) at `http://127.0.0.1:12345` for enhanced auto-labeling. The app works without it.

### Start the server

```powershell
cd ml/beibingyang_yolo
.\.venv\Scripts\python.exe train_server.py
```

Open `http://127.0.0.1:5000` in the browser.

### Usage

1. **Upload video** — drag and drop or click the upload area (MP4, AVI, MOV, MKV, WEBM, etc.).
2. **Enter the target object name** — use English, e.g. `beibingyang_can`, `bottle`, `red_car`.
3. **(Optional) Advanced settings** — adjust frame extraction FPS, confidence threshold, train/val split ratio, epochs, image size, batch size, and device.
4. **(Optional) Enable LLM** — check "use local LLM" to let `qwen3.5` generate richer detection prompts and verify low-confidence detections visually.
5. **Click "Start Training"** — the pipeline runs in the background with real-time progress:
   - **Extract frames** — OpenCV extracts keyframes at the configured FPS.
   - **Auto-label** — YOLO-World detects the target object using text prompts and generates YOLO format labels. If LLM is enabled, it enhances prompts and verifies uncertain detections.
   - **Split dataset** — frames are split into train/val sets and a `dataset.yaml` is generated.
   - **Train** — Ultralytics trains a YOLO11n model with the configured parameters.
6. **Download** — when training finishes, click the download button to save the `.pt` model file.

### Pipeline architecture

```text
Browser UI (HTML/CSS/JS)
  │
  ├─ POST /api/upload      → save video to _uploads/
  ├─ POST /api/process     → start background pipeline thread
  ├─ GET  /api/status      → poll progress (step, %, logs)
  ├─ GET  /api/llm_status  → check local LLM availability
  └─ GET  /api/download    → download trained .pt file

Backend (Flask, single file)
  │
  ├─ extract_frames()      → cv2.VideoCapture → JPG frames
  ├─ auto_label_frames()   → YOLO-World text-prompted detection
  │    ├─ generate_prompts()      → default heuristic prompts
  │    ├─ generate_llm_prompts()  → LLM-enhanced prompts (optional)
  │    └─ llm_analyze_image()     → visual verification (optional)
  ├─ split_dataset()       → train/val split + dataset.yaml
  └─ YOLO.train()          → ultralytics training API
```

### Output locations

```text
ml/beibingyang_yolo/
  _uploads/        → uploaded videos (gitignored)
  _dataset/        → extracted frames, labels, dataset.yaml (gitignored)
  _outputs/        → training run directories and final .pt models (gitignored)
```

The final model is saved as `_outputs/{class_name}_{timestamp}.pt`.
