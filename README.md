# CruiseCar

Smart cruise car project with two subprojects:

- `android-app`: Android 6.0+ Kotlin app. It can run as a sender controller or a receiver mounted on the car.
- `esp32-firmware`: ESP-IDF firmware. ESP32 exposes a Classic Bluetooth SPP service and parses controller packets.

## Current Architecture

```text
Sender Android
  UDP broadcast discovery
  TCP control channel
Receiver Android
  Classic Bluetooth SPP
ESP32
  packet parser
  motor-control hook
```

Android 6.0 cannot act as a Bluetooth HID gamepad through the official Android API, so the first version uses Classic Bluetooth SPP between the receiver phone and ESP32. The packet model is still gamepad-shaped so ESP32 can later add a Bluetooth HID Host input path for real controllers.

## Android Modes

The sender and receiver apps now share a 10-byte TCP control frame model:

- Manual control: sender shows the on-screen gamepad and sends gamepad frames to the receiver. The receiver transparently forwards those frames to ESP32 over SPP.
- Real-time video: sender sends a mode frame, then connects to the receiver WebRTC signaling server on port `42102`. The receiver publishes camera + microphone through WebRTC, and the sender renders the remote media.
- Smart follow: sender sends a mode frame. The receiver starts camera preview + OpenCV color tracking locally and sends generated gamepad frames directly to ESP32, so the sender does not need to keep driving.
- Smart patrol: reserved mode frame and UI entry. The receiver stops motion and logs the selected mode until route planning is added.

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
