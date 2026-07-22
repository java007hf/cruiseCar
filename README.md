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

## Control Packet

Binary packet, 10 bytes:

```text
AA 55 01 LX LY RX RY BTN_L BTN_H SUM
```

- `AA 55`: header
- `01`: protocol version
- `LX`, `LY`, `RX`, `RY`: joystick axes, `0..255`, center `128`
- `BTN_L`, `BTN_H`: 16-bit button bitmask, little-endian
- `SUM`: low 8 bits of the sum of the first 9 bytes

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

On this machine, Android debug build has been verified. ESP-IDF source has been
created for IDF v6.0.1, but the local ESP-IDF/CMake environment currently drops
the `xtensa-esp32-elf-*` compiler lookup during CMake configure even after the
compiler is visible from the shell. Re-run the script above after fixing the IDF
environment.
