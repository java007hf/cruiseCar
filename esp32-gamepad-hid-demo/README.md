# ESP32 Bluetooth Gamepad HID Demo

This is a standalone ESP-IDF demo for testing whether a Bluetooth gamepad can send HID input reports to an ESP32.

It scans for Bluetooth HID devices, opens the first discovered gamepad-like device, and prints every input report to the serial log.

## Build

```powershell
cd esp32-gamepad-hid-demo
.\scripts\build.ps1
```

Flash and monitor:

```powershell
idf.py flash monitor
```

Put the gamepad in Bluetooth pairing mode before booting or reset the ESP32 after the gamepad enters pairing mode.

For Classic Bluetooth SSP, this demo uses `ESP_BT_IO_CAP_NONE` so controllers can use Just Works pairing without a displayed confirmation code. If a legacy controller still asks for a PIN, the fallback reply is `0000`.

## Expected Log

When the controller connects and you press buttons or move sticks, logs should include lines like:

```text
I gamepad_hid_demo: INPUT usage=GAMEPAD map=0 report=1 len=8
I gamepad_hid_demo: raw:
I gamepad_hid_demo: lx=128 ly=127 rx=130 ry=126 hat=8 buttons=0x00000001
```

Different controllers use different HID report formats. The raw hex dump is the source of truth; the parsed `lx/ly/rx/ry/hat/buttons` line is a useful guess for common gamepads.
