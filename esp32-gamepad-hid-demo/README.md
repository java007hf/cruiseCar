# ESP32 Bluetooth Gamepad HID Demo

This is a standalone ESP-IDF demo for testing whether a Bluetooth gamepad can send HID input reports to an ESP32 and drive the small three-wheel car.

It scans for Bluetooth HID devices, opens the first discovered gamepad-like device, prints every input report to the serial log, and maps the left stick to two TB6612-driven geared motors.

## Motor Wiring

The motor pins are aligned with the reference `MotorEncoder6612_V2.0.ino` demo:

```text
TB6612 STBY: GPIO27

Left motor A:
  PWMA: GPIO13
  AIN1: GPIO14
  AIN2: GPIO12

Right motor B:
  PWMB: GPIO33
  BIN1: GPIO25
  BIN2: GPIO26

Left encoder:
  E1A: GPIO22 / D22
  E1B: GPIO23 / D23

Right encoder:
  E2A: GPIO21 / D21
  E2B: GPIO19 / D19
```

The right motor direction is inverted in firmware to match the mirrored physical mounting from the reference chassis.
The encoder inputs use ESP-IDF PCNT quadrature decoding. If a wheel's tick count direction is inverted, the current balancing logic still works because it compares absolute pulse deltas.
ESP32 GPIO inputs are 3.3 V only. If your encoder board outputs 5 V on `E1A/E1B/E2A/E2B`, use level shifting or power the encoder logic from 3.3 V if the module supports it.

## Driving

- Left stick up/down controls throttle.
- Left stick left/right controls steering.
- The firmware mixes throttle and steering into differential left/right wheel speeds.
- When the target is straight, the control loop reads `E1A/E1B` and `E2A/E2B` every 50 ms and adjusts PWM so both wheels report similar encoder pulse counts.
- If the HID controller disconnects, both motors stop.
- PWM is limited to `210/255` by default for safer first tests. Adjust `MOTOR_MAX_PWM` in `main.c` after verifying direction and wiring.

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
