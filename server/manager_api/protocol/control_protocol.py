from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


PACKET_SIZE = 10
HEADER_0 = 0xAA
HEADER_1 = 0x55


class FrameType(IntEnum):
    GAMEPAD = 0x01
    MODE = 0x02
    SERVO = 0x03
    STATUS = 0x04
    COMMAND = 0x05


class ControlMode(IntEnum):
    MANUAL = 0x00
    VIDEO_CALL = 0x01
    SMART_FOLLOW = 0x02
    PATROL = 0x03
    VIDEO_OFF = 0x04


CMD_CONNECT_ESP32 = 0x01


@dataclass(frozen=True)
class ParsedFrame:
    frame_type: FrameType
    payload: dict


def checksum(packet: bytes | bytearray) -> int:
    return sum(packet[: PACKET_SIZE - 1]) & 0xFF


def is_valid_packet(packet: bytes | bytearray) -> bool:
    return (
        len(packet) == PACKET_SIZE
        and packet[0] == HEADER_0
        and packet[1] == HEADER_1
        and checksum(packet) == packet[9]
        and packet[2] in {item.value for item in FrameType}
    )


def parse_packet(packet: bytes | bytearray) -> ParsedFrame | None:
    if not is_valid_packet(packet):
        return None

    frame_type = FrameType(packet[2])
    if frame_type == FrameType.GAMEPAD:
        payload = {
            "lx": packet[3],
            "ly": packet[4],
            "rx": packet[5],
            "ry": packet[6],
            "buttons": packet[7] | (packet[8] << 8),
        }
    elif frame_type == FrameType.MODE:
        payload = {"mode": safe_mode(packet[3]).name.lower()}
    elif frame_type == FrameType.SERVO:
        payload = {"index": packet[3], "angle": min(packet[4], 180)}
    elif frame_type == FrameType.STATUS:
        payload = {"esp_connected": packet[3] != 0, "mode": safe_mode(packet[4]).name.lower()}
    else:
        payload = {"code": packet[3]}

    return ParsedFrame(frame_type=frame_type, payload=payload)


def packet_to_hex(packet: bytes | bytearray) -> str:
    return " ".join(f"{item:02X}" for item in packet)


def packet_from_hex(value: str) -> bytes:
    compact = value.replace(" ", "").replace("0x", "").replace(",", "")
    if len(compact) != PACKET_SIZE * 2:
        raise ValueError(f"hex packet must contain {PACKET_SIZE} bytes")
    packet = bytes.fromhex(compact)
    if not is_valid_packet(packet):
        raise ValueError("invalid control packet")
    return packet


def gamepad_packet(lx: int = 128, ly: int = 128, rx: int = 128, ry: int = 128, buttons: int = 0) -> bytes:
    packet = bytearray(PACKET_SIZE)
    packet[0] = HEADER_0
    packet[1] = HEADER_1
    packet[2] = FrameType.GAMEPAD
    packet[3] = clamp_byte(lx)
    packet[4] = clamp_byte(ly)
    packet[5] = clamp_byte(rx)
    packet[6] = clamp_byte(ry)
    packet[7] = buttons & 0xFF
    packet[8] = (buttons >> 8) & 0xFF
    packet[9] = checksum(packet)
    return bytes(packet)


def mode_packet(mode: str | int | ControlMode) -> bytes:
    packet = base_packet(FrameType.MODE)
    packet[3] = mode_value(mode)
    packet[9] = checksum(packet)
    return bytes(packet)


# 舵机物理行程安全下限: 与 Android 发送端 ServoVerticalView 的 130°~180° 对齐,
# 低于 130° 会损坏舵机。这里作为最后一道防线, 无论命令来自 Web 还是 HTTP API 都限制下限。
SERVO_ANGLE_MIN = 130
SERVO_ANGLE_MAX = 180


def servo_packet(index: int = 0, angle: int = 90) -> bytes:
    packet = base_packet(FrameType.SERVO)
    packet[3] = clamp_byte(index)
    packet[4] = max(SERVO_ANGLE_MIN, min(int(angle), SERVO_ANGLE_MAX))
    packet[9] = checksum(packet)
    return bytes(packet)


def status_packet(esp_connected: bool = False, mode: str | int | ControlMode = ControlMode.MANUAL) -> bytes:
    packet = base_packet(FrameType.STATUS)
    packet[3] = 1 if esp_connected else 0
    packet[4] = mode_value(mode)
    packet[9] = checksum(packet)
    return bytes(packet)


def command_packet(code: int = CMD_CONNECT_ESP32) -> bytes:
    packet = base_packet(FrameType.COMMAND)
    packet[3] = clamp_byte(code)
    packet[9] = checksum(packet)
    return bytes(packet)


def packet_from_command(command: dict) -> bytes:
    if "hex" in command:
        return packet_from_hex(str(command["hex"]))

    kind = str(command.get("type", "")).lower()
    if kind == "gamepad":
        return gamepad_packet(
            lx=int(command.get("lx", 128)),
            ly=int(command.get("ly", 128)),
            rx=int(command.get("rx", 128)),
            ry=int(command.get("ry", 128)),
            buttons=int(command.get("buttons", 0)),
        )
    if kind == "mode":
        return mode_packet(command.get("mode", ControlMode.MANUAL))
    if kind == "servo":
        return servo_packet(index=int(command.get("index", 0)), angle=int(command.get("angle", 90)))
    if kind in {"cmd", "command"}:
        return command_packet(code=int(command.get("code", CMD_CONNECT_ESP32)))

    raise ValueError("command type must be one of: gamepad, mode, servo, command, or provide hex")


def base_packet(frame_type: FrameType) -> bytearray:
    packet = bytearray(PACKET_SIZE)
    packet[0] = HEADER_0
    packet[1] = HEADER_1
    packet[2] = frame_type.value
    return packet


def clamp_byte(value: int) -> int:
    return max(0, min(int(value), 255))


def mode_value(mode: str | int | ControlMode) -> int:
    if isinstance(mode, ControlMode):
        return mode.value
    if isinstance(mode, int):
        return safe_mode(mode).value
    normalized = str(mode).strip().upper().replace("-", "_")
    try:
        return ControlMode[normalized].value
    except KeyError as exc:
        raise ValueError(f"unknown control mode: {mode}") from exc


def safe_mode(value: int) -> ControlMode:
    try:
        return ControlMode(int(value))
    except ValueError:
        return ControlMode.MANUAL
