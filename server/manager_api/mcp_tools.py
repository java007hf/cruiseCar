from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from manager_api.protocol.control_protocol import (
    CMD_CONNECT_ESP32,
    ControlMode,
    SERVO_ANGLE_MAX,
    SERVO_ANGLE_MIN,
    packet_from_command,
)

if TYPE_CHECKING:
    from control_server.server import ConnectionHub
    from manager_api.storage.store import Store

logger = logging.getLogger(__name__)

DIRECTION_MAP = {
    "forward": {"lx": 128, "ly": 255},
    "backward": {"lx": 128, "ly": 0},
    "left": {"lx": 0, "ly": 128},
    "right": {"lx": 255, "ly": 128},
    "forward_left": {"lx": 0, "ly": 255},
    "forward_right": {"lx": 255, "ly": 255},
    "backward_left": {"lx": 0, "ly": 0},
    "backward_right": {"lx": 255, "ly": 0},
    "stop": {"lx": 128, "ly": 128},
}

MODE_MAP = {
    "manual": ControlMode.MANUAL,
    "smart_follow": ControlMode.SMART_FOLLOW,
    "patrol": ControlMode.PATROL,
}

TOOL_DEFINITIONS = [
    {
        "name": "car_move",
        "description": "Control car movement direction and speed. Use 'stop' to halt.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": list(DIRECTION_MAP.keys()),
                    "description": "Movement direction",
                },
                "speed": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "default": 50,
                    "description": "Speed percentage (0-100). Ignored for 'stop'.",
                },
            },
            "required": ["direction"],
        },
    },
    {
        "name": "car_set_mode",
        "description": "Switch the car driving mode.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": list(MODE_MAP.keys()),
                    "description": "Driving mode",
                },
            },
            "required": ["mode"],
        },
    },
    {
        "name": "car_set_servo",
        "description": f"Adjust camera servo angle ({SERVO_ANGLE_MIN}-{SERVO_ANGLE_MAX} degrees).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "angle": {
                    "type": "integer",
                    "minimum": SERVO_ANGLE_MIN,
                    "maximum": SERVO_ANGLE_MAX,
                    "description": f"Servo angle in degrees ({SERVO_ANGLE_MIN}-{SERVO_ANGLE_MAX})",
                },
            },
            "required": ["angle"],
        },
    },
    {
        "name": "car_connect",
        "description": "Send Bluetooth connect command to the ESP32 car.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "car_get_status",
        "description": "Get current car status (online, ESP32 connection, mode).",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


def _speed_to_axis(speed: int) -> int:
    speed = max(0, min(100, speed))
    return int(128 + (127 * speed / 100))


def handle_tool_call(device_id: str, tool_name: str, arguments: dict[str, Any], store: "Store", hub: "ConnectionHub | None", event_loop: asyncio.AbstractEventLoop | None) -> dict[str, Any]:
    if tool_name == "car_get_status":
        return _handle_get_status(device_id, store, hub)
    if tool_name == "car_connect":
        return _handle_dispatch(device_id, {"type": "command", "code": CMD_CONNECT_ESP32}, store, hub, event_loop)
    if tool_name == "car_move":
        return _handle_move(device_id, arguments, store, hub, event_loop)
    if tool_name == "car_set_mode":
        return _handle_set_mode(device_id, arguments, store, hub, event_loop)
    if tool_name == "car_set_servo":
        return _handle_set_servo(device_id, arguments, store, hub, event_loop)
    return {"error": f"unknown tool: {tool_name}"}


def _handle_get_status(device_id: str, store: "Store", hub: "ConnectionHub | None") -> dict[str, Any]:
    receiver = store.get_receiver(device_id)
    if not receiver:
        return {"content": [{"type": "text", "text": "Car not found"}]}
    online = receiver.get("online", False)
    if hub:
        online = device_id in hub.receivers
    return {
        "content": [
            {
                "type": "text",
                "text": (
                    f"device_id={device_id}\n"
                    f"online={online}\n"
                    f"esp_connected={receiver.get('esp_connected', False)}\n"
                    f"mode={receiver.get('mode', 'unknown')}\n"
                    f"name={receiver.get('name', '')}"
                ),
            }
        ]
    }


def _handle_move(device_id: str, args: dict[str, Any], store: "Store", hub: "ConnectionHub | None", event_loop: asyncio.AbstractEventLoop | None) -> dict[str, Any]:
    direction = str(args.get("direction", "stop")).lower()
    if direction not in DIRECTION_MAP:
        return {"content": [{"type": "text", "text": f"Invalid direction: {direction}. Use: {', '.join(DIRECTION_MAP.keys())}"}]}
    speed = int(args.get("speed", 50))
    axes = DIRECTION_MAP[direction]
    if direction == "stop":
        lx, ly = 128, 128
    else:
        lx = axes["lx"]
        base_ly = axes["ly"]
        if base_ly == 128:
            ly = 128
        elif base_ly > 128:
            ly = _speed_to_axis(speed)
        else:
            ly = 255 - _speed_to_axis(speed)
    command = {"type": "gamepad", "lx": lx, "ly": ly, "rx": 128, "ry": 128, "buttons": 0}
    return _handle_dispatch(device_id, command, store, hub, event_loop)


def _handle_set_mode(device_id: str, args: dict[str, Any], store: "Store", hub: "ConnectionHub | None", event_loop: asyncio.AbstractEventLoop | None) -> dict[str, Any]:
    mode = str(args.get("mode", "")).lower()
    if mode not in MODE_MAP:
        return {"content": [{"type": "text", "text": f"Invalid mode: {mode}. Use: {', '.join(MODE_MAP.keys())}"}]}
    command = {"type": "mode", "mode": mode}
    return _handle_dispatch(device_id, command, store, hub, event_loop)


def _handle_set_servo(device_id: str, args: dict[str, Any], store: "Store", hub: "ConnectionHub | None", event_loop: asyncio.AbstractEventLoop | None) -> dict[str, Any]:
    angle = int(args.get("angle", SERVO_ANGLE_MIN))
    angle = max(SERVO_ANGLE_MIN, min(SERVO_ANGLE_MAX, angle))
    command = {"type": "servo", "index": 0, "angle": angle}
    return _handle_dispatch(device_id, command, store, hub, event_loop)


def _handle_dispatch(device_id: str, command: dict[str, Any], store: "Store", hub: "ConnectionHub | None", event_loop: asyncio.AbstractEventLoop | None) -> dict[str, Any]:
    receiver = store.get_receiver(device_id)
    if not receiver:
        return {"content": [{"type": "text", "text": f"Car {device_id} not found"}]}
    try:
        packet = packet_from_command(command)
    except ValueError as exc:
        return {"content": [{"type": "text", "text": f"Invalid command: {exc}"}]}

    delivered = False
    if hub and event_loop and event_loop.is_running():
        try:
            future = asyncio.run_coroutine_threadsafe(hub.send_to_receiver(device_id, packet), event_loop)
            delivered = bool(future.result(timeout=3))
        except Exception:
            logger.exception("MCP dispatch failed device=%s", device_id)

    if not delivered:
        store.enqueue_command(device_id, packet, " ".join(f"{b:02X}" for b in packet))

    status = "delivered" if delivered else "queued (car offline)"
    return {"content": [{"type": "text", "text": f"Command {status}"}]}
