from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import secrets
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from manager_api.config.settings import ServerConfig, load_config

logger = logging.getLogger(__name__)


@dataclass
class XiaozhiSession:
    device_id: str
    session_id: str = ""
    audio_params: dict[str, Any] = field(default_factory=dict)
    connected: bool = False
    connected_at: float = 0.0
    last_activity: float = 0.0


class XiaozhiBridge:
    """Manages per-device WebSocket connections to xiaozhi server.

    Each cruiseCar device gets its own WS connection to xiaozhi,
    identified by device_id. The bridge handles:
    - WebSocket connect/disconnect lifecycle
    - Hello handshake
    - Forwarding text messages (listen/detect)
    - Receiving LLM replies and audio frames
    - Heartbeat (ping/pong)
    """

    def __init__(
        self,
        config: ServerConfig | None = None,
        on_llm_reply: Callable[[str, str, str], Awaitable[None]] | None = None,
        on_audio_frame: Callable[[str, bytes], Awaitable[None]] | None = None,
        on_tts_state: Callable[[str, str, str], Awaitable[None]] | None = None,
    ):
        self.config = config or load_config()
        self._sessions: dict[str, XiaozhiSession] = {}
        self._writers: dict[str, asyncio.StreamWriter] = {}
        self._read_tasks: dict[str, asyncio.Task] = {}
        self._ping_tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._on_llm_reply = on_llm_reply
        self._on_audio_frame = on_audio_frame
        self._on_tts_state = on_tts_state

    async def connect(self, device_id: str, device_name: str = "") -> XiaozhiSession:
        if device_id in self._sessions:
            session = self._sessions[device_id]
            session.last_activity = time.time()
            return session

        ws_url = self.config.xiaozhi_ws_url
        host, port, path = _parse_ws_url(ws_url)

        reader, writer = await asyncio.open_connection(host, port)
        try:
            await self._ws_handshake(reader, writer, host, port, path, device_id)
        except Exception:
            writer.close()
            raise

        session = XiaozhiSession(
            device_id=device_id,
            connected=True,
            connected_at=time.time(),
            last_activity=time.time(),
        )

        async with self._lock:
            self._sessions[device_id] = session
            self._writers[device_id] = writer

        hello_response = await self._send_hello(reader, writer, device_id, device_name)
        session.session_id = hello_response.get("session_id", "")
        session.audio_params = hello_response.get("audio_params", {})
        logger.info("xiaozhi connected: device=%s session=%s", device_id, session.session_id)

        read_task = asyncio.create_task(self._read_loop(device_id, reader))
        ping_task = asyncio.create_task(self._ping_loop(device_id))
        self._read_tasks[device_id] = read_task
        self._ping_tasks[device_id] = ping_task

        return session

    async def disconnect(self, device_id: str) -> None:
        async with self._lock:
            self._sessions.pop(device_id, None)
            writer = self._writers.pop(device_id, None)
            read_task = self._read_tasks.pop(device_id, None)
            ping_task = self._ping_tasks.pop(device_id, None)

        if ping_task:
            ping_task.cancel()
        if read_task:
            read_task.cancel()
        if writer:
            try:
                await self._ws_send_close(writer)
                writer.close()
            except Exception:
                pass
        logger.info("xiaozhi disconnected: device=%s", device_id)

    async def send_text(self, device_id: str, text: str) -> bool:
        writer = self._writers.get(device_id)
        if not writer:
            return False
        session = self._sessions.get(device_id)
        if session:
            session.last_activity = time.time()
        msg = json.dumps({"type": "listen", "state": "detect", "text": text}, ensure_ascii=False)
        try:
            await self._ws_send_text(writer, msg)
            return True
        except Exception:
            logger.exception("xiaozhi send_text failed: device=%s", device_id)
            await self.disconnect(device_id)
            return False

    async def send_audio_start(self, device_id: str) -> bool:
        writer = self._writers.get(device_id)
        if not writer:
            return False
        msg = json.dumps({"type": "listen", "state": "start", "mode": "auto"}, ensure_ascii=False)
        try:
            await self._ws_send_text(writer, msg)
            return True
        except Exception:
            return False

    async def send_audio_stop(self, device_id: str) -> bool:
        writer = self._writers.get(device_id)
        if not writer:
            return False
        msg = json.dumps({"type": "listen", "state": "stop"}, ensure_ascii=False)
        try:
            await self._ws_send_text(writer, msg)
            return True
        except Exception:
            return False

    async def send_audio_frame(self, device_id: str, opus_data: bytes) -> bool:
        writer = self._writers.get(device_id)
        if not writer:
            return False
        try:
            await self._ws_send_binary(writer, opus_data)
            return True
        except Exception:
            return False

    async def send_abort(self, device_id: str) -> bool:
        writer = self._writers.get(device_id)
        if not writer:
            return False
        session = self._sessions.get(device_id)
        msg = json.dumps({"type": "abort", "session_id": session.session_id if session else ""}, ensure_ascii=False)
        try:
            await self._ws_send_text(writer, msg)
            return True
        except Exception:
            return False

    def get_session(self, device_id: str) -> XiaozhiSession | None:
        return self._sessions.get(device_id)

    def is_connected(self, device_id: str) -> bool:
        return device_id in self._sessions

    async def _ws_handshake(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
        host: str, port: int, path: str, device_id: str,
    ) -> None:
        ws_key = base64.b64encode(secrets.token_bytes(16)).decode("ascii")
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {ws_key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            f"Sec-WebSocket-Protocol: device-{device_id}\r\n"
            "\r\n"
        )
        writer.write(request.encode("ascii"))
        await writer.drain()

        status_line = await asyncio.wait_for(reader.readline(), timeout=5)
        if b"101" not in status_line:
            raise ConnectionError(f"WebSocket upgrade failed: {status_line.decode().strip()}")

        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=5)
            if line in {b"\r\n", b"\n", b""}:
                break

    async def _send_hello(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, device_id: str, device_name: str) -> dict[str, Any]:
        hello = {
            "type": "hello",
            "device_id": device_id,
            "device_name": device_name or f"cruiseCar-{device_id}",
            "device_mac": device_id,
            "token": "",
            "audio_params": {
                "format": "opus",
                "sample_rate": 16000,
                "channels": 1,
                "frame_duration": 60,
            },
        }
        await self._ws_send_text(writer, json.dumps(hello, ensure_ascii=False))
        opcode, payload = await self._ws_recv(reader)
        if opcode != 0x1:
            raise ConnectionError(f"Expected text frame for hello response, got opcode={opcode}")
        response = json.loads(payload.decode("utf-8"))
        if response.get("type") != "hello":
            raise ConnectionError(f"Expected hello response, got: {response}")
        return response

    async def _read_loop(self, device_id: str, reader: asyncio.StreamReader) -> None:
        writer = self._writers.get(device_id)
        try:
            while True:
                opcode, payload = await self._ws_recv(reader)
                if opcode == 0x8:
                    logger.info("xiaozhi close frame: device=%s", device_id)
                    break
                if opcode == 0x9:
                    if writer:
                        await self._ws_send_pong(writer, payload)
                    continue
                if opcode == 0x1:
                    await self._handle_text_message(device_id, payload)
                elif opcode == 0x2:
                    await self._handle_binary_message(device_id, payload)
        except asyncio.CancelledError:
            return
        except asyncio.IncompleteReadError:
            logger.info("xiaozhi connection closed: device=%s", device_id)
        except Exception:
            logger.exception("xiaozhi read_loop error: device=%s", device_id)
        finally:
            await self.disconnect(device_id)

    async def _handle_text_message(self, device_id: str, payload: bytes) -> None:
        try:
            msg = json.loads(payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        msg_type = msg.get("type", "")
        session = self._sessions.get(device_id)
        if session:
            session.last_activity = time.time()

        if msg_type == "llm":
            text = msg.get("text", "")
            session_id = msg.get("session_id", "")
            logger.info("xiaozhi llm reply: device=%s text=%s", device_id, text[:80])
            if self._on_llm_reply:
                await self._on_llm_reply(device_id, text, session_id)
        elif msg_type == "stt":
            text = msg.get("text", "")
            logger.info("xiaozhi stt: device=%s text=%s", device_id, text)
        elif msg_type == "tts":
            state = msg.get("state", "")
            text = msg.get("text", "")
            if self._on_tts_state:
                await self._on_tts_state(device_id, state, text)
        elif msg_type == "pong":
            pass
        else:
            logger.debug("xiaozhi msg: device=%s type=%s", device_id, msg_type)

    async def _handle_binary_message(self, device_id: str, payload: bytes) -> None:
        if self._on_audio_frame:
            await self._on_audio_frame(device_id, payload)

    async def _ping_loop(self, device_id: str) -> None:
        try:
            while True:
                await asyncio.sleep(20)
                writer = self._writers.get(device_id)
                if not writer:
                    break
                try:
                    await self._ws_send_text(writer, json.dumps({"type": "ping"}))
                except Exception:
                    break
        except asyncio.CancelledError:
            return

    async def _ws_recv(self, reader_or_writer: Any) -> tuple[int, bytes]:
        reader = reader_or_writer
        head = await reader.readexactly(2)
        opcode = head[0] & 0x0F
        masked = (head[1] & 0x80) != 0
        length = head[1] & 0x7F
        if length == 126:
            length = struct.unpack(">H", await reader.readexactly(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", await reader.readexactly(8))[0]
        if length > 10 * 1024 * 1024:
            raise ValueError(f"frame too large: {length}")
        mask = await reader.readexactly(4) if masked else b""
        payload = await reader.readexactly(length) if length else b""
        if masked:
            payload = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
        return opcode, payload

    @staticmethod
    async def _ws_send_text(writer: asyncio.StreamWriter, text: str) -> None:
        await _ws_send_frame(writer, 0x1, text.encode("utf-8"))

    @staticmethod
    async def _ws_send_binary(writer: asyncio.StreamWriter, data: bytes) -> None:
        await _ws_send_frame(writer, 0x2, data)

    @staticmethod
    async def _ws_send_pong(writer: asyncio.StreamWriter, payload: bytes) -> None:
        await _ws_send_frame(writer, 0xA, payload)

    @staticmethod
    async def _ws_send_close(writer: asyncio.StreamWriter) -> None:
        await _ws_send_frame(writer, 0x8, b"")


async def _ws_send_frame(writer: asyncio.StreamWriter, opcode: int, payload: bytes) -> None:
    header = bytearray([0x80 | opcode])
    length = len(payload)
    mask_key = secrets.token_bytes(4)
    if length < 126:
        header.append(0x80 | length)
    elif length <= 0xFFFF:
        header.append(0x80 | 126)
        header.extend(struct.pack(">H", length))
    else:
        header.append(0x80 | 127)
        header.extend(struct.pack(">Q", length))
    header.extend(mask_key)
    masked = bytes(byte ^ mask_key[i % 4] for i, byte in enumerate(payload))
    writer.write(bytes(header) + masked)
    await writer.drain()


def _parse_ws_url(url: str) -> tuple[str, int, str]:
    url = url.strip()
    if url.startswith("wss://"):
        scheme_end = 6
        default_port = 443
    elif url.startswith("ws://"):
        scheme_end = 5
        default_port = 8000
    else:
        raise ValueError(f"Invalid WebSocket URL scheme: {url}")
    rest = url[scheme_end:]
    path_idx = rest.find("/")
    if path_idx == -1:
        host_port = rest
        path = "/"
    else:
        host_port = rest[:path_idx]
        path = rest[path_idx:]
    if ":" in host_port:
        host, port_str = host_port.rsplit(":", 1)
        port = int(port_str)
    else:
        host = host_port
        port = default_port
    return host, port, path
