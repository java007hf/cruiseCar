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
from urllib import request as urlrequest
from urllib.parse import quote, urlsplit, urlunsplit

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


@dataclass
class XiaozhiEvent:
    seq: int
    device_id: str
    type: str
    created_at: float
    session_id: str = ""
    state: str = ""
    text: str = ""
    audio_b64: str = ""
    audio_size: int = 0


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
        self._events: dict[str, list[XiaozhiEvent]] = {}
        self._event_seq = 0
        self._lock = asyncio.Lock()
        self._on_llm_reply = on_llm_reply
        self._on_audio_frame = on_audio_frame
        self._on_tts_state = on_tts_state

    async def connect(self, device_id: str, device_name: str = "") -> XiaozhiSession:
        if device_id in self._sessions:
            session = self._sessions[device_id]
            session.last_activity = time.time()
            return session

        xiaozhi_device_id = _xiaozhi_device_id(device_id)
        ws_url = self.config.xiaozhi_ws_url
        token = self.config.xiaozhi_ws_token
        if not token and self.config.xiaozhi_ota_url:
            try:
                ota = await asyncio.to_thread(_fetch_ota_config_with_fallback, self.config.xiaozhi_ota_url, xiaozhi_device_id, device_name)
                token = str((ota.get("websocket") or {}).get("token", ""))
                logger.info(
                    "xiaozhi OTA token fetched: device=%s xiaozhi_device=%s token=%s",
                    device_id,
                    xiaozhi_device_id,
                    "yes" if token else "no",
                )
            except Exception:
                logger.exception("xiaozhi OTA token fetch failed: device=%s", device_id)
        ws_url = _with_xiaozhi_query(ws_url, xiaozhi_device_id, token)
        host, port, path = _parse_ws_url(ws_url)

        reader, writer = await asyncio.open_connection(host, port)
        try:
            await self._ws_handshake(reader, writer, host, port, path, xiaozhi_device_id, token)
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

        hello_response = await self._send_hello(reader, writer, device_id, xiaozhi_device_id, device_name, token)
        session.session_id = hello_response.get("session_id", "")
        session.audio_params = hello_response.get("audio_params", {})
        logger.info("xiaozhi connected: device=%s session=%s", device_id, session.session_id)
        self._append_event(device_id, "connected", session_id=session.session_id)

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
        self._append_event(device_id, "disconnected")

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
            logger.info("xiaozhi text listen sent: device=%s", device_id)
            self._append_event(device_id, "listen", state="detect", text=text)
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
            logger.info("xiaozhi audio start sent: device=%s", device_id)
            self._append_event(device_id, "listen", state="start")
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
            logger.info("xiaozhi audio stop sent: device=%s", device_id)
            self._append_event(device_id, "listen", state="stop")
            return True
        except Exception:
            return False

    async def send_audio_frame(self, device_id: str, opus_data: bytes) -> bool:
        writer = self._writers.get(device_id)
        if not writer:
            return False
        try:
            await self._ws_send_binary(writer, opus_data)
            self._append_event(device_id, "audio_uplink", audio_size=len(opus_data))
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

    def get_events(self, device_id: str, after_seq: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        events = self._events.get(device_id, [])
        selected = [event for event in events if event.seq > after_seq]
        return [
            {
                "seq": event.seq,
                "device_id": event.device_id,
                "type": event.type,
                "created_at": event.created_at,
                "session_id": event.session_id,
                "state": event.state,
                "text": event.text,
                "audio_b64": event.audio_b64,
                "audio_size": event.audio_size,
            }
            for event in selected[: max(1, min(500, limit))]
        ]

    def _append_event(
        self,
        device_id: str,
        event_type: str,
        session_id: str = "",
        state: str = "",
        text: str = "",
        audio_b64: str = "",
        audio_size: int = 0,
    ) -> None:
        self._event_seq += 1
        queue = self._events.setdefault(device_id, [])
        queue.append(
            XiaozhiEvent(
                seq=self._event_seq,
                device_id=device_id,
                type=event_type,
                created_at=time.time(),
                session_id=session_id,
                state=state,
                text=text,
                audio_b64=audio_b64,
                audio_size=audio_size,
            )
        )
        if len(queue) > 500:
            del queue[:-500]

    async def _ws_handshake(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
        host: str, port: int, path: str, device_id: str, token: str,
    ) -> None:
        ws_key = base64.b64encode(secrets.token_bytes(16)).decode("ascii")
        auth_header = f"Authorization: Bearer {token}\r\n" if token else ""
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {ws_key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            f"Device-Id: {device_id}\r\n"
            f"Client-Id: {device_id}\r\n"
            f"{auth_header}"
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

    async def _send_hello(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        device_id: str,
        xiaozhi_device_id: str,
        device_name: str,
        token: str,
    ) -> dict[str, Any]:
        hello = {
            "type": "hello",
            "device_id": xiaozhi_device_id,
            "device_name": device_name or f"cruiseCar-{device_id}",
            "device_mac": xiaozhi_device_id,
            "token": token,
            "features": {
                "mcp": False,
                "emoji": True,
            },
            "audio_params": {
                "format": "opus",
                "sample_rate": 16000,
                "channels": 1,
                "frame_duration": 60,
            },
        }
        await self._ws_send_text(writer, json.dumps(hello, ensure_ascii=False))
        deadline = time.time() + 8
        last_response: Any = None
        last_session_id = ""
        while time.time() < deadline:
            opcode, payload = await self._ws_recv(reader)
            if opcode == 0x2:
                await self._handle_binary_message(device_id, payload)
                continue
            if opcode != 0x1:
                raise ConnectionError(f"Expected text frame for hello response, got opcode={opcode}")
            try:
                response = json.loads(payload.decode("utf-8"))
            except json.JSONDecodeError as exc:
                snippet = payload[:200].decode("utf-8", errors="replace")
                raise ConnectionError(f"Invalid hello response from xiaozhi: {snippet}") from exc
            if response.get("type") == "hello":
                return response
            last_response = response
            last_session_id = str(response.get("session_id", last_session_id))
            logger.info("xiaozhi pre-hello message: device=%s msg=%s", device_id, response)
            await self._handle_text_message(device_id, payload)
        if last_response:
            logger.warning("xiaozhi did not send hello before data; treating connection as established: device=%s", device_id)
            return {"type": "hello", "session_id": last_session_id, "audio_params": hello["audio_params"]}
        raise ConnectionError("Expected hello response, got no xiaozhi messages")

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
            self._append_event(device_id, "llm", session_id=session_id, text=text)
            if self._on_llm_reply:
                await self._on_llm_reply(device_id, text, session_id)
        elif msg_type == "stt":
            text = msg.get("text", "")
            logger.info("xiaozhi stt: device=%s text=%s", device_id, text)
            self._append_event(device_id, "stt", session_id=msg.get("session_id", ""), text=text)
        elif msg_type == "tts":
            state = msg.get("state", "")
            text = msg.get("text", "")
            logger.info("xiaozhi tts: device=%s state=%s text=%s", device_id, state, text[:80])
            self._append_event(device_id, "tts", session_id=msg.get("session_id", ""), state=state, text=text)
            if self._on_tts_state:
                await self._on_tts_state(device_id, state, text)
        elif msg_type == "pong":
            pass
        else:
            logger.debug("xiaozhi msg: device=%s type=%s", device_id, msg_type)

    async def _handle_binary_message(self, device_id: str, payload: bytes) -> None:
        self._append_event(
            device_id,
            "audio_downlink",
            audio_b64=base64.b64encode(payload).decode("ascii"),
            audio_size=len(payload),
        )
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


def _with_xiaozhi_query(url: str, device_id: str, token: str) -> str:
    parts = urlsplit(url)
    query = parts.query
    additions = [
        ("device-id", device_id),
        ("client-id", device_id),
    ]
    if token:
        bearer = token if token.startswith("Bearer ") else f"Bearer {token}"
        additions.append(("authorization", bearer))
    suffix = "&".join(f"{quote(key)}={quote(value)}" for key, value in additions)
    query = f"{query}&{suffix}" if query else suffix
    return urlunsplit((parts.scheme, parts.netloc, parts.path or "/", query, parts.fragment))


def _fetch_ota_config(ota_url: str, device_id: str, device_name: str) -> dict[str, Any]:
    body = json.dumps(
        {
            "version": 0,
            "uuid": "",
            "application": {
                "name": "cruisecar-android-receiver",
                "version": "1.0.0",
                "compile_time": "2026-08-29 00:00:00",
                "idf_version": "4.4.3",
                "elf_sha256": "cruisecar",
            },
            "ota": {"label": "cruisecar-android-receiver"},
            "board": {
                "type": device_name or "CruiseCar",
                "ssid": "CruiseCar",
                "rssi": 0,
                "channel": 0,
                "ip": "127.0.0.1",
                "mac": device_id,
            },
            "flash_size": 0,
            "minimum_free_heap_size": 0,
            "mac_address": device_id,
            "chip_model_name": "android",
            "chip_info": {"model": 0, "cores": 0, "revision": 0, "features": 0},
            "partition_table": [{"label": "", "type": 0, "subtype": 0, "address": 0, "size": 0}],
        },
        ensure_ascii=False,
    ).encode("utf-8")
    req = urlrequest.Request(
        ota_url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Device-Id": device_id,
            "Client-Id": device_id,
        },
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_ota_config_with_fallback(ota_url: str, device_id: str, device_name: str) -> dict[str, Any]:
    candidates = [ota_url]
    if ":8003/" in ota_url:
        candidates.append(ota_url.replace(":8003/", ":8002/"))
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return _fetch_ota_config(candidate, device_id, device_name)
        except Exception as exc:
            last_error = exc
            logger.warning("xiaozhi OTA request failed: url=%s error=%s", candidate, exc)
    if last_error:
        raise last_error
    raise ValueError("no xiaozhi OTA URL configured")


def _xiaozhi_device_id(device_id: str) -> str:
    value = device_id.strip()
    parts = value.split(":")
    if len(parts) == 6 and all(len(part) == 2 and all(ch in "0123456789abcdefABCDEF" for ch in part) for part in parts):
        return value.upper()
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    # Locally administered unicast MAC, stable per CruiseCar device_id.
    octets = [0x02, digest[0], digest[1], digest[2], digest[3], digest[4]]
    return ":".join(f"{octet:02X}" for octet in octets)
